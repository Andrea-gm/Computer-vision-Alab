"""Sideways vial detector B: white-cap mask → PCA/minAreaRect → oriented AABB.

Finds opaque white screw-cap blobs, grows a nearby clear-body mask (edges +
mid-tone + depth holes), estimates the vial principal axis with PCA (fallback
minAreaRect), builds an oriented rectangle near expected H/diameter aspect,
then returns the axis-aligned cover. Depth rejects walls and bright slabs.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

NAME = "B_rotated_white_cap"

_ROOT = Path(__file__).resolve().parents[2]
_DIMS_PATH = _ROOT / "vial_3d_dims.json"

_DEFAULT_TOTAL_H_MM = 72.69
_DEFAULT_LID_DIAM_MM = 54.22
_EXPECTED_ASPECT = _DEFAULT_TOTAL_H_MM / _DEFAULT_LID_DIAM_MM  # ≈ 1.34

# Sideways AABB h/w band (slightly wider than live 0.40–0.71 for synth variance).
_SIDEWAYS_HW_MIN = 0.34
_SIDEWAYS_HW_MAX = 0.80

_MIN_CAP_AREA = 35
_MAX_CAP_FRAC = 0.18
_SIDEWAYS_AXIS_COS = 0.55
_CLEAR_FAR_DELTA_MM = 180.0
_MIN_DEPTH_MM = 80.0
_MAX_DEPTH_MM = 5000.0


def _load_expected_aspect() -> float:
    if not _DIMS_PATH.is_file():
        return _EXPECTED_ASPECT
    try:
        data = json.loads(_DIMS_PATH.read_text(encoding="utf-8"))
        h = float(data.get("total_height_mm", _DEFAULT_TOTAL_H_MM))
        d = float(data.get("lid_diameter_mm", _DEFAULT_LID_DIAM_MM))
        if h > 1.0 and d > 1.0:
            return h / d
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return _EXPECTED_ASPECT


def _align_depth(depth_mm: np.ndarray | None, w: int, h: int) -> np.ndarray | None:
    if depth_mm is None:
        return None
    d = np.asarray(depth_mm, dtype=np.float32)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
    return d


def _white_cap_mask(color_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    best = ((val >= 200) & (sat <= 55)).astype(np.uint8) * 255
    best_frac = float(best.mean() / 255.0)
    if not (0.0008 <= best_frac <= 0.22):
        for vmin in (185, 170, 210, 220, 230):
            trial = ((val >= vmin) & (sat <= 55)).astype(np.uint8) * 255
            frac = float(trial.mean() / 255.0)
            if 0.0008 <= frac <= 0.22 and abs(frac - 0.05) < abs(best_frac - 0.05):
                best, best_frac = trial, frac
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    best = cv2.morphologyEx(best, cv2.MORPH_OPEN, k)
    best = cv2.morphologyEx(best, cv2.MORPH_CLOSE, k)
    # Keep end-cap lids as compact blobs (vertical close helps tall end-caps).
    best = cv2.morphologyEx(
        best, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    )
    return best


def _cap_depth_mm(depth: np.ndarray | None, x: int, y: int, bw: int, bh: int) -> float | None:
    if depth is None:
        return None
    cx, cy = x + bw // 2, y + bh // 2
    r = max(2, min(bw, bh) // 3)
    patch = depth[max(0, cy - r) : cy + r + 1, max(0, cx - r) : cx + r + 1]
    valid = patch[(patch > _MIN_DEPTH_MM) & (patch < _MAX_DEPTH_MM)]
    if valid.size < 5:
        return None
    return float(np.median(valid))


def _side_body_score(
    color_bgr: np.ndarray,
    depth: np.ndarray | None,
    cap_box: tuple[int, int, int, int],
    side: str,
    cap_z: float | None,
) -> tuple[float, int]:
    """Score clear-cup evidence to the left/right of a cap. Returns (score, extent_px)."""
    h, w = color_bgr.shape[:2]
    x, y, bw, bh = cap_box
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gy_abs = np.abs(gy)

    max_reach = int(max(bw, bh) * 2.4)
    min_cup = max(8, int(max(bw, bh) * 0.35))
    inset = max(1, bh // 6)
    y0 = y + inset
    y1 = y + bh - inset
    if y1 <= y0:
        y0, y1 = y, y + bh

    step = -1 if side == "left" else 1
    start = x - 1 if side == "left" else x + bw
    extent = 0
    gap = 0
    hole_hits = 0
    edge_hits = 0
    tone_hits = 0
    left_clear_zone = 0  # columns after leaving invalid-depth cup

    for i in range(max_reach):
        col = start + i * step
        if col < 0 or col >= w:
            break
        band_v = val[y0:y1, col]
        band_s = sat[y0:y1, col]
        if band_v.size == 0:
            break
        white_frac = float(((band_v >= 190) & (band_s <= 55)).mean())
        if white_frac >= 0.55 and extent >= min_cup:
            break
        if white_frac >= 0.75:
            break

        tone = float(((band_v >= 35) & (band_v <= 160) & (band_s <= 90)).mean())
        edge = 0.0
        if y1 - y0 >= 4:
            top = float(gy_abs[y0 : y0 + 2, col].mean())
            bot = float(gy_abs[y1 - 2 : y1, col].mean())
            edge = 0.5 * (top + bot)

        invalid = 0.0
        if depth is not None:
            dcol = depth[y0:y1, col]
            invalid = float((dcol <= _MIN_DEPTH_MM).mean())

        in_clear = invalid >= 0.25
        strong_edge = edge >= 28.0
        ok = in_clear or strong_edge or (tone >= 0.55 and extent > 0 and hole_hits >= 2 and invalid >= 0.10)

        if ok:
            extent += 1
            gap = 0
            if in_clear:
                hole_hits += 1
                left_clear_zone = 0
            else:
                if hole_hits >= 3:
                    left_clear_zone += 1
            if strong_edge:
                edge_hits += 1
            if tone >= 0.45:
                tone_hits += 1
            # Desk beyond the cup: solid depth after a clear run → stop.
            if left_clear_zone >= 2 and extent >= min_cup:
                break
        else:
            gap += 1
            if extent > 0 and gap >= 2:
                break
            if extent == 0 and gap >= 2:
                break

    if extent < min_cup:
        return 0.0, extent

    if hole_hits < max(3, min_cup // 2) and edge_hits < max(4, min_cup // 2):
        return 0.0, extent

    score = 1.0 * extent + 3.0 * hole_hits + 1.2 * edge_hits + 0.5 * tone_hits
    return float(score), extent


def _grow_combined_mask(
    color_bgr: np.ndarray,
    one_cap: np.ndarray,
    cap_box: tuple[int, int, int, int],
    side: str,
    extent: int,
) -> np.ndarray:
    """Paint cap + rectangular body band on the chosen side for PCA/minAreaRect."""
    h, w = color_bgr.shape[:2]
    x, y, bw, bh = cap_box
    mask = one_cap.copy()
    pad_y = max(1, bh // 10)
    y0 = max(0, y - pad_y)
    y1 = min(h, y + bh + pad_y)
    if side == "left":
        x0 = max(0, x - extent)
        x1 = x
    else:
        x0 = x + bw
        x1 = min(w, x + bw + extent)
    if x1 > x0:
        # Keep only non-white mid pixels inside the band (plus edges).
        hsv = cv2.cvtColor(color_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        gray = cv2.cvtColor(color_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 90)
        body = (((val < 185) | (sat > 50)) & (val >= 30) & (val <= 175)).astype(np.uint8) * 255
        body = cv2.bitwise_or(body, (edges > 0).astype(np.uint8) * 255)
        # Fill the geometric band lightly so PCA sees the full length even if
        # clear plastic is sparse — but only where not bright-white.
        geo = np.zeros_like(body)
        geo[:, :] = 255
        white = ((val >= 190) & (sat <= 55)).astype(np.uint8) * 255
        geo = cv2.bitwise_and(geo, cv2.bitwise_not(white))
        band = cv2.bitwise_or(body, geo)
        mask[y0:y1, x0:x1] = cv2.bitwise_or(mask[y0:y1, x0:x1], band)
    return mask


def _pca_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    if points.shape[0] < 12:
        return None
    mean = points.mean(axis=0)
    cov = np.cov(points - mean, rowvar=False)
    if not np.isfinite(cov).all():
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    axis = eigvecs[:, 0]
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return None
    return mean, axis / n, float(eigvals[0] / max(eigvals[1], 1e-9))


def _min_area_axis(contour: np.ndarray):
    if contour is None or len(contour) < 5:
        return None
    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), _ang = rect
    if rw < 1.0 or rh < 1.0:
        return None
    box = cv2.boxPoints(rect)
    d01 = box[1] - box[0]
    d12 = box[2] - box[1]
    if np.linalg.norm(d01) >= np.linalg.norm(d12):
        long_vec, long_len, short_len = d01, float(np.linalg.norm(d01)), float(np.linalg.norm(d12))
    else:
        long_vec, long_len, short_len = d12, float(np.linalg.norm(d12)), float(np.linalg.norm(d01))
    if long_len < 1e-6:
        return None
    return (
        np.array([cx, cy], dtype=np.float64),
        long_vec / long_len,
        long_len / max(short_len, 1e-6),
        long_len,
        short_len,
        rect,
    )


def _oriented_rect_to_aabb(
    center: np.ndarray,
    long_axis: np.ndarray,
    long_len: float,
    short_len: float,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    long_u = long_axis / max(float(np.linalg.norm(long_axis)), 1e-9)
    short_u = np.array([-long_u[1], long_u[0]], dtype=np.float64)
    hl, hs = 0.5 * long_len, 0.5 * short_len
    corners = np.stack(
        [
            center + hl * long_u + hs * short_u,
            center + hl * long_u - hs * short_u,
            center - hl * long_u + hs * short_u,
            center - hl * long_u - hs * short_u,
        ],
        axis=0,
    )
    x0 = int(np.floor(corners[:, 0].min()))
    y0 = int(np.floor(corners[:, 1].min()))
    x1 = int(np.ceil(corners[:, 0].max()))
    y1 = int(np.ceil(corners[:, 1].max()))
    x0 = max(0, min(frame_w - 1, x0))
    y0 = max(0, min(frame_h - 1, y0))
    x1 = max(x0 + 1, min(frame_w, x1))
    y1 = max(y0 + 1, min(frame_h, y1))
    return x0, y0, x1 - x0, y1 - y0


def _depth_rejects(
    depth: np.ndarray | None,
    box: tuple[int, int, int, int],
    cap_box: tuple[int, int, int, int],
    side: str,
    frame_w: int,
    frame_h: int,
) -> bool:
    """True if box looks like a wall / solid slab (no clear-cup depth)."""
    x, y, bw, bh = box
    if bw * bh >= 0.25 * frame_w * frame_h:
        return True
    if depth is None:
        return False

    # Require clear-cup holes on the body side of the cap.
    cx, cy, cw, ch = cap_box
    inset = max(1, ch // 6)
    y0, y1 = cy + inset, cy + ch - inset
    if side == "left":
        x0, x1 = max(0, cx - max(8, bw // 2)), cx
    else:
        x0, x1 = cx + cw, min(frame_w, cx + cw + max(8, bw // 2))
    if x1 <= x0 or y1 <= y0:
        return True
    body = depth[y0:y1, x0:x1]
    if body.size < 8:
        return True
    cap_z = _cap_depth_mm(depth, cx, cy, cw, ch)
    invalid = float((body <= _MIN_DEPTH_MM).mean())
    # Opaque slabs keep near-depth; clear cups are mostly invalid.
    near = 0.0
    if cap_z is not None:
        near = float(((body > _MIN_DEPTH_MM) & (np.abs(body - cap_z) < 100)).mean())
    if invalid < 0.15:
        return True
    if near > 0.50:
        return True

    # Huge flat wall patch.
    patch = depth[y : y + bh, x : x + bw]
    valid = patch[(patch > _MIN_DEPTH_MM) & (patch < _MAX_DEPTH_MM)]
    if valid.size >= 40 and bw * bh >= 0.10 * frame_w * frame_h:
        if float(np.std(valid)) < 20.0:
            return True
    return False


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter <= 0:
        return 0.0
    return float(inter / max(aw * ah + bw * bh - inter, 1))


def _nms(boxes: list[tuple[int, int, int, int]], scores: list[float], thr: float = 0.4):
    if not boxes:
        return []
    order = np.argsort(np.asarray(scores))[::-1]
    kept: list[tuple[int, int, int, int]] = []
    for idx in order:
        box = boxes[int(idx)]
        if any(_iou(box, k) >= thr for k in kept):
            continue
        kept.append(box)
    return kept


def detect_sideways(
    color_bgr,
    depth_mm,
    intrinsics=None,
) -> list[tuple[int, int, int, int]]:
    """Detect sideways vials; return AABBs covering oriented vial rectangles."""
    del intrinsics
    if color_bgr is None or getattr(color_bgr, "size", 0) == 0:
        return []
    frame_h, frame_w = color_bgr.shape[:2]
    depth = _align_depth(depth_mm, frame_w, frame_h)
    expected_aspect = _load_expected_aspect()

    cap_mask = _white_cap_mask(color_bgr)
    contours, _ = cv2.findContours(cap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_side = max(8, int(min(frame_w, frame_h) * 0.012))

    boxes: list[tuple[int, int, int, int]] = []
    scores: list[float] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < _MIN_CAP_AREA or area > _MAX_CAP_FRAC * frame_w * frame_h:
            continue
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if max(cw, ch) < min_side:
            continue
        if cw > 0.55 * frame_w or ch > 0.55 * frame_h:
            continue
        # End-cap / compact lid: reject huge wide wall strips.
        cap_aspect = cw / max(ch, 1)
        if cap_aspect > 3.5 and cw > 60:
            continue
        # Top-down-ish large square lids without horizontal body are handled later.
        hull = cv2.convexHull(contour)
        hull_a = float(cv2.contourArea(hull)) or 1.0
        solidity = float(area / hull_a)
        if solidity < 0.55:
            continue

        one_cap = np.zeros_like(cap_mask)
        cv2.drawContours(one_cap, [contour], -1, 255, thickness=-1)
        cap_z = _cap_depth_mm(depth, cx, cy, cw, ch)

        left_s, left_e = _side_body_score(color_bgr, depth, (cx, cy, cw, ch), "left", cap_z)
        right_s, right_e = _side_body_score(color_bgr, depth, (cx, cy, cw, ch), "right", cap_z)
        if left_s <= 0 and right_s <= 0:
            continue
        if right_s >= left_s:
            side, extent, side_score = "right", right_e, right_s
        else:
            side, extent, side_score = "left", left_e, left_s

        combined = _grow_combined_mask(color_bgr, one_cap, (cx, cy, cw, ch), side, extent)
        ys, xs = np.where(combined > 0)
        if xs.size < 20:
            continue
        pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])

        comb_contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not comb_contours:
            continue
        big = max(comb_contours, key=cv2.contourArea)
        mar = _min_area_axis(big)
        pca = _pca_axis(pts)

        # Axis from PCA if horizontal; else minAreaRect; else hard horizontal.
        axis = None
        mean = None
        long_len = float(cw + extent)
        short_len = float(max(ch, int(cw * 1.2)))  # end-cap: height ≈ Ø

        if pca is not None and abs(float(pca[1][0])) >= _SIDEWAYS_AXIS_COS:
            mean, axis, _ratio = pca
            proj = (pts - mean) @ axis
            long_len = float(proj.max() - proj.min())
            # Short extent from perpendicular projection.
            perp = np.array([-axis[1], axis[0]])
            short_len = float(np.ptp((pts - mean) @ perp))
        if axis is None and mar is not None and abs(float(mar[1][0])) >= _SIDEWAYS_AXIS_COS:
            mean, axis, _asp, long_len, short_len, _rect = mar
        if axis is None:
            axis = np.array([1.0 if side == "right" else -1.0, 0.0], dtype=np.float64)
            mean = np.array([cx + cw * 0.5, cy + ch * 0.5], dtype=np.float64)
            long_len = float(cw + extent)
            short_len = float(ch)

        axis_cos = abs(float(axis[0])) / max(float(np.linalg.norm(axis)), 1e-9)
        if axis_cos < _SIDEWAYS_AXIS_COS:
            continue

        # Size: observed cap+cup span, snapped into H/diam–ish band.
        short_len = max(float(ch), short_len * 0.85, 8.0)
        obs_long = float(cw + extent)
        target = short_len * expected_aspect
        long_len = max(obs_long, target * 0.90)
        # Keep AABB h/w inside sideways band: long <= short / hw_min.
        max_long = short_len / max(_SIDEWAYS_HW_MIN, 1e-6)
        long_len = min(long_len, max_long, short_len * 2.55)

        # Orient axis from cap into the body; center on the grown segment.
        cap_c = np.array([cx + cw * 0.5, cy + ch * 0.5], dtype=np.float64)
        want = np.array([1.0 if side == "right" else -1.0, 0.0], dtype=np.float64)
        if float(np.dot(axis, want)) < 0:
            axis = -axis
        # Near-horizontal: place AABB directly from cap+extent (stable).
        if abs(float(axis[0])) >= 0.92:
            pad = max(1, int(0.04 * short_len))
            if side == "right":
                aabb = (
                    max(0, cx - pad),
                    max(0, cy - pad),
                    min(frame_w - max(0, cx - pad), int(round(long_len)) + 2 * pad),
                    min(frame_h - max(0, cy - pad), int(round(short_len)) + 2 * pad),
                )
            else:
                x0 = max(0, int(round(cx + cw - long_len)) - pad)
                aabb = (
                    x0,
                    max(0, cy - pad),
                    min(frame_w - x0, int(round(long_len)) + 2 * pad),
                    min(frame_h - max(0, cy - pad), int(round(short_len)) + 2 * pad),
                )
        else:
            center = cap_c + axis * (0.5 * long_len - 0.5 * float(cw))
            aabb = _oriented_rect_to_aabb(
                center, axis, long_len, short_len, frame_w, frame_h
            )
        ax, ay, aw, ah = aabb
        if aw < 16 or ah < 10:
            continue
        hw = ah / max(aw, 1)
        if not (_SIDEWAYS_HW_MIN <= hw <= _SIDEWAYS_HW_MAX):
            continue

        if _depth_rejects(depth, aabb, (cx, cy, cw, ch), side, frame_w, frame_h):
            continue

        aspect = long_len / max(short_len, 1.0)
        aspect_score = max(0.0, 1.0 - abs(aspect - expected_aspect) / max(expected_aspect, 1e-6))
        score = 0.40 * axis_cos + 0.25 * aspect_score + 0.15 * solidity + 0.20 * min(
            1.0, side_score / 200.0
        )
        boxes.append(aabb)
        scores.append(score)

    return _nms(boxes, scores)


# ---------------------------------------------------------------------------
# Synthetic self-test
# ---------------------------------------------------------------------------


def _synth_sideways(
    width: int = 320,
    height: int = 240,
    lid_left: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Paint a sideways clear vial with opaque white cap + synthetic depth."""
    rng = np.random.default_rng(7)
    color = np.full((height, width, 3), 88, dtype=np.uint8)
    color = np.clip(
        color.astype(np.int16) + rng.integers(-10, 11, size=color.shape, dtype=np.int16),
        0,
        255,
    ).astype(np.uint8)

    vial_h = 40
    vial_w = int(round(vial_h / 0.55))  # harness-like h/w ≈ 0.55
    x0 = width // 2 - vial_w // 2
    y0 = height // 2 - vial_h // 2
    lid_w = max(10, int(vial_w * 0.22))
    body_w = vial_w - lid_w

    if lid_left:
        lx, ly, lw, lh = x0, y0, lid_w, vial_h
        bx, by, bw, bh = x0 + lid_w, y0, body_w, vial_h
    else:
        lx, ly, lw, lh = x0 + body_w, y0, lid_w, vial_h
        bx, by, bw, bh = x0, y0, body_w, vial_h

    color[by : by + bh, bx : bx + bw] = (70, 68, 65)
    color[by, bx : bx + bw] = (35, 35, 35)
    color[by + bh - 1, bx : bx + bw] = (35, 35, 35)
    color[ly : ly + lh, lx : lx + lw] = (245, 245, 245)

    depth = np.full((height, width), 1100.0, dtype=np.float32)
    depth[ly : ly + lh, lx : lx + lw] = 520.0
    depth[by : by + bh, bx : bx + bw] = 0.0

    return color, depth, (x0, y0, vial_w, vial_h)


if __name__ == "__main__":
    img, depth, gt = _synth_sideways(lid_left=True)
    dets = detect_sideways(img, depth, intrinsics=None)
    print(f"NAME={NAME}")
    print(f"gt={gt}")
    print(f"detections={dets}")
    if not dets:
        print("best_iou=0.000")
        raise SystemExit(1)
    best = max(dets, key=lambda d: _iou(d, gt))
    print(f"best_iou={_iou(best, gt):.3f}")
    if _iou(best, gt) < 0.5:
        raise SystemExit(1)

    img2, depth2, gt2 = _synth_sideways(lid_left=False)
    dets2 = detect_sideways(img2, depth2)
    print(f"gt_right={gt2} detections={dets2}")
    if not dets2 or max(_iou(d, gt2) for d in dets2) < 0.5:
        raise SystemExit(1)

    wall = np.full((240, 320, 3), 200, dtype=np.uint8)
    wall_d = np.full((240, 320), 800.0, dtype=np.float32)
    empty = detect_sideways(wall, wall_d)
    print(f"wall_detections={empty}")
    if empty:
        raise SystemExit(1)

    # Harness-like opaque white slab should not fire.
    slab = np.full((240, 320, 3), 90, dtype=np.uint8)
    slab[80:130, 40:110] = (230, 230, 230)
    slab_d = np.full((240, 320), 1100.0, dtype=np.float32)
    slab_d[80:130, 40:110] = 600.0
    print(f"slab_detections={detect_sideways(slab, slab_d)}")
    if detect_sideways(slab, slab_d):
        raise SystemExit(1)

    print("self-test OK")
