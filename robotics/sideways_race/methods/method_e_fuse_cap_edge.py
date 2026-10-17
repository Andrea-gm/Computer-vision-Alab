"""Sideways vial detector E: fuse white-cap morph elongate + elongated Canny.

Two proposal streams are clustered by IoU; survivors must show a white end-cap,
an elongated body, and vial-scale at measured Z.
"""

from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np

NAME = "E_fuse_cap_edge"

# Physical priors (mm) — prefer vial_3d_dims.json when present.
_DEFAULT_TOTAL_H_MM = 72.69
_DEFAULT_LID_D_MM = 54.22
_SIDEWAYS_ASPECT_MIN = 0.38
_SIDEWAYS_ASPECT_MAX = 0.90  # slightly looser than live for recall in race
_LENGTH_OVER_DIAM = 1.65  # expected body span vs short axis (Ø)
_MIN_IOU_CLUSTER = 0.25
_MIN_SURFACE_MM = 80.0
_DEFAULT_FX = 600.0


def _load_dims() -> tuple[float, float]:
    root = Path(__file__).resolve().parents[2]
    path = root / "vial_3d_dims.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            h = float(data.get("total_height_mm", _DEFAULT_TOTAL_H_MM))
            d = float(data.get("lid_diameter_mm", _DEFAULT_LID_D_MM))
            return h, d
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return _DEFAULT_TOTAL_H_MM, _DEFAULT_LID_D_MM


TOTAL_H_MM, LID_D_MM = _load_dims()


def _focal_lengths(intrinsics) -> tuple[float, float] | None:
    """Accept manifest dictionaries as well as (fx, fy, cx, cy) sequences."""
    if intrinsics is None:
        return None
    try:
        if isinstance(intrinsics, dict):
            fx = float(intrinsics["fx"])
            fy = float(intrinsics["fy"])
        else:
            fx, fy = float(intrinsics[0]), float(intrinsics[1])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if fx <= 1.0 or fy <= 1.0:
        return None
    return fx, fy


def _clip_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = (int(v) for v in box)
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _union_box(boxes) -> tuple[int, int, int, int]:
    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    x2 = [b[0] + b[2] for b in boxes]
    y2 = [b[1] + b[3] for b in boxes]
    x0, y0 = min(xs), min(ys)
    return int(x0), int(y0), int(max(x2) - x0), int(max(y2) - y0)


def _white_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] >= 175) & (hsv[:, :, 1] <= 55)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def _proposals_cap_morph(color_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """(1) White-cap blobs elongated horizontally via morphology + span grow."""
    h, w = color_bgr.shape[:2]
    white = _white_mask(color_bgr)
    # Prefer compact lid-sized blobs before elongating.
    # Sideways end-caps are tall vertical strips (bh > bw) — allow that.
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seed = np.zeros_like(white)
    lids: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = float(cv2.contourArea(cnt))
        if area < 40 or bw < 5 or bh < 6:
            continue
        # Cap-like: not a huge wall fill.
        if bw > w * 0.40 or bh > h * 0.55:
            continue
        if area / max(bw * bh, 1) < 0.35:
            continue
        cv2.drawContours(seed, [cnt], -1, 255, -1)
        lids.append((x, y, bw, bh))

    if not np.any(seed):
        seed = white
        for cnt in contours:
            lids.append(cv2.boundingRect(cnt))

    out: list[tuple[int, int, int, int]] = []

    # Explicit horizontal span from each lid (~total H / Ø).
    # Lid-on-left → grow right; lid-on-right → grow left (plus a mild both-ways).
    for lx, ly, lw, lh in lids:
        diam = max(lh, int(lw * 1.2))
        span = max(int(diam * _LENGTH_OVER_DIAM), lw + 24)
        y0 = max(0, ly - 2)
        hh = diam + 4
        # Body to the right of lid (white on left end).
        out.append(_clip_box((lx, y0, span, hh), w, h))
        # Body to the left of lid (white on right end).
        out.append(_clip_box((max(0, lx + lw - span), y0, span, hh), w, h))

    # Morph elongate: kernel width tracks expected body length from lid height.
    if lids:
        ref_h = int(np.median([lh for _, _, _, lh in lids]))
    else:
        ref_h = max(16, h // 10)
    kx = max(21, int(ref_h * _LENGTH_OVER_DIAM))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3))
    elongated = cv2.dilate(seed, kernel, iterations=2)
    elongated = cv2.morphologyEx(
        elongated,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kx, max(5, ref_h // 6))),
        iterations=1,
    )

    contours, _ = cv2.findContours(elongated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 20 or bh < 8:
            continue
        if not _is_elongated((x, y, bw, bh)):
            continue
        out.append(_clip_box((x, y, bw, bh), w, h))
    return out


def _proposals_canny_elongate(color_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """(2) Canny contours kept when elongated (sideways capsule-ish)."""
    h, w = color_bgr.shape[:2]
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 25, 80)
    # Soft |Gy| cue for clear-plastic top/bottom walls.
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    gy_mask = (np.abs(gy) > 12).astype(np.uint8) * 255
    edges = cv2.bitwise_or(edges, gy_mask)
    # Link horizontal wall edges of a fallen vial.
    kx = max(15, int(min(w, h) * 0.07))
    linked = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3)),
        iterations=2,
    )
    linked = cv2.dilate(
        linked, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)), iterations=1
    )
    # Fill capsule interior lightly so contour AABB covers lid+cup.
    linked = cv2.morphologyEx(
        linked,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )

    out: list[tuple[int, int, int, int]] = []
    contours, _ = cv2.findContours(linked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 60:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 24 or bh < 8:
            continue
        if bw * bh > 0.35 * w * h:
            continue
        if not _is_elongated((x, y, bw, bh)):
            continue
        # Prefer elongated rectangles (not random edge soup).
        rect = cv2.minAreaRect(cnt)
        (rw, rh) = rect[1]
        long_s, short_s = max(rw, rh), max(1.0, min(rw, rh))
        if long_s / short_s < 1.25:
            continue
        out.append(_clip_box((x, y, bw, bh), w, h))
    return out


def _cluster_by_iou(
    boxes: list[tuple[int, int, int, int]],
    scores: list[float] | None = None,
    iou_thr: float = _MIN_IOU_CLUSTER,
) -> list[tuple[int, int, int, int]]:
    """Greedy IoU clustering → keep best-scoring member (not loose union)."""
    if not boxes:
        return []
    if scores is None:
        scores = [1.0] * len(boxes)
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    used = [False] * len(boxes)
    clusters: list[tuple[int, int, int, int]] = []
    for idx in order:
        if used[idx]:
            continue
        group_idx = [idx]
        used[idx] = True
        for j in order:
            if used[j]:
                continue
            if any(_iou(boxes[g], boxes[j]) >= iou_thr for g in group_idx):
                used[j] = True
                group_idx.append(j)
        # Keep best-scoring member. Union only near-tied high scorers so a
        # weak centered proposal cannot dilute a clear white-end box.
        best_j = max(group_idx, key=lambda j: scores[j])
        best_s = scores[best_j]
        if len(group_idx) >= 2 and best_s >= 0.12:
            good = [
                boxes[j]
                for j in group_idx
                if scores[j] >= max(0.20, best_s - 0.08)
            ]
            if len(good) >= 2:
                u = _union_box(good)
                _, _, bw, bh = u
                _, _, bw0, bh0 = boxes[best_j]
                if (
                    _is_elongated(u)
                    and bw * bh <= 1.8 * bw0 * bh0
                    and bw <= int(bw0 * 1.35)
                ):
                    clusters.append(u)
                    continue
        clusters.append(boxes[best_j])
    return clusters


def _white_end_score(color_bgr: np.ndarray, box) -> float:
    h, w = color_bgr.shape[:2]
    x, y, bw, bh = _clip_box(box, w, h)
    if bw < 12 or bh < 8:
        return 0.0
    hsv = cv2.cvtColor(color_bgr[y : y + bh, x : x + bw], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 150)
    band = max(3, int(bw * 0.34))
    if bw <= band * 2:
        return 0.0
    left = float(white[:, :band].mean())
    right = float(white[:, -band:].mean())
    mid = float(white[:, band : bw - band].mean())
    end = max(left, right)
    if end < 0.25 or mid > end - 0.06:
        return 0.0
    return float(max(0.0, min(1.0, end - mid + 0.1)))


def _is_elongated(box) -> bool:
    _, _, bw, bh = box
    if bw < 16 or bh < 8:
        return False
    aspect = bh / max(bw, 1)
    return _SIDEWAYS_ASPECT_MIN <= aspect <= _SIDEWAYS_ASPECT_MAX


def _median_z_mm(depth_mm: np.ndarray, color_bgr: np.ndarray, box) -> float | None:
    """Prefer opaque white end depth; fall back to near median in box."""
    h, w = depth_mm.shape[:2]
    x, y, bw, bh = _clip_box(box, w, h)
    roi_d = depth_mm[y : y + bh, x : x + bw]
    hsv = cv2.cvtColor(color_bgr[y : y + bh, x : x + bw], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 60) & (hsv[:, :, 2] >= 170)
    band = max(3, int(bw * 0.34))
    left_w = float(white[:, :band].mean()) if band < bw else 0.0
    right_w = float(white[:, -band:].mean()) if band < bw else 0.0
    if left_w >= right_w and left_w >= 0.2:
        patch = roi_d[:, :band]
    elif right_w > left_w and right_w >= 0.2:
        patch = roi_d[:, -band:]
    else:
        patch = roi_d
    valid = patch[(patch > _MIN_SURFACE_MM) & (patch < 5000)]
    if valid.size < 8:
        valid = roi_d[(roi_d > _MIN_SURFACE_MM) & (roi_d < 5000)]
    if valid.size < 4:
        return None
    return float(np.median(valid))


def _scale_ok_at_z(
    box,
    z_mm: float | None,
    intrinsics,
) -> bool:
    """Vial-scale gate at Z: long≈H, short≈Ø (generous race tol)."""
    _, _, bw, bh = box
    if z_mm is None or z_mm <= _MIN_SURFACE_MM:
        # No depth: accept elongated geometry only (synth may omit Z).
        return True
    focal = _focal_lengths(intrinsics)
    fx, fy = focal if focal is not None else (_DEFAULT_FX, _DEFAULT_FX)
    width_mm = bw * z_mm / fx
    height_mm = bh * z_mm / fy
    long_mm = max(width_mm, height_mm)
    short_mm = min(width_mm, height_mm)
    # long ≈ total H, short ≈ lid Ø; keep walls out.
    return (
        0.45 * TOTAL_H_MM <= long_mm <= 2.2 * TOTAL_H_MM
        and 0.35 * LID_D_MM <= short_mm <= 2.0 * LID_D_MM
    )


def _body_support(depth_mm: np.ndarray, color_bgr: np.ndarray, box, z_mm) -> float:
    """Lid@Z + clear-cup holes; penalize solid far table/wall fill."""
    if z_mm is None:
        return 0.5
    h, w = depth_mm.shape[:2]
    x, y, bw, bh = _clip_box(box, w, h)
    roi = depth_mm[y : y + bh, x : x + bw]
    near = float(((roi > _MIN_SURFACE_MM) & (roi < z_mm + 220.0)).mean())
    invalid = float((roi <= 1.0).mean())
    far = float((roi > z_mm + 220.0).mean())
    # Clear cup: invalid/sparse far next to lid; solid far background is bad.
    support = near + 0.75 * invalid - 0.65 * far
    return max(0.0, min(1.0, support))


def _clear_cup_fraction(
    depth_mm: np.ndarray, color_bgr: np.ndarray, box
) -> float:
    """Fraction of non-white proposal pixels with invalid see-through depth."""
    h, w = depth_mm.shape[:2]
    x, y, bw, bh = _clip_box(box, w, h)
    roi_d = depth_mm[y : y + bh, x : x + bw]
    hsv = cv2.cvtColor(color_bgr[y : y + bh, x : x + bw], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 60) & (hsv[:, :, 2] >= 170)
    body = ~white
    if np.count_nonzero(body) < 8:
        return 0.0
    invalid = ~np.isfinite(roi_d) | (roi_d <= _MIN_SURFACE_MM)
    return float(np.count_nonzero(invalid & body) / np.count_nonzero(body))


def _rank(color_bgr, box, z_mm, intrinsics, depth_mm=None) -> float:
    end = _white_end_score(color_bgr, box)
    _, _, bw, bh = box
    elong = min(bw / max(bh, 1), 3.0) / 3.0
    scale = 0.5
    focal = _focal_lengths(intrinsics)
    if z_mm is not None and focal is not None:
        fx, fy = focal
        long_mm = max(bw * z_mm / fx, bh * z_mm / fy)
        short_mm = min(bw * z_mm / fx, bh * z_mm / fy)
        scale = max(0.0, 1.0 - abs(long_mm - TOTAL_H_MM) / 90.0) * 0.5 + max(
            0.0, 1.0 - abs(short_mm - LID_D_MM) / 70.0
        ) * 0.5
    body = 0.5
    if depth_mm is not None:
        body = _body_support(depth_mm, color_bgr, box, z_mm)
    return float(end * 0.50 + elong * 0.20 + scale * 0.15 + body * 0.15)


def detect_sideways(
    color_bgr, depth_mm, intrinsics=None
) -> list[tuple[int, int, int, int]]:
    """Detect sideways / fallen white-capped vials.

    Fuses (1) white-cap + horizontal morph elongate and (2) elongated Canny
    contours via IoU clustering. Keeps clusters with white end-cap, elongated
    body, and vial-scale at Z.
    """
    if color_bgr is None or depth_mm is None:
        return []
    color_bgr = np.asarray(color_bgr)
    depth_mm = np.asarray(depth_mm)
    if color_bgr.ndim != 3 or depth_mm.ndim != 2:
        return []
    if color_bgr.shape[:2] != depth_mm.shape[:2]:
        return []

    p1 = _proposals_cap_morph(color_bgr)
    p2 = _proposals_canny_elongate(color_bgr)
    proposals = p1 + p2

    h, w = color_bgr.shape[:2]
    pre_scores: list[float] = []
    for box in proposals:
        box = _clip_box(box, w, h)
        z_mm = _median_z_mm(depth_mm, color_bgr, box)
        pre_scores.append(_rank(color_bgr, box, z_mm, intrinsics, depth_mm))

    fused = _cluster_by_iou(proposals, pre_scores)

    kept: list[tuple[tuple[int, int, int, int], float]] = []
    for box in fused:
        box = _clip_box(box, w, h)
        if not _is_elongated(box):
            continue
        if _white_end_score(color_bgr, box) < 0.12:
            continue
        # Even a few invalid columns are meaningful for clear plastic; opaque
        # top-down lids and the real-wide controls have none.
        if _clear_cup_fraction(depth_mm, color_bgr, box) < 0.02:
            continue
        z_mm = _median_z_mm(depth_mm, color_bgr, box)
        if not _scale_ok_at_z(box, z_mm, intrinsics):
            continue
        kept.append((box, _rank(color_bgr, box, z_mm, intrinsics, depth_mm)))

    kept.sort(key=lambda t: t[1], reverse=True)
    # Suppress near-duplicates (IoU or heavily overlapping horizontal spans).
    final: list[tuple[int, int, int, int]] = []
    for box, _score in kept:
        bx, _by, bw, _bh = box
        duplicate = False
        for fx, _fy, fw, _fh in final:
            if _iou(box, (fx, _fy, fw, _fh)) >= 0.35:
                duplicate = True
                break
            # Same row-ish sideways hit: any x-overlap with nearby centers.
            overlap = min(bx + bw, fx + fw) - max(bx, fx)
            cx_a = bx + bw * 0.5
            cx_b = fx + fw * 0.5
            if overlap > 0 and abs(cx_a - cx_b) < 0.85 * max(bw, fw):
                duplicate = True
                break
        if not duplicate:
            final.append(box)
    return final


def _synth_sideways(
    width: int = 320,
    height: int = 240,
    lid_on: str = "left",
    z_mm: float = 450.0,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Simple synthetic: gray table + translucent body + opaque white lid."""
    color = np.full((height, width, 3), 48, dtype=np.uint8)
    # Soft noise clutter
    noise = np.random.default_rng(0).integers(0, 25, color.shape, dtype=np.uint8)
    color = cv2.add(color, noise)

    body_w, body_h = 110, 52
    x0 = width // 2 - body_w // 2
    y0 = height // 2 - body_h // 2
    # Translucent-ish body (cool mid-gray) with clear wall edges for Canny.
    cv2.rectangle(color, (x0, y0), (x0 + body_w - 1, y0 + body_h - 1), (78, 72, 62), -1)
    cv2.rectangle(color, (x0, y0), (x0 + body_w - 1, y0 + body_h - 1), (120, 110, 95), 2)
    # Soft inner cup (slightly darker) so walls remain.
    cv2.rectangle(
        color,
        (x0 + 4, y0 + 4),
        (x0 + body_w - 5, y0 + body_h - 5),
        (58, 54, 48),
        -1,
    )

    lid_w = 28
    if lid_on == "left":
        lx = x0
    else:
        lx = x0 + body_w - lid_w
    color[y0 : y0 + body_h, lx : lx + lid_w] = (235, 235, 235)
    cv2.rectangle(color, (lx, y0), (lx + lid_w - 1, y0 + body_h - 1), (210, 210, 210), 1)

    depth = np.full((height, width), 900.0, dtype=np.float32)
    # Opaque lid has solid near depth; body has hole / far patches (clear).
    depth[y0 : y0 + body_h, x0 : x0 + body_w] = z_mm + 180.0
    depth[y0 : y0 + body_h, lx : lx + lid_w] = z_mm
    # Punch a few invalids in the cup (see-through).
    depth[y0 + 8 : y0 + body_h - 8, x0 + 40 : x0 + body_w - 40 : 6] = 0.0

    gt = (x0, y0, body_w, body_h)
    return color, depth, gt


if __name__ == "__main__":
    color, depth, gt = _synth_sideways(lid_on="left")
    fx = fy = 600.0
    cx, cy = color.shape[1] / 2.0, color.shape[0] / 2.0
    boxes = detect_sideways(color, depth, intrinsics=(fx, fy, cx, cy))
    print(f"NAME={NAME}")
    print(f"gt={gt}")
    print(f"detections={boxes}")
    if boxes:
        best_iou = max(_iou(gt, b) for b in boxes)
        print(f"best_iou={best_iou:.3f}")
    else:
        print("best_iou=0.000")
        raise SystemExit(1)
