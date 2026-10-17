"""J_research: cap-ellipse + depth-hole geometry with gated race supplements.

Classical RGB-D stack for clear vials with white caps:

- B + B2 depth-hole fallback (recall / zero-FP anchor on the race set)
- Ellipse-validated cap seeds fused to horizontal depth-hole components →
  ``minAreaRect`` AABB for angled bodies
- Optional A/D proposals accepted only with hole-fraction + single-end-cap gates

No training, no global orientation CNN, no keypoint mainline.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

NAME = "J_research"

try:
    from sideways_race.methods.method_a_cap_horizontal_grow import (
        detect_sideways as _detect_a,
    )
    from sideways_race.methods.method_d_edge_template import (
        detect_sideways as _detect_d,
    )
    from sideways_race.methods.method_b2_improved import _depth_hole_fallback
    from sideways_race.methods.method_b_rotated_white_cap import (
        detect_sideways as _detect_b,
    )
except ModuleNotFoundError:
    from method_a_cap_horizontal_grow import detect_sideways as _detect_a  # type: ignore[no-redef]
    from method_d_edge_template import detect_sideways as _detect_d  # type: ignore[no-redef]
    from method_b2_improved import _depth_hole_fallback  # type: ignore[no-redef]
    from method_b_rotated_white_cap import detect_sideways as _detect_b  # type: ignore[no-redef]

_MIN_Z = 80.0
_MAX_Z = 5000.0
_MIN_HOLE_FRAC = 0.35
_MIN_WHITE_END_SUPP = 0.14
_MIN_WHITE_END_D = 0.22
_DIMS = Path(__file__).resolve().parents[2] / "vial_3d_dims.json"
_DEFAULT_FX = 575.0

# Sideways AABB height/width band (matches race synth + live gate).
_HW_MIN = 0.34
_HW_MAX = 0.82


def _align_depth(depth_mm, w: int, h: int) -> np.ndarray | None:
    if depth_mm is None:
        return None
    d = np.asarray(depth_mm, dtype=np.float32)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
    return d


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _fx_fy(intrinsics) -> tuple[float, float]:
    if isinstance(intrinsics, dict):
        return float(intrinsics.get("fx", _DEFAULT_FX)), float(
            intrinsics.get("fy", intrinsics.get("fx", _DEFAULT_FX))
        )
    if intrinsics is not None:
        try:
            return float(intrinsics[0]), float(intrinsics[1])
        except (IndexError, TypeError, ValueError):
            pass
    return _DEFAULT_FX, _DEFAULT_FX


def _cap_depth(depth: np.ndarray, box) -> float | None:
    x, y, w, h = box
    ix, iy = max(1, w // 5), max(1, h // 6)
    patch = depth[y + iy : y + h - iy, x + ix : x + w - ix]
    valid = patch[np.isfinite(patch) & (patch > _MIN_Z) & (patch < _MAX_Z)]
    return float(np.median(valid)) if valid.size >= 5 else None


def _ellipse_cap_candidates(
    color_bgr: np.ndarray,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Compact near-white blobs with a plausible side-on cap ellipse."""
    h, w = color_bgr.shape[:2]
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    val, sat = hsv[:, :, 2], hsv[:, :, 1]
    white = ((val >= 188) & (sat <= 58)).astype(np.uint8) * 255
    white = cv2.morphologyEx(
        white, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[tuple[int, int, int, int], float]] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 28 or len(cnt) < 5:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 5 or bh < 8 or bw > 0.22 * w or bh > 0.16 * h:
            continue
        if bw * bh > 2200:
            continue
        fill = area / max(float(bw * bh), 1.0)
        if fill < 0.48:
            continue
        try:
            (_cx, _cy), (ma, mi), _ang = cv2.fitEllipse(cnt)
        except cv2.error:
            continue
        if ma < 1 or mi < 1:
            continue
        ecc = min(ma, mi) / max(ma, mi)
        if ecc < 0.32 or ecc > 0.98:
            continue
        ar = bw / max(bh, 1)
        if not (0.10 <= ar <= 1.05):
            continue
        score = fill * (0.55 + 0.45 * ecc)
        out.append(((x, y, bw, bh), score))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def _depth_hole_mask(depth: np.ndarray) -> np.ndarray:
    holes = (depth <= _MIN_Z).astype(np.uint8)
    holes = cv2.morphologyEx(
        holes,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)),
    )
    holes = cv2.morphologyEx(
        holes,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)),
    )
    return holes


def _body_rgb_mask(color_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    tone = (val >= 25) & (val <= 195) & (sat <= 120)
    rim = gy >= 14.0
    return (tone | rim).astype(np.uint8)


def _component_touching_cap(
    holes: np.ndarray,
    body_rgb: np.ndarray,
    cap: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Horizontal union of depth-failure blob attached to the cap (morph seed)."""
    x, y, bw, bh = cap
    h, w = holes.shape
    pad = max(2, bh // 6)
    seed = np.zeros_like(holes)
    x0, x1 = max(0, x - pad), min(w, x + bw + pad)
    y0, y1 = max(0, y - pad), min(h, y + bh + pad)
    seed[y0:y1, x0:x1] = 255
    candidate = cv2.bitwise_and(holes * 255, body_rgb * 255)
    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)),
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    cap_cx = x + bw // 2
    cap_cy = y + bh // 2
    best_i = -1
    best_area = 0
    for i in range(1, n):
        cx, cy, cw, ch, area = [int(v) for v in stats[i]]
        if area < 40 or cw < 10:
            continue
        if not (cx <= cap_cx <= cx + cw and cy <= cap_cy <= cy + ch):
            continue
        if area > best_area:
            best_area = area
            best_i = i
    if best_i < 0:
        return None
    return (labels == best_i).astype(np.uint8)


def _box_from_cap_body(
    cap: tuple[int, int, int, int],
    body: np.ndarray | None,
    cap_z: float,
    intrinsics,
) -> tuple[int, int, int, int] | None:
    x, y, bw, bh = cap
    h, w = body.shape if body is not None else (0, 0)
    if body is not None and body.any():
        ys, xs = np.where(body > 0)
        bx0, bx1 = int(xs.min()), int(xs.max()) + 1
        by0, by1 = int(ys.min()), int(ys.max()) + 1
        # Keep vertical extent tight: cap band dominates sideways silhouette.
        by0 = max(by0, y - 1)
        by1 = min(by1, y + bh + 1)
        if by1 - by0 < bh:
            by0, by1 = y, y + bh
        x0, y0 = min(x, bx0), by0
        x1, y1 = max(x + bw, bx1), by1
    else:
        x0, y0, x1, y1 = x, y, x + bw, y + bh

    tw, th = x1 - x0, y1 - y0
    if tw < 18 or th < 8:
        return None
    hw = th / max(tw, 1)
    if not (_HW_MIN <= hw <= _HW_MAX):
        return None
    if tw < 1.15 * th:
        return None

    fx, fy = _fx_fy(intrinsics)
    if fx > 1 and fy > 1:
        long_mm = tw * cap_z / fx
        short_mm = th * cap_z / fy
        total_h, lid_d = 72.69, 54.22
        try:
            data = json.loads(_DIMS.read_text(encoding="utf-8"))
            total_h = float(data.get("total_height_mm", total_h))
            lid_d = float(data.get("lid_diameter_mm", lid_d))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        if not (0.30 * total_h <= long_mm <= 2.6 * total_h):
            return None
        if not (0.25 * lid_d <= short_mm <= 2.2 * lid_d):
            return None

    return (max(0, x0), max(0, y0), min(tw, w - x0), min(th, h - y0))


def _white_end_score(color_bgr: np.ndarray, box) -> float:
    h, w = color_bgr.shape[:2]
    x, y, bw, bh = box
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    bw = max(1, min(bw, w - x))
    bh = max(1, min(bh, h - y))
    hsv = cv2.cvtColor(color_bgr[y : y + bh, x : x + bw], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 150)
    band = max(3, int(bw * 0.32))
    if bw <= band * 2:
        return 0.0
    left = float(white[:, :band].mean())
    right = float(white[:, -band:].mean())
    mid = float(white[:, band : bw - band].mean())
    end = max(left, right)
    if end < 0.20 or mid > end - 0.04:
        return 0.0
    return float(max(0.0, min(1.0, end - mid + 0.06)))


def _clear_cup_hole_frac(depth: np.ndarray, box) -> float:
    x, y, bw, bh = box
    roi = depth[y : y + bh, x : x + bw]
    band = max(3, int(bw * 0.32))
    if bw <= band * 2:
        return 0.0
    mid = roi[:, band : bw - band]
    return float((mid <= _MIN_Z).mean())


def _oriented_aabb_from_mask(
    mask: np.ndarray, frame_w: int, frame_h: int
) -> tuple[int, int, int, int] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 40 or len(cnt) < 5:
        return None
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    x0 = int(np.floor(box[:, 0].min()))
    y0 = int(np.floor(box[:, 1].min()))
    x1 = int(np.ceil(box[:, 0].max()))
    y1 = int(np.ceil(box[:, 1].max()))
    x0 = max(0, min(frame_w - 1, x0))
    y0 = max(0, min(frame_h - 1, y0))
    x1 = max(x0 + 1, min(frame_w, x1))
    y1 = max(y0 + 1, min(frame_h, y1))
    return x0, y0, x1 - x0, y1 - y0


def _oriented_cap_proposals(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    intrinsics,
) -> list[tuple[int, int, int, int]]:
    """Ellipse cap + attached depth hole → minAreaRect AABB (rotated hard positives)."""
    frame_h, frame_w = color_bgr.shape[:2]
    holes = _depth_hole_mask(depth)
    body_rgb = _body_rgb_mask(color_bgr)
    out: list[tuple[int, int, int, int]] = []

    for cap, _score in _ellipse_cap_candidates(color_bgr):
        z = _cap_depth(depth, cap)
        if z is None:
            continue
        body = _component_touching_cap(holes, body_rgb, cap)
        x, y, bw, bh = cap
        union = np.zeros((frame_h, frame_w), np.uint8)
        union[y : y + bh, x : x + bw] = 255
        if body is not None:
            union = cv2.bitwise_or(union, body * 255)
        box = _oriented_aabb_from_mask(union, frame_w, frame_h)
        if box is None:
            box = _box_from_cap_body(cap, body, z, intrinsics)
        if box is None:
            continue
        hw = box[3] / max(box[2], 1)
        if not (_HW_MIN <= hw <= _HW_MAX) or box[2] < 1.15 * box[3]:
            continue
        if _white_end_score(color_bgr, box) < 0.12:
            continue
        if _clear_cup_hole_frac(depth, box) < _MIN_HOLE_FRAC:
            continue
        out.append(box)
    return out


def _supplement_ok(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    box: tuple[int, int, int, int],
    core: list[tuple[int, int, int, int]],
    *,
    source: str,
) -> bool:
    if any(_iou(box, c) >= 0.38 for c in core):
        return False
    _, _, bw, bh = box
    if bw < 16 or bh < 8:
        return False
    hw = bh / max(bw, 1)
    if not (_HW_MIN <= hw <= _HW_MAX) or bw < 1.15 * bh:
        return False
    if _clear_cup_hole_frac(depth, box) < _MIN_HOLE_FRAC:
        return False
    end = _white_end_score(color_bgr, box)
    if source == "D":
        return end >= _MIN_WHITE_END_D
    return end >= _MIN_WHITE_END_SUPP


def _merge_b2(
    primary: list[tuple[int, int, int, int]],
    fallback: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    result = list(fallback)
    for box in primary:
        if not any(_iou(box, other) >= 0.45 for other in result):
            result.append(box)
    return result


def _dedupe(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    ranked = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for box in ranked:
        if not any(_iou(box, k) >= 0.42 for k in kept):
            kept.append(box)
    return kept


def detect_sideways(
    color_bgr,
    depth_mm,
    intrinsics=None,
) -> list[tuple[int, int, int, int]]:
    if color_bgr is None or getattr(color_bgr, "size", 0) == 0:
        return []
    color = np.asarray(color_bgr)
    if color.ndim != 3 or color.shape[2] < 3:
        return []
    frame_h, frame_w = color.shape[:2]
    depth = _align_depth(depth_mm, frame_w, frame_h)
    if depth is None:
        return []

    # B + B2 depth-hole fallback (recall / FP anchor), plus gated supplements.
    core = _merge_b2(_detect_b(color, depth, intrinsics), _depth_hole_fallback(color, depth))
    for box in _oriented_cap_proposals(color, depth, intrinsics):
        if _supplement_ok(color, depth, box, core, source="J"):
            core.append(box)
    for box in _detect_a(color, depth, intrinsics) or []:
        b = tuple(int(v) for v in box)
        if _supplement_ok(color, depth, b, core, source="A"):
            core.append(b)
    for box in _detect_d(color, depth, intrinsics) or []:
        b = tuple(int(v) for v in box)
        if _supplement_ok(color, depth, b, core, source="D"):
            core.append(b)
    return _dedupe(core)


def _synth_sideways(lid_left: bool = True):
    """Minimal synth patch matching the race harness."""
    height, width = 240, 320
    color = np.full((height, width, 3), 90, dtype=np.uint8)
    vial_h = 42
    vial_w = int(round(vial_h / 0.55))
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
    for lid_left in (True, False):
        image, depth_image, gt = _synth_sideways(lid_left=lid_left)
        dets = detect_sideways(image, depth_image, {"fx": 575.0, "fy": 575.0})
        best = max((_iou(d, gt) for d in dets), default=0.0)
        print(f"lid_left={lid_left} gt={gt} dets={dets} best_iou={best:.3f}")
        if best < 0.5:
            raise SystemExit(1)

    wall = np.full((240, 320, 3), 200, dtype=np.uint8)
    wall_d = np.full((240, 320), 800.0, dtype=np.float32)
    if detect_sideways(wall, wall_d):
        raise SystemExit("wall false positive")

    upright = np.full((240, 320, 3), 90, dtype=np.uint8)
    upright_d = np.full((240, 320), 1100.0, dtype=np.float32)
    upright[70:82, 140:178] = 245
    upright[82:135, 142:176] = 65
    upright_d[70:82, 140:178] = 520.0
    upright_d[82:135, 142:176] = 0.0
    if detect_sideways(upright, upright_d):
        raise SystemExit("upright false positive")

    print("self-test OK")
