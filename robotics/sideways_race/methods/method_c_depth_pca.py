"""Method C: seed white cap, PCA on nearby depth, grow to vial length/diam AABB.

Uses ``vial_3d_dims.json`` when present (~73 mm total length, ~54 mm lid diameter).
Rejects upright principal axes. Pure OpenCV + numpy; does not open cameras.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

NAME = "C_depth_pca"

_ROOT = Path(__file__).resolve().parents[2]
_DIMS_PATH = _ROOT / "vial_3d_dims.json"

# Defaults match fitted race dims when JSON is missing.
_DEFAULT_LENGTH_MM = 72.69
_DEFAULT_DIAM_MM = 54.22

_MIN_DEPTH_MM = 80.0
_MAX_DEPTH_MM = 5000.0
_Z_BAND_MM = 80.0
_MIN_PCA_POINTS = 40
_HORIZONTAL_DOT = 0.55  # |axis · (1,0)| must exceed this (reject upright)
_MAX_DETECTIONS = 8


def _load_vial_mm() -> tuple[float, float]:
    """Return (total_length_mm, lid_diameter_mm)."""
    length = _DEFAULT_LENGTH_MM
    diam = _DEFAULT_DIAM_MM
    if not _DIMS_PATH.is_file():
        return length, diam
    try:
        data = json.loads(_DIMS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return length, diam
    diam = float(data.get("lid_diameter_mm", diam))
    if "total_height_mm" in data:
        length = float(data["total_height_mm"])
    else:
        lid_h = float(data.get("lid_height_mm", 16.0))
        cup_h = float(data.get("cup_height_mm", 57.0))
        length = lid_h + cup_h
    return length, diam


def _default_intrinsics(width: int, height: int) -> tuple[float, float, float, float]:
    """HFOV≈90° prior used elsewhere in this project when K is missing."""
    fx = fy = float(width) / (2.0 * np.tan(np.radians(45.0)))
    return fx, fy, width * 0.5, height * 0.5


def _normalize_intrinsics(intrinsics, width: int, height: int):
    if intrinsics is None:
        return _default_intrinsics(width, height)
    try:
        if isinstance(intrinsics, dict):
            fx = float(intrinsics["fx"])
            fy = float(intrinsics["fy"])
            cx = float(intrinsics.get("cx", width * 0.5))
            cy = float(intrinsics.get("cy", height * 0.5))
        else:
            fx, fy, cx, cy = (float(v) for v in intrinsics[:4])
    except (KeyError, TypeError, ValueError, IndexError):
        return _default_intrinsics(width, height)
    if fx <= 1.0 or fy <= 1.0:
        return _default_intrinsics(width, height)
    return fx, fy, cx, cy


def _clip_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = (int(round(v)) for v in box)
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    return x, y, w, h


def _align_depth(depth_mm: np.ndarray, width: int, height: int) -> np.ndarray:
    if depth_mm is None:
        return np.zeros((height, width), dtype=np.float32)
    d = np.asarray(depth_mm, dtype=np.float32)
    if d.shape[:2] == (height, width):
        return d
    return cv2.resize(d, (width, height), interpolation=cv2.INTER_NEAREST)


def _white_cap_mask(color_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    mask = ((value >= 175) & (saturation <= 55)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _cap_seeds(mask: np.ndarray, min_area: int = 40) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seeds = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 6 or h < 6:
            continue
        # A side-on end cap is a narrow vertical strip.  Square top-down lids,
        # wide upright lids, and bright rectangular clutter are not seeds.
        if w / float(h) > 0.82:
            continue
        # Cap-ish: not a huge wall slab.
        if area > 0.12 * mask.shape[0] * mask.shape[1]:
            continue
        solidity = area / max(float(w * h), 1.0)
        if solidity < 0.35:
            continue
        seeds.append((x, y, w, h))
    seeds.sort(key=lambda b: b[2] * b[3], reverse=True)
    return seeds[:16]


def _median_z(depth: np.ndarray, box) -> float | None:
    x, y, w, h = box
    roi = depth[y : y + h, x : x + w]
    valid = roi[(roi > _MIN_DEPTH_MM) & (roi < _MAX_DEPTH_MM) & np.isfinite(roi)]
    if valid.size < 8:
        return None
    return float(np.median(valid))


def _collect_nearby_points(
    depth: np.ndarray,
    cap_box,
    cap_z: float,
    length_px: float,
    diam_px: float,
) -> np.ndarray:
    """Gather (u, v) pixels with valid depth near the cap at similar Z."""
    h, w = depth.shape[:2]
    cx = cap_box[0] + cap_box[2] * 0.5
    cy = cap_box[1] + cap_box[3] * 0.5
    # Search a bit beyond expected vial extent so the body rim can enter PCA.
    rad_x = int(max(cap_box[2], length_px) * 1.15 + diam_px * 0.35)
    rad_y = int(max(cap_box[3], diam_px) * 1.15)
    x0 = max(0, int(cx - rad_x))
    x1 = min(w, int(cx + rad_x) + 1)
    y0 = max(0, int(cy - rad_y))
    y1 = min(h, int(cy + rad_y) + 1)
    roi = depth[y0:y1, x0:x1]
    ys, xs = np.where(
        np.isfinite(roi)
        & (roi > _MIN_DEPTH_MM)
        & (roi < _MAX_DEPTH_MM)
        & (np.abs(roi - cap_z) <= _Z_BAND_MM)
    )
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    pts = np.column_stack((xs.astype(np.float64) + x0, ys.astype(np.float64) + y0))
    # Subsample dense clouds for stable PCA.
    if pts.shape[0] > 2500:
        idx = np.linspace(0, pts.shape[0] - 1, 2500).astype(np.int32)
        pts = pts[idx]
    return pts


def _clear_cup_support(
    depth: np.ndarray, cap_box, length_px: float
) -> tuple[float, float]:
    """Return invalid-depth support immediately left and right of a cap."""
    h, w = depth.shape[:2]
    x, y, bw, bh = cap_box
    reach = max(8, int(round(length_px - bw)))

    def side(x0: int, x1: int) -> float:
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y), min(h, y + bh)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        roi = depth[y0:y1, x0:x1]
        return float((~np.isfinite(roi) | (roi <= _MIN_DEPTH_MM)).mean())

    return side(x - reach, x), side(x + bw, x + bw + reach)


def _pca_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (mean_xy, unit_long_axis) or None."""
    if points.shape[0] < _MIN_PCA_POINTS:
        return None
    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    axis = vecs[:, order[0]].astype(np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return None
    axis = axis / n
    # Prefer pointing roughly +x for stability (sign flipped later using body).
    if axis[0] < 0:
        axis = -axis
    return mean, axis


def _axis_is_horizontal(axis: np.ndarray) -> bool:
    return abs(float(axis[0])) >= _HORIZONTAL_DOT


def _body_direction(
    color_bgr: np.ndarray,
    cap_box,
    axis: np.ndarray,
    mean_xy: np.ndarray,
    pts: np.ndarray,
    length_px: float,
) -> np.ndarray:
    """Unit axis pointing from lid into the cup (depth extent + color)."""
    h, w = color_bgr.shape[:2]
    cx = cap_box[0] + cap_box[2] * 0.5
    cy = cap_box[1] + cap_box[3] * 0.5
    cap_c = np.array([cx, cy], dtype=np.float64)

    pos_score = 0.0
    neg_score = 0.0
    if pts.shape[0] >= 8:
        proj = (pts - cap_c) @ axis
        pos = proj[proj > 2.0]
        neg = proj[proj < -2.0]
        if pos.size:
            pos_score += float(pos.max()) + 0.15 * float(pos.size)
        if neg.size:
            neg_score += float((-neg).max()) + 0.15 * float(neg.size)

    # Color: prefer the side that is less white / more body-like midtones.
    step = max(8.0, 0.35 * length_px)
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 150)
    value = hsv[:, :, 2]

    def probe(sign: float) -> tuple[float, float]:
        px = cx + sign * axis[0] * step
        py = cy + sign * axis[1] * step
        x0 = int(max(0, px - 5))
        x1 = int(min(w, px + 6))
        y0 = int(max(0, py - 5))
        y1 = int(min(h, py + 6))
        if x1 <= x0 or y1 <= y0:
            return 1.0, 0.0
        roi_w = white[y0:y1, x0:x1]
        roi_v = value[y0:y1, x0:x1]
        body = (~roi_w) & (roi_v >= 40) & (roi_v <= 170)
        return float(roi_w.mean()), float(body.mean())

    w_pos, b_pos = probe(+1.0)
    w_neg, b_neg = probe(-1.0)
    pos_score += (1.0 - w_pos) * 20.0 + b_pos * 25.0
    neg_score += (1.0 - w_neg) * 20.0 + b_neg * 25.0

    # PCA mean side is a weak prior when rim points dominate one side.
    if float(np.dot(mean_xy - cap_c, axis)) > 0:
        pos_score += 8.0
    else:
        neg_score += 8.0

    return axis if pos_score >= neg_score else -axis


def _grow_aabb(
    color_bgr: np.ndarray,
    cap_box,
    mean_xy: np.ndarray,
    axis: np.ndarray,
    length_px: float,
    diam_px: float,
    pts: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """Place an AABB of expected length×diam along the horizontal PCA axis.

    Cap stays at one end; extent grows into the cup using depth/color cues.
    """
    h, w = color_bgr.shape[:2]
    cx = cap_box[0] + cap_box[2] * 0.5
    cy = cap_box[1] + cap_box[3] * 0.5
    body_dir = _body_direction(color_bgr, cap_box, axis, mean_xy, pts, length_px)

    # Cap center is near one end; vial center is half-length into the body.
    half_len = 0.5 * length_px
    cap_along_axis = cap_box[2] * abs(float(body_dir[0])) + cap_box[3] * abs(
        float(body_dir[1])
    )
    center = np.array([cx, cy], dtype=np.float64) + body_dir * max(
        0.0, half_len - 0.5 * cap_along_axis
    )
    to_mean = mean_xy - np.array([cx, cy], dtype=np.float64)
    if float(np.dot(to_mean, body_dir)) > 0:
        center = 0.7 * center + 0.3 * mean_xy

    half_w = 0.5 * length_px
    half_h = 0.5 * diam_px
    ax = abs(float(body_dir[0]))
    ay = abs(float(body_dir[1]))
    perp = np.array([-body_dir[1], body_dir[0]], dtype=np.float64)
    ext_x = half_w * ax + half_h * abs(float(perp[0]))
    ext_y = half_w * ay + half_h * abs(float(perp[1]))
    x0 = int(np.floor(center[0] - ext_x))
    y0 = int(np.floor(center[1] - ext_y))
    x1 = int(np.ceil(center[0] + ext_x))
    y1 = int(np.ceil(center[1] + ext_y))
    box = _clip_box((x0, y0, x1 - x0, y1 - y0), w, h)
    bw, bh = box[2], box[3]
    if bw < 12 or bh < 8:
        return None
    if bh / float(bw) > 0.85:
        return None
    if bw / float(max(bh, 1)) < 1.05:
        return None
    return box


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
    return float(inter / max(union, 1))


def _nms(boxes: list[tuple[int, int, int, int]], thr: float = 0.45):
    kept: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if all(_iou(box, k) < thr for k in kept):
            kept.append(box)
    return kept


def detect_sideways(color_bgr, depth_mm, intrinsics=None) -> list[tuple[int, int, int, int]]:
    """Detect sideways vials via white-cap seed + depth PCA growth.

    Returns axis-aligned ``(x, y, w, h)`` boxes covering lid + cup.
    """
    if color_bgr is None or getattr(color_bgr, "size", 0) == 0:
        return []
    image = np.asarray(color_bgr)
    if image.ndim != 3 or image.shape[2] < 3:
        return []
    height, width = image.shape[:2]
    depth = _align_depth(depth_mm, width, height)
    fx, fy, _, _ = _normalize_intrinsics(intrinsics, width, height)
    length_mm, diam_mm = _load_vial_mm()

    mask = _white_cap_mask(image)
    seeds = _cap_seeds(mask)
    if not seeds:
        return []

    proposals: list[tuple[float, tuple[int, int, int, int]]] = []
    for cap in seeds:
        cap = _clip_box(cap, width, height)
        cap_z = _median_z(depth, cap)
        if cap_z is None:
            continue
        length_px = float(length_mm) * fx / cap_z
        projected_diam_px = float(diam_mm) * fy / cap_z
        # The opaque end-cap directly observes the short dimension more
        # reliably than a physical projection when calibration or depth is
        # approximate.
        diam_px = float(cap[3])
        # Sanity: projected vial must be visible at image scale.
        if length_px < 18 or projected_diam_px < 10:
            continue
        if length_px > width * 1.4 or projected_diam_px > height * 1.2:
            continue

        left_clear, right_clear = _clear_cup_support(depth, cap, length_px)
        if max(left_clear, right_clear) < 0.15:
            continue

        pts = _collect_nearby_points(depth, cap, cap_z, length_px, diam_px)
        pca = _pca_axis(pts)
        if pca is None:
            # Fallback: assume horizontal from image geometry of the cap neighborhood.
            # Prefer the direction of larger horizontal depth extent around the cap.
            axis = np.array([1.0, 0.0], dtype=np.float64)
            mean_xy = np.array(
                [cap[0] + cap[2] * 0.5, cap[1] + cap[3] * 0.5], dtype=np.float64
            )
            if pts.shape[0] >= 12:
                mean_xy = pts.mean(axis=0)
            else:
                continue
        else:
            mean_xy, axis = pca
            if not _axis_is_horizontal(axis):
                # With a clear cup the only near-depth pixels may be the
                # vertical opaque cap itself.  Its PCA axis is vertical even
                # though the vial necessarily extends horizontally.
                axis = np.array([1.0, 0.0], dtype=np.float64)

        # Invalid depth through the clear cup is the strongest direction cue.
        if right_clear > left_clear + 0.08:
            axis = np.array([1.0, 0.0], dtype=np.float64)
            mean_xy = np.array(
                [cap[0] + cap[2], cap[1] + cap[3] * 0.5], dtype=np.float64
            )
        elif left_clear > right_clear + 0.08:
            axis = np.array([-1.0, 0.0], dtype=np.float64)
            mean_xy = np.array(
                [cap[0], cap[1] + cap[3] * 0.5], dtype=np.float64
            )

        box = _grow_aabb(image, cap, mean_xy, axis, length_px, diam_px, pts)
        if box is None:
            continue
        # Score: longer horizontal + more PCA support.
        score = box[2] / max(box[3], 1) + 0.001 * min(pts.shape[0], 2000)
        proposals.append((score, box))

    proposals.sort(key=lambda t: t[0], reverse=True)
    boxes = [b for _, b in proposals]
    return _nms(boxes)[:_MAX_DETECTIONS]


def _synth_sideways(
    width: int = 640,
    height: int = 400,
    *,
    lid_on_left: bool = True,
    z_mm: float = 350.0,
):
    """Synthetic fallen vial sized from dims+Z+K (white end-cap + depth rim)."""
    image = np.full((height, width, 3), 95, dtype=np.uint8)
    image[::5, :, :] = (88, 88, 88)
    length_mm, diam_mm = _load_vial_mm()
    fx, fy, cx, cy = _default_intrinsics(width, height)
    length_px = int(round(length_mm * fx / z_mm))
    diam_px = int(round(diam_mm * fy / z_mm))
    length_px = max(40, min(width - 40, length_px))
    diam_px = max(24, min(height - 40, diam_px))
    lid_w = max(12, min(int(round(diam_px * 0.35)), length_px // 3))
    x0 = (width - length_px) // 2
    y0 = (height - diam_px) // 2
    image[y0 : y0 + diam_px, x0 : x0 + length_px] = (70, 72, 68)
    image[y0 : y0 + diam_px, x0] = (40, 40, 40)
    image[y0 : y0 + diam_px, x0 + length_px - 1] = (40, 40, 40)
    image[y0, x0 : x0 + length_px] = (45, 45, 45)
    image[y0 + diam_px - 1, x0 : x0 + length_px] = (45, 45, 45)
    if lid_on_left:
        image[y0 : y0 + diam_px, x0 : x0 + lid_w] = (245, 245, 245)
    else:
        image[y0 : y0 + diam_px, x0 + length_px - lid_w : x0 + length_px] = (
            245,
            245,
            245,
        )
    gt = (x0, y0, length_px, diam_px)

    depth = np.full((height, width), 1800.0, dtype=np.float32)
    bx, by, bw, bh = gt
    if lid_on_left:
        depth[by : by + bh, bx : bx + lid_w] = z_mm
        body_x0, body_x1 = bx + lid_w, bx + bw
    else:
        depth[by : by + bh, bx + bw - lid_w : bx + bw] = z_mm
        body_x0, body_x1 = bx, bx + bw - lid_w
    # Body rim at near-Z so PCA sees a horizontal cloud beyond the lid.
    depth[by : by + 3, body_x0:body_x1] = z_mm + 4.0
    depth[by + bh - 3 : by + bh, body_x0:body_x1] = z_mm + 4.0
    depth[by : by + bh, body_x0 : min(body_x0 + 2, body_x1)] = z_mm + 6.0
    depth[by : by + bh, max(body_x1 - 2, body_x0) : body_x1] = z_mm + 6.0
    cup = depth[by + 3 : by + bh - 3, body_x0 + 2 : body_x1 - 2]
    if cup.size:
        cup[:] = 0.0
        cup[::4, ::5] = 1600.0

    K = (fx, fy, cx, cy)
    return image, depth, K, gt


def _self_test() -> None:
    print(f"NAME={NAME}")
    for lid_left in (True, False):
        image, depth, K, gt = _synth_sideways(lid_on_left=lid_left)
        boxes = detect_sideways(image, depth, K)
        side = "left" if lid_left else "right"
        print(f"{side}: gt={gt} boxes={boxes}")
        if not boxes:
            raise SystemExit(f"self-test failed: no detections ({side})")
        best_iou = max(_iou(gt, b) for b in boxes)
        print(f"{side}: best_iou={best_iou:.3f}")
        if best_iou < 0.45:
            raise SystemExit(f"self-test failed: IoU {best_iou:.3f} < 0.45 ({side})")
    image, depth, K, _ = _synth_sideways(lid_on_left=True)
    uh, uw = image.shape[0], image.shape[1]
    upright = np.full((uh, uw, 3), 95, dtype=np.uint8)
    ux, uy, vw, vh = uw // 2 - 18, uh // 2 - 70, 36, 140
    upright[uy : uy + vh, ux : ux + vw] = (70, 70, 65)
    upright[uy : uy + 18, ux : ux + vw] = (245, 245, 245)
    ud = np.full((uh, uw), 1800.0, dtype=np.float32)
    ud[uy : uy + 18, ux : ux + vw] = 350.0
    ud[uy + 18 : uy + vh, ux : ux + 2] = 355.0
    ud[uy + 18 : uy + vh, ux + vw - 2 : ux + vw] = 355.0
    upright_boxes = detect_sideways(upright, ud, K)
    print(f"upright_boxes={upright_boxes}")
    upright_gt = (ux, uy, vw, vh)
    for b in upright_boxes:
        if _iou(upright_gt, b) >= 0.3:
            raise SystemExit(f"self-test failed: upright false positive {b}")
    print("self-test ok")


if __name__ == "__main__":
    _self_test()
