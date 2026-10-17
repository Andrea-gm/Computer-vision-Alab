"""Depth-aware geometric vial finder (point cloud / plane / cylinder).

Pipeline
--------
1. Seed opaque white caps from RGB (HSV).
2. Align depth; deproject neighborhood to camera-frame XYZ (mm).
3. Estimate a coarse support plane from valid far-ish depth.
4. Cluster points near each seed (Z-band + height-above-plane).
5. 3D PCA → cylinder axis; size prior from ``vial_3d_dims.json``.
6. Score with clear-cup depth holes + size consistency; emit 2D boxes + 3D poses.

When depth is missing, a weak RGB→pseudo-depth prior is used (documented
limitation): vertical image position + projected size from the fitted vial
model. That mode is for ROS topic wiring only — not competitive with real
Orbbec depth.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

NAME = "agent_08_depth_geometry"

# Jessica/robotics is the sibling of Jessica/ros2_ws.
_CAMERA_ROOT = Path(
    os.environ.get(
        "ORBBEC_CAMERA_ROOT",
        Path(__file__).resolve().parents[4] / "robotics",
    )
)
_DIMS_CANDIDATES = (
    _CAMERA_ROOT / "vial_3d_dims.json",
    Path(__file__).resolve().parents[1] / "config" / "vial_3d_dims.json",
)

_DEFAULT_LENGTH_MM = 72.69
_DEFAULT_DIAM_MM = 54.22
_MIN_DEPTH_MM = 80.0
_MAX_DEPTH_MM = 5000.0
_Z_BAND_MM = 90.0
_MIN_POINTS = 35
_MAX_DETECTIONS = 8


@dataclass
class VialDetection:
    """One geometric vial hypothesis."""

    box_xywh: tuple[int, int, int, int]
    score: float
    center_xyz_mm: tuple[float, float, float]
    axis_xyz: tuple[float, float, float]
    radius_mm: float
    height_mm: float
    orientation: str  # "upright" | "sideways"
    used_real_depth: bool
    n_points: int = 0


def _load_vial_mm(dims_path: Path | None = None) -> tuple[float, float]:
    length = _DEFAULT_LENGTH_MM
    diam = _DEFAULT_DIAM_MM
    path = dims_path
    if path is None:
        for cand in _DIMS_CANDIDATES:
            if cand.is_file():
                path = cand
                break
    if path is None or not Path(path).is_file():
        return length, diam
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return length, diam
    diam = float(data.get("lid_diameter_mm", diam))
    if "total_height_mm" in data:
        length = float(data["total_height_mm"])
    else:
        length = float(data.get("lid_height_mm", 16.0)) + float(
            data.get("cup_height_mm", 57.0)
        )
    return length, diam


def default_intrinsics(width: int, height: int) -> tuple[float, float, float, float]:
    """HFOV≈90° prior used elsewhere in this project when K is missing."""
    fx = fy = float(width) / (2.0 * np.tan(np.radians(45.0)))
    return fx, fy, width * 0.5, height * 0.5


def normalize_intrinsics(
    intrinsics: Any, width: int, height: int
) -> tuple[float, float, float, float]:
    if intrinsics is None:
        return default_intrinsics(width, height)
    try:
        if isinstance(intrinsics, dict):
            fx = float(intrinsics["fx"])
            fy = float(intrinsics["fy"])
            cx = float(intrinsics.get("cx", width * 0.5))
            cy = float(intrinsics.get("cy", height * 0.5))
        else:
            fx, fy, cx, cy = (float(v) for v in intrinsics[:4])
    except (KeyError, TypeError, ValueError, IndexError):
        return default_intrinsics(width, height)
    if fx <= 1.0 or fy <= 1.0:
        return default_intrinsics(width, height)
    return fx, fy, cx, cy


def align_depth(depth_mm: np.ndarray | None, width: int, height: int) -> np.ndarray:
    if depth_mm is None:
        return np.zeros((height, width), dtype=np.float32)
    d = np.asarray(depth_mm, dtype=np.float32)
    if d.ndim == 3:
        d = d[:, :, 0]
    if d.shape[:2] == (height, width):
        return d
    return cv2.resize(d, (width, height), interpolation=cv2.INTER_NEAREST)


def pseudo_depth_from_rgb(
    color_bgr: np.ndarray,
    intrinsics: Any = None,
    length_mm: float | None = None,
) -> np.ndarray:
    """Weak RGB→pseudo-depth for topic wiring when depth is absent.

    Limitation: not metric; assumes a roughly fronto-parallel scene with the
    vial near mid-field (~400 mm) and uses white-cap scale for local Z.
    """
    h, w = color_bgr.shape[:2]
    fx, fy, _, _ = normalize_intrinsics(intrinsics, w, h)
    if length_mm is None:
        length_mm, _ = _load_vial_mm()
    depth = np.full((h, w), 900.0, dtype=np.float32)
    # Mild vertical gradient (table-like): lower rows slightly nearer.
    rows = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    depth = depth * (1.05 - 0.15 * rows)

    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] >= 175) & (hsv[:, :, 1] <= 55)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 40:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        diam_px = float(max(bw, bh))
        if diam_px < 6:
            continue
        # Z ≈ f * D / d
        z = float(fy) * float(_DEFAULT_DIAM_MM) / max(diam_px, 1.0)
        z = float(np.clip(z, 150.0, 1200.0))
        pad = int(max(bw, bh) * 1.5)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        depth[y0:y1, x0:x1] = z
    return depth


def _clip_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = (int(round(v)) for v in box)
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    return x, y, w, h


def _white_cap_mask(color_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] >= 175) & (hsv[:, :, 1] <= 55)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _cap_seeds(mask: np.ndarray, min_area: int = 40) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seeds: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 6 or h < 6:
            continue
        if area > 0.12 * mask.shape[0] * mask.shape[1]:
            continue
        solidity = area / max(float(w * h), 1.0)
        if solidity < 0.30:
            continue
        # Reject huge wall-like bright slabs; keep end-caps and top lids.
        if max(w, h) / float(min(w, h)) > 4.5:
            continue
        seeds.append((x, y, w, h))
    seeds.sort(key=lambda b: b[2] * b[3], reverse=True)
    return seeds[:20]


def _median_z(depth: np.ndarray, box) -> float | None:
    x, y, w, h = box
    roi = depth[y : y + h, x : x + w]
    valid = roi[(roi > _MIN_DEPTH_MM) & (roi < _MAX_DEPTH_MM) & np.isfinite(roi)]
    if valid.size < 6:
        return None
    return float(np.median(valid))


def _deproject(
    depth: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    z = depth[v.astype(np.int32), u.astype(np.int32)].astype(np.float64)
    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy
    return np.column_stack((x, y, z))


def _estimate_plane(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Fit plane n·x + d = 0 with n pointing roughly toward camera (-Z)."""
    if points.shape[0] < 80:
        return None
    # Subsample
    if points.shape[0] > 4000:
        idx = np.linspace(0, points.shape[0] - 1, 4000).astype(np.int32)
        pts = points[idx]
    else:
        pts = points
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vh[-1].astype(np.float64)
    nrm = np.linalg.norm(normal)
    if nrm < 1e-9:
        return None
    normal = normal / nrm
    # Prefer normal with positive Z component toward camera? Camera looks +Z into scene;
    # table normal often near -Y (upright) or +Z. Keep sign so n·centroid < 0 when possible.
    d = -float(np.dot(normal, centroid))
    if d > 0:
        normal = -normal
        d = -d
    return normal, d


def _height_above_plane(points: np.ndarray, plane) -> np.ndarray:
    normal, d = plane
    return points @ normal + d


def _collect_seed_cloud(
    depth: np.ndarray,
    cap_box,
    cap_z: float,
    search_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (uv Nx2, mask of valid pixels in ROI)."""
    h, w = depth.shape[:2]
    cx = cap_box[0] + cap_box[2] * 0.5
    cy = cap_box[1] + cap_box[3] * 0.5
    rad = int(max(20.0, search_px))
    x0 = max(0, int(cx - rad))
    x1 = min(w, int(cx + rad) + 1)
    y0 = max(0, int(cy - rad))
    y1 = min(h, int(cy + rad) + 1)
    roi = depth[y0:y1, x0:x1]
    ys, xs = np.where(
        np.isfinite(roi)
        & (roi > _MIN_DEPTH_MM)
        & (roi < _MAX_DEPTH_MM)
        & (np.abs(roi - cap_z) <= _Z_BAND_MM)
    )
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float64), (x0, y0, x1, y1)
    uv = np.column_stack((xs.astype(np.float64) + x0, ys.astype(np.float64) + y0))
    if uv.shape[0] > 3000:
        idx = np.linspace(0, uv.shape[0] - 1, 3000).astype(np.int32)
        uv = uv[idx]
    return uv, (x0, y0, x1, y1)


def _clear_cup_support(depth: np.ndarray, cap_box, length_px: float) -> tuple[float, float, float, float]:
    """Invalid-depth fraction left/right/below/above the cap (cup see-through cue)."""
    h, w = depth.shape[:2]
    x, y, bw, bh = cap_box
    reach = max(8, int(round(length_px - min(bw, bh))))

    def frac(x0, x1, y0, y1) -> float:
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        roi = depth[y0:y1, x0:x1]
        return float((~np.isfinite(roi) | (roi <= _MIN_DEPTH_MM)).mean())

    left = frac(x - reach, x, y, y + bh)
    right = frac(x + bw, x + bw + reach, y, y + bh)
    above = frac(x, x + bw, y - reach, y)
    below = frac(x, x + bw, y + bh, y + bh + reach)
    return left, right, above, below


def _pca_3d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if points.shape[0] < _MIN_POINTS:
        return None
    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    if cov.shape != (3, 3) or not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    axis = vecs[:, order[0]].astype(np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return None
    return mean, axis / n


def _project_aabb(
    center_xyz: np.ndarray,
    axis: np.ndarray,
    half_len: float,
    radius: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Project a coarse cylinder AABB into the image."""
    # Build 8 corners of an oriented box around the cylinder.
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    # Orthonormal basis.
    tmp = np.array([0.0, 1.0, 0.0]) if abs(axis[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    b1 = np.cross(axis, tmp)
    b1 = b1 / (np.linalg.norm(b1) + 1e-12)
    b2 = np.cross(axis, b1)
    corners = []
    for s in (-1.0, 1.0):
        for t in (-1.0, 1.0):
            for u in (-1.0, 1.0):
                p = center_xyz + s * half_len * axis + t * radius * b1 + u * radius * b2
                if p[2] <= 1.0:
                    continue
                uu = fx * p[0] / p[2] + cx
                vv = fy * p[1] / p[2] + cy
                corners.append((uu, vv))
    if len(corners) < 4:
        return None
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0, x1 = int(np.floor(min(xs))), int(np.ceil(max(xs)))
    y0, y1 = int(np.floor(min(ys))), int(np.ceil(max(ys)))
    return _clip_box((x0, y0, x1 - x0, y1 - y0), width, height)


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


def _nms(dets: list[VialDetection], thr: float = 0.45) -> list[VialDetection]:
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: list[VialDetection] = []
    for d in dets:
        if all(_iou(d.box_xywh, k.box_xywh) < thr for k in kept):
            kept.append(d)
    return kept


def detect_vials(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray | None = None,
    intrinsics: Any = None,
    *,
    dims_path: Path | None = None,
    allow_pseudo_depth: bool = True,
    min_score: float = 0.25,
) -> list[VialDetection]:
    """Run depth-geometry vial detection.

    Returns list of ``VialDetection`` (2D box + 3D cylinder pose).
    """
    if color_bgr is None or getattr(color_bgr, "size", 0) == 0:
        return []
    image = np.asarray(color_bgr)
    if image.ndim != 3 or image.shape[2] < 3:
        return []
    height, width = image.shape[:2]
    length_mm, diam_mm = _load_vial_mm(dims_path)
    fx, fy, cx, cy = normalize_intrinsics(intrinsics, width, height)

    used_real = depth_mm is not None and np.asarray(depth_mm).size > 0
    if used_real:
        depth = align_depth(depth_mm, width, height)
        # Treat all-zero / empty depth as missing.
        if not np.any((depth > _MIN_DEPTH_MM) & (depth < _MAX_DEPTH_MM)):
            used_real = False
    else:
        depth = None  # type: ignore

    if not used_real:
        if not allow_pseudo_depth:
            return []
        depth = pseudo_depth_from_rgb(image, (fx, fy, cx, cy), length_mm)

    # Global plane from sparse valid depth (background / table).
    ys_g, xs_g = np.where(
        (depth > 200.0) & (depth < _MAX_DEPTH_MM) & np.isfinite(depth)
    )
    plane = None
    if xs_g.size >= 200:
        if xs_g.size > 5000:
            idx = np.linspace(0, xs_g.size - 1, 5000).astype(np.int32)
            xs_s, ys_s = xs_g[idx], ys_g[idx]
        else:
            xs_s, ys_s = xs_g, ys_g
        cloud_g = _deproject(depth, xs_s.astype(np.float64), ys_s.astype(np.float64), fx, fy, cx, cy)
        plane = _estimate_plane(cloud_g)

    mask = _white_cap_mask(image)
    seeds = _cap_seeds(mask)
    if not seeds:
        return []

    detections: list[VialDetection] = []
    for cap in seeds:
        cap = _clip_box(cap, width, height)
        cap_z = _median_z(depth, cap)
        if cap_z is None:
            continue
        length_px = float(length_mm) * fx / cap_z
        diam_px = float(diam_mm) * fy / cap_z
        if length_px < 14 or diam_px < 8:
            continue
        if length_px > width * 1.5 or diam_px > height * 1.4:
            continue

        left_c, right_c, above_c, below_c = _clear_cup_support(depth, cap, length_px)
        clear_max = max(left_c, right_c, above_c, below_c)

        uv, _roi = _collect_seed_cloud(depth, cap, cap_z, max(length_px, diam_px) * 1.2)
        if uv.shape[0] < _MIN_POINTS:
            continue
        pts = _deproject(depth, uv[:, 0], uv[:, 1], fx, fy, cx, cy)

        if plane is not None:
            heights = _height_above_plane(pts, plane)
            # Keep points that are elevated relative to the support plane OR near the cap Z band.
            elev = np.abs(heights)
            keep = elev <= max(length_mm * 1.2, 120.0)
            if keep.sum() >= _MIN_POINTS:
                pts = pts[keep]
                uv = uv[keep]

        pca = _pca_3d(pts)
        if pca is None:
            continue
        mean_xyz, axis = pca

        # Cap center (image + 3D).
        cu = cap[0] + cap[2] * 0.5
        cv = cap[1] + cap[3] * 0.5
        cap_xyz = np.array(
            [
                (cu - cx) * cap_z / fx,
                (cv - cy) * cap_z / fy,
                cap_z,
            ],
            dtype=np.float64,
        )

        # Prefer clear-cup direction for orientation (reliable for transparent vials).
        horiz_clear = max(left_c, right_c)
        vert_clear = max(above_c, below_c)
        if horiz_clear >= vert_clear and horiz_clear >= 0.12:
            orientation = "sideways"
        elif vert_clear >= 0.12:
            orientation = "upright"
        else:
            abs_ax = abs(float(axis[0]))
            abs_ay = abs(float(axis[1]))
            orientation = "upright" if abs_ay >= abs_ax and abs_ay >= 0.45 else "sideways"

        # Clear-cup gate: transparent body should show depth holes next to the lid.
        if used_real:
            if orientation == "sideways" and clear_max < 0.18:
                continue
            if orientation == "upright" and clear_max < 0.10:
                continue

        radius = 0.5 * diam_mm
        half_len = 0.5 * length_mm

        # Image-plane body direction (for tight 2D AABB) + 3D axis for PoseArray.
        if orientation == "sideways":
            if right_c >= left_c:
                body_img = np.array([1.0, 0.0], dtype=np.float64)
                body_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                body_img = np.array([-1.0, 0.0], dtype=np.float64)
                body_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
            # Diam from opaque end-cap height is more stable than projected prior.
            diam_use = float(max(cap[3], min(diam_px, cap[3] * 1.8)))
            length_use = float(length_px)
            # Cap at one end; grow along body into the clear cup.
            if body_img[0] >= 0:
                x0 = float(cap[0])
                x1 = x0 + length_use
            else:
                x1 = float(cap[0] + cap[2])
                x0 = x1 - length_use
            y_c = cv
            y0 = y_c - 0.5 * diam_use
            y1 = y_c + 0.5 * diam_use
            box = _clip_box((x0, y0, x1 - x0, y1 - y0), width, height)
            if box[2] / float(max(box[3], 1)) < 1.05:
                continue
        else:
            if below_c >= above_c:
                body_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            else:
                body_dir = np.array([0.0, -1.0, 0.0], dtype=np.float64)
            diam_use = float(max(cap[2], min(diam_px, cap[2] * 1.8)))
            length_use = float(length_px)
            if body_dir[1] >= 0:
                y0 = float(cap[1])
                y1 = y0 + length_use
            else:
                y1 = float(cap[1] + cap[3])
                y0 = y1 - length_use
            x0 = cu - 0.5 * diam_use
            x1 = cu + 0.5 * diam_use
            box = _clip_box((x0, y0, x1 - x0, y1 - y0), width, height)
            if box[3] / float(max(box[2], 1)) < 1.05:
                continue

        # Refine 3D axis with PCA when it agrees with image body direction.
        body_dir3 = body_dir.copy()
        to_mean = mean_xyz - cap_xyz
        if float(np.dot(to_mean, axis)) < 0:
            axis = -axis
        if orientation == "sideways" and abs(float(axis[0])) >= 0.35:
            body_dir3 = axis
            body_dir3[1] = 0.0
            body_dir3[2] = 0.0
            n = np.linalg.norm(body_dir3)
            if n > 1e-6:
                body_dir3 = body_dir3 / n
                if np.sign(body_dir3[0]) != np.sign(body_dir[0]):
                    body_dir3 = -body_dir3
            else:
                body_dir3 = body_dir
        elif orientation == "upright" and abs(float(axis[1])) >= 0.35:
            body_dir3 = axis
            body_dir3[0] = 0.0
            n = np.linalg.norm(body_dir3)
            body_dir3 = body_dir3 / n if n > 1e-6 else body_dir
            if np.sign(body_dir3[1]) != np.sign(body_dir[1]):
                body_dir3 = -body_dir3

        center = cap_xyz + body_dir3 * half_len
        if np.linalg.norm(mean_xyz - center) < length_mm:
            center = 0.65 * center + 0.35 * mean_xyz

        # Optional: tighten with projected cylinder if it overlaps the image AABB.
        proj = _project_aabb(
            center, body_dir3, half_len, radius, fx, fy, cx, cy, width, height
        )
        if proj is not None and _iou(box, proj) >= 0.25:
            # Intersection of image AABB and projected cylinder AABB.
            x0 = max(box[0], proj[0])
            y0 = max(box[1], proj[1])
            x1 = min(box[0] + box[2], proj[0] + proj[2])
            y1 = min(box[1] + box[3], proj[1] + proj[3])
            if x1 > x0 + 8 and y1 > y0 + 6:
                # Prefer the image-plane length prior box (tighter GT match); keep it.
                pass

        aspect = box[2] / float(max(box[3], 1))
        size_score = 1.0 - min(
            1.0,
            abs(diam_px - float(max(cap[2], cap[3]))) / max(diam_px, 1.0),
        )
        score = (
            0.35 * min(1.0, clear_max / 0.45)
            + 0.30 * min(1.0, pts.shape[0] / 400.0)
            + 0.25 * size_score
            + 0.10 * (1.0 if used_real else 0.2)
        )
        if orientation == "sideways":
            score += 0.05 * min(2.0, aspect) / 2.0
        if score < min_score:
            continue

        detections.append(
            VialDetection(
                box_xywh=box,
                score=float(score),
                center_xyz_mm=(float(center[0]), float(center[1]), float(center[2])),
                axis_xyz=(float(body_dir3[0]), float(body_dir3[1]), float(body_dir3[2])),
                radius_mm=float(radius),
                height_mm=float(length_mm),
                orientation=orientation,
                used_real_depth=bool(used_real),
                n_points=int(pts.shape[0]),
            )
        )

    return _nms(detections)[:_MAX_DETECTIONS]


def detect_sideways(color_bgr, depth_mm, intrinsics=None):
    """Sideways-race compatible API: list of (x, y, w, h)."""
    dets = detect_vials(color_bgr, depth_mm, intrinsics, allow_pseudo_depth=False)
    return [d.box_xywh for d in dets if d.orientation == "sideways"]


def annotate(
    color_bgr: np.ndarray, detections: list[VialDetection]
) -> np.ndarray:
    out = color_bgr.copy()
    for d in detections:
        x, y, w, h = d.box_xywh
        color = (40, 200, 40) if d.orientation == "sideways" else (40, 160, 220)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"{d.orientation} {d.score:.2f} z={d.center_xyz_mm[2]:.0f}"
        cv2.putText(
            out,
            label,
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        # Project center.
        # (already have box; draw axis tip roughly)
        cx = x + w // 2
        cy = y + h // 2
        tip = (
            int(cx + d.axis_xyz[0] * 28),
            int(cy + d.axis_xyz[1] * 28),
        )
        cv2.arrowedLine(out, (cx, cy), tip, color, 2, tipLength=0.35)
    mode = "RGB-D" if any(d.used_real_depth for d in detections) else (
        "pseudo-depth" if detections else "none"
    )
    cv2.putText(
        out,
        f"agent_08 depth_geometry ({mode}) n={len(detections)}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"agent_08 depth_geometry ({mode}) n={len(detections)}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return out
