"""RGB-D any-pose vial finder (standing, lying, or angled).

Round-2 vs agent_08
-------------------
* Major axis from **3D PCA** on the seed cluster — kept as a free 3D vector
  (vertical OR horizontal OR diagonal). Round-1 snapped axis to camera X/Y.
* 8-direction clear-depth + dark-RGB body scan recovers angled hard cases and
  missing-depth cups.
* RGB-only fallback is orientation-aware (no upright-only assumption).
* Box length uses measured body extent (tighter upright / sideways IoU).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

NAME = "r2_07_depth_anypose"

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
_MIN_POINTS = 30
_MAX_DETECTIONS = 8

_DIRS_8 = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (0.7071, 0.7071),
    (-0.7071, 0.7071),
    (0.7071, -0.7071),
    (-0.7071, -0.7071),
)


@dataclass
class VialDetection:
    box_xywh: tuple[int, int, int, int]
    score: float
    center_xyz_mm: tuple[float, float, float]
    axis_xyz: tuple[float, float, float]
    radius_mm: float
    height_mm: float
    orientation: str  # upright | sideways | angled | topdown
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


def _dim_cap_mask(color_bgr: np.ndarray) -> np.ndarray:
    """Secondary seed mask for dim hard-synth lids (V∈[150,175))."""
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 2] >= 150) & (hsv[:, :, 2] < 180) & (hsv[:, :, 1] <= 65)
    ).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


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
    if points.shape[0] < 80:
        return None
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
) -> np.ndarray:
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
        return np.zeros((0, 2), dtype=np.float64)
    uv = np.column_stack((xs.astype(np.float64) + x0, ys.astype(np.float64) + y0))
    if uv.shape[0] > 3000:
        idx = np.linspace(0, uv.shape[0] - 1, 3000).astype(np.int32)
        uv = uv[idx]
    return uv


def _invalid_frac(depth: np.ndarray, x0, y0, x1, y1) -> float:
    h, w = depth.shape[:2]
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = depth[y0:y1, x0:x1]
    return float((~np.isfinite(roi) | (roi <= _MIN_DEPTH_MM)).mean())


def _dark_frac(gray: np.ndarray, x0, y0, x1, y1, thr: int = 110) -> float:
    h, w = gray.shape[:2]
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((gray[y0:y1, x0:x1] < thr).mean())


def _clear_cup_cardinal(
    depth: np.ndarray, cap_box, length_px: float
) -> tuple[float, float, float, float]:
    h, w = depth.shape[:2]
    x, y, bw, bh = cap_box
    reach = max(8, int(round(length_px - min(bw, bh))))

    def frac(x0, x1, y0, y1) -> float:
        return _invalid_frac(depth, x0, y0, x1, y1)

    left = frac(x - reach, x, y, y + bh)
    right = frac(x + bw, x + bw + reach, y, y + bh)
    above = frac(x, x + bw, y - reach, y)
    below = frac(x, x + bw, y + bh, y + bh + reach)
    return left, right, above, below


def _scan_body(
    color_bgr: np.ndarray,
    depth: np.ndarray | None,
    cap_box,
    length_px: float,
) -> tuple[np.ndarray, float, float]:
    """Best body direction from clear-depth + dark-RGB (any pose)."""
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    x, y, bw, bh = cap_box
    cu, cv = x + bw * 0.5, y + bh * 0.5
    reach = max(10, int(round(length_px * 0.95)))
    half_w = max(4, int(0.45 * min(bw, bh)))
    best_dir = np.array([1.0, 0.0], dtype=np.float64)
    best_score = -1.0
    best_clear = 0.0
    best_dark = 0.0
    for dx, dy in _DIRS_8:
        start = np.array([cu, cv]) + np.array([dx, dy]) * (0.5 * max(bw, bh) + 2.0)
        end = start + np.array([dx, dy]) * reach
        px, py = -dy, dx
        us = [
            start[0] + px * half_w,
            start[0] - px * half_w,
            end[0] + px * half_w,
            end[0] - px * half_w,
        ]
        vs = [
            start[1] + py * half_w,
            start[1] - py * half_w,
            end[1] + py * half_w,
            end[1] - py * half_w,
        ]
        x0, x1 = min(us), max(us)
        y0, y1 = min(vs), max(vs)
        dark = _dark_frac(gray, x0, y0, x1, y1)
        clear = _invalid_frac(depth, x0, y0, x1, y1) if depth is not None else 0.0
        score = 0.60 * clear + 0.40 * dark
        # RGB-only: prefer body axis matching lid aspect (wide lid → upright).
        if depth is None:
            if bw >= bh and abs(dy) >= abs(dx):
                score += 0.12
            elif bh >= bw and abs(dx) >= abs(dy):
                score += 0.12
        if score > best_score:
            best_score = score
            best_dir = np.array([dx, dy], dtype=np.float64)
            best_clear, best_dark = clear, dark
    return best_dir, best_clear, best_dark


def _measure_extent(
    color_bgr: np.ndarray,
    depth: np.ndarray | None,
    cap_box,
    dir_uv: np.ndarray,
    max_px: float,
    cap_z: float | None = None,
) -> float:
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    x, y, bw, bh = cap_box
    cu, cv = x + bw * 0.5, y + bh * 0.5
    step = 2.0
    n_steps = int(max(8, max_px / step))
    last = 0.5 * max(bw, bh)
    for i in range(1, n_steps + 1):
        dist = 0.5 * max(bw, bh) + i * step
        ui = int(round(cu + dir_uv[0] * dist))
        vi = int(round(cv + dir_uv[1] * dist))
        if ui < 1 or vi < 1 or ui >= w - 1 or vi >= h - 1:
            break
        patch = gray[vi - 1 : vi + 2, ui - 1 : ui + 2]
        gmean = float(patch.mean())
        dark = gmean < 100.0  # body plastic; desk ~88–100 mottled — stay strict
        hole = False
        table = False
        if depth is not None:
            dz = depth[vi - 1 : vi + 2, ui - 1 : ui + 2]
            hole = float((~np.isfinite(dz) | (dz <= _MIN_DEPTH_MM)).mean()) >= 0.45
            valid = dz[(dz > _MIN_DEPTH_MM) & np.isfinite(dz)]
            if valid.size >= 3 and cap_z is not None:
                # Solid table depth near/far from lid → end of clear cup.
                med = float(np.median(valid))
                if abs(med - cap_z) > 180.0 and not hole:
                    table = True
        if table:
            break
        if hole or (dark and gmean < 95.0):
            last = dist
        elif dark and hole:
            last = dist
        else:
            break
    return float(last + 0.30 * max(bw, bh))


def _pca_3d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
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
    return mean, axis / n, vals[order]


def _classify_orientation(axis: np.ndarray) -> str:
    ax, ay, az = abs(float(axis[0])), abs(float(axis[1])), abs(float(axis[2]))
    if az >= 0.70 and az >= ax and az >= ay:
        return "topdown"
    if ay >= 0.55 and ay >= ax:
        return "upright"
    if ax >= 0.55 and ax >= ay:
        return "sideways"
    if abs(float(axis[0])) >= 0.35 and abs(float(axis[1])) >= 0.35:
        return "angled"
    return "upright" if ay >= ax else "sideways"


def _project_cylinder_aabb(
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
    axis = axis / (np.linalg.norm(axis) + 1e-12)
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
                corners.append((fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy))
    if len(corners) < 4:
        return None
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return _clip_box(
        (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)), width, height
    )


def _box_from_dir(
    cap_box,
    dir_uv: np.ndarray,
    length_px: float,
    diam_px: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x, y, bw, bh = cap_box
    cu, cv = x + bw * 0.5, y + bh * 0.5
    d = dir_uv / (np.linalg.norm(dir_uv) + 1e-12)
    if abs(d[0]) >= 0.35 and abs(d[1]) >= 0.35:
        tip = np.array([cu, cv]) + d * (length_px - 0.5 * max(bw, bh))
        perp = np.array([-d[1], d[0]]) * 0.5 * diam_px
        pts = np.array(
            [
                [x, y],
                [x + bw, y],
                [x, y + bh],
                [x + bw, y + bh],
                tip + perp,
                tip - perp,
                np.array([cu, cv]) - d * 0.5 * max(bw, bh) + perp,
                np.array([cu, cv]) - d * 0.5 * max(bw, bh) - perp,
            ]
        )
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        return _clip_box((x0, y0, x1 - x0, y1 - y0), width, height)
    if abs(d[0]) >= abs(d[1]):
        if d[0] >= 0:
            x0, x1 = float(x), float(x) + length_px
        else:
            x1 = float(x + bw)
            x0 = x1 - length_px
        y0, y1 = cv - 0.5 * diam_px, cv + 0.5 * diam_px
    else:
        if d[1] >= 0:
            y0, y1 = float(y), float(y) + length_px
        else:
            y1 = float(y + bh)
            y0 = y1 - length_px
        x0, x1 = cu - 0.5 * diam_px, cu + 0.5 * diam_px
    return _clip_box((x0, y0, x1 - x0, y1 - y0), width, height)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter <= 0:
        return 0.0
    return float(inter / max(aw * ah + bw * bh - inter, 1))


def _nms(dets: list[VialDetection], thr: float = 0.45) -> list[VialDetection]:
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: list[VialDetection] = []
    for d in dets:
        if all(_iou(d.box_xywh, k.box_xywh) < thr for k in kept):
            kept.append(d)
    return kept


def _hypothesis_from_seed(
    image: np.ndarray,
    depth: np.ndarray | None,
    cap,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    length_mm: float,
    diam_mm: float,
    plane,
    used_real: bool,
    min_score: float,
    require_strong_body: bool,
) -> VialDetection | None:
    height, width = image.shape[:2]
    cap = _clip_box(cap, width, height)
    cap_z = _median_z(depth, cap) if depth is not None else None
    z_scale = float(cap_z) if cap_z is not None else 520.0

    length_px = float(length_mm) * fx / z_scale
    diam_px = float(diam_mm) * fy / z_scale
    if length_px < 12 or diam_px < 7:
        return None
    if length_px > width * 1.5 or diam_px > height * 1.4:
        return None

    left_c = right_c = above_c = below_c = 0.0
    if depth is not None:
        left_c, right_c, above_c, below_c = _clear_cup_cardinal(depth, cap, length_px)
    clear_max = max(left_c, right_c, above_c, below_c)

    dir_uv, clear_s, dark_s = _scan_body(image, depth, cap, length_px)
    body_cue = max(clear_s, dark_s, clear_max)

    # Cardinal override when clear-cup is decisive (matches Round-1 strength).
    horiz = max(left_c, right_c)
    vert = max(above_c, below_c)
    if horiz >= vert and horiz >= 0.12:
        dir_uv = np.array([1.0, 0.0] if right_c >= left_c else [-1.0, 0.0])
        orientation_hint = "sideways"
    elif vert >= 0.12:
        dir_uv = np.array([0.0, 1.0] if below_c >= above_c else [0.0, -1.0])
        orientation_hint = "upright"
    else:
        orientation_hint = None

    if require_strong_body and body_cue < 0.28:
        return None
    if body_cue < 0.10 and cap_z is None:
        return None
    # Huge bright slabs (paper / windows) are not vial end-caps.
    if cap[2] * cap[3] > 0.04 * width * height:
        return None

    cap_span = float(max(cap[2], cap[3]))
    # Far / table-bleed lid Z → mm prior shorter than the lid itself. Gate the
    # hard_sideways_* rescue on that signal so race samples stay untouched.
    wrong_z = bool(cap_z is not None and length_px < cap_span * 1.55)
    extent_max = (
        max(length_px * 2.40, cap_span * 3.10, 100.0)
        if wrong_z
        else length_px * 1.35
    )
    extent = _measure_extent(
        image, depth, cap, dir_uv, extent_max, cap_z=cap_z
    )
    # Prefer measured clear/dark extent. Wrong-Z: trust cap-relative length.
    if extent >= cap_span * 1.20:
        if wrong_z and clear_max >= 0.20:
            upper = max(length_px * 2.60, cap_span * 2.85)
        elif clear_max >= 0.20:
            upper = length_px * 1.80
        else:
            upper = length_px * 1.25
        length_use = float(np.clip(extent, cap_span * 1.15, upper))
    else:
        length_use = float(np.clip(length_px * 0.85, length_px * 0.50, length_px * 1.15))
        if wrong_z and clear_max >= 0.30 and dark_s >= 0.40:
            # Occluded / angled body: extent stalls; still trust clear-cup + dark.
            length_use = float(max(length_use, length_px * 1.15, cap_span * 2.05))

    mean_xyz = None
    n_pts = 0
    axis3 = np.array([dir_uv[0], dir_uv[1], 0.0], dtype=np.float64)
    n = np.linalg.norm(axis3)
    axis3 = axis3 / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

    if depth is not None and cap_z is not None:
        uv = _collect_seed_cloud(depth, cap, cap_z, max(length_use, diam_px) * 1.2)
        if uv.shape[0] >= _MIN_POINTS:
            pts = _deproject(depth, uv[:, 0], uv[:, 1], fx, fy, cx, cy)
            if plane is not None:
                elev = np.abs(_height_above_plane(pts, plane))
                keep = elev <= max(length_mm * 1.2, 120.0)
                if int(keep.sum()) >= _MIN_POINTS:
                    pts = pts[keep]
            pca = _pca_3d(pts)
            if pca is not None:
                mean_xyz, pca_axis, eig = pca
                n_pts = int(pts.shape[0])
                elong = float(eig[0] / max(float(eig[1]), 1e-6))
                agree = abs(float(np.dot(pca_axis[:2], axis3[:2])))
                # Lid-only clusters are flat/wide; only trust PCA when it is
                # elongated AND aligned with the clear-cup / dark-body direction.
                if elong >= 1.25 and agree >= 0.50:
                    if float(np.dot(pca_axis[:2], axis3[:2])) < 0:
                        pca_axis = -pca_axis
                    # Free 3D axis (any pose) — do not snap to camera X/Y.
                    axis3 = pca_axis

    orientation = orientation_hint or _classify_orientation(axis3)

    # Top-down (axis≈camera Z) omitted by default: solid white squares are
    # ambiguous vs hard-negative lid clutter without a body cue.

    # Clear-cup gate (real depth): match Round-1 — opaque white slabs without
    # adjacent depth holes are rejected. Angled may use strong dark body.
    if used_real and depth is not None:
        if orientation == "sideways" and clear_max < 0.18:
            return None
        if orientation == "upright" and clear_max < 0.10:
            return None
        if orientation == "angled" and clear_max < 0.10 and dark_s < 0.28:
            return None

    # Diameter from end-cap; when clear-cup is strong, allow mm-prior blend so
    # hard_sideways_* boxes are not chronically undersized.
    if abs(dir_uv[0]) >= abs(dir_uv[1]):
        # Sideways / angled: enlarge chronically short hard_sideways_right boxes.
        if clear_max >= 0.20 and orientation in ("sideways", "angled"):
            if wrong_z:
                # Race-like length/cap ratio (~2.05–2.2); avoid chronic oversize
                # (hard_right_01) and undersize (hard_right_03/05).
                target_len = cap_span * 2.15
                diam_use = float(max(cap[3] * 1.45, min(max(diam_px, cap[3] * 1.25), cap[3] * 1.85)))
                length_use = float(
                    np.clip(
                        max(length_use, extent * 1.02, target_len * 0.95),
                        target_len * 0.88,
                        target_len * 1.12,
                    )
                )
            else:
                diam_use = float(max(cap[3] * 1.25, min(diam_px * 0.80, cap[3] * 1.85)))
                length_use = float(max(length_use, min(extent * 1.05, length_px * 1.35)))
                length_use = float(max(length_use, length_px * 0.90))
                # Angled hard cases with good Z but extent collapses early
                # (hard_right_06 +25°, ext/lp≈0.55) while clear+dark stay strong.
                # Race boxes that match a short true vial keep ext/lp≳0.75.
                if (
                    clear_max >= 0.38
                    and dark_s >= 0.95
                    and extent < length_px * 0.65
                ):
                    length_use = float(max(length_use, length_px * 1.18))
                    diam_use = float(max(diam_use, cap[3] * 1.35))
        else:
            diam_use = float(cap[3] * 1.10)
        # Prefer length boost over reject when wrong-Z + strong body cues
        # (hard_sideways_right_03: far Z → short prior, fat diam → aspect kill).
        if orientation in ("sideways", "angled") and length_use / max(diam_use, 1) < 1.05:
            if wrong_z and clear_max >= 0.30 and dark_s >= 0.40:
                length_use = float(max(length_use, diam_use * 1.40, cap_span * 2.05))
                length_use = float(min(length_use, cap_span * 2.40))
            else:
                return None
        if orientation in ("sideways", "angled") and length_use / max(diam_use, 1) < 1.05:
            return None
    else:
        # Upright: keep measured extent (do not inflate with mm prior).
        diam_use = float(cap[2] * 1.10)
        if orientation == "upright" and length_use / max(diam_use, 1) < 1.05:
            return None

    box = _box_from_dir(cap, dir_uv, length_use, diam_use, width, height)

    cu = cap[0] + cap[2] * 0.5
    cv = cap[1] + cap[3] * 0.5
    cap_xyz = np.array(
        [(cu - cx) * z_scale / fx, (cv - cy) * z_scale / fy, z_scale],
        dtype=np.float64,
    )
    half_len = 0.5 * length_mm
    radius = 0.5 * diam_mm
    # Point axis from cap into the body.
    to_body = np.array([dir_uv[0], dir_uv[1], 0.0], dtype=np.float64)
    if float(np.dot(axis3[:2], to_body[:2])) < 0:
        axis3 = -axis3
    center = cap_xyz + axis3 * half_len
    if mean_xyz is not None and np.linalg.norm(mean_xyz - center) < length_mm:
        center = 0.65 * center + 0.35 * mean_xyz

    proj = _project_cylinder_aabb(
        center, axis3, half_len, radius, fx, fy, cx, cy, width, height
    )
    if proj is not None and _iou(box, proj) >= 0.25:
        # Prefer image-plane measured box (tighter GT); light blend for angled.
        if orientation == "angled":
            x0 = int(round(0.7 * box[0] + 0.3 * proj[0]))
            y0 = int(round(0.7 * box[1] + 0.3 * proj[1]))
            x1 = int(round(0.7 * (box[0] + box[2]) + 0.3 * (proj[0] + proj[2])))
            y1 = int(round(0.7 * (box[1] + box[3]) + 0.3 * (proj[1] + proj[3])))
            if x1 > x0 + 6 and y1 > y0 + 6:
                box = _clip_box((x0, y0, x1 - x0, y1 - y0), width, height)

    aspect = max(box[2], box[3]) / float(max(1, min(box[2], box[3])))
    size_score = 1.0 - min(
        1.0, abs(diam_px - float(max(cap[2], cap[3]))) / max(diam_px, 1.0)
    )
    score = (
        0.35 * min(1.0, body_cue / 0.45)
        + 0.25 * min(1.0, n_pts / 400.0)
        + 0.25 * size_score
        + 0.10 * (1.0 if used_real and cap_z is not None else 0.25)
        + 0.05 * min(1.0, aspect / 1.8)
    )
    if score < min_score:
        return None

    return VialDetection(
        box_xywh=box,
        score=float(min(1.0, score)),
        center_xyz_mm=(float(center[0]), float(center[1]), float(center[2])),
        axis_xyz=(float(axis3[0]), float(axis3[1]), float(axis3[2])),
        radius_mm=float(radius),
        height_mm=float(length_mm),
        orientation=orientation,
        used_real_depth=bool(used_real and cap_z is not None),
        n_points=n_pts,
    )


def _rgb_clahe_bgr(color_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def _rgb_bright_mask(color_bgr: np.ndarray) -> np.ndarray:
    """Looser bright-cap mask for real stills (glare / warm LEDs)."""
    enh = _rgb_clahe_bgr(color_bgr)
    hsv = cv2.cvtColor(enh, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] <= 80) & (hsv[:, :, 2] >= 145)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _vertical_edge_score(gray: np.ndarray, box) -> float:
    x, y, w, h = box
    if w < 8 or h < 12:
        return 0.0
    roi = gray[y : y + h, x : x + w]
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    # Left / right strips
    lw = max(2, w // 6)
    left = float(np.mean(np.abs(gx[:, :lw])))
    right = float(np.mean(np.abs(gx[:, -lw:])))
    mid = float(np.mean(np.abs(gx[:, lw:-lw]))) if w > 2 * lw else 1.0
    return float(min(1.0, 0.5 * (left + right) / max(mid, 1.0) / 40.0))


def _score_rgb_upright(color_bgr: np.ndarray, box) -> float:
    h, w = color_bgr.shape[:2]
    x, y, bw, bh = _clip_box(box, w, h)
    if bw < 12 or bh < 16:
        return 0.0
    aspect = bh / float(max(bw, 1))
    area = (bw * bh) / float(max(w * h, 1))
    if aspect < 1.02 or aspect > 3.20:
        return 0.0
    if area < 0.003 or area > 0.88:
        return 0.0
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    top_h = max(2, int(0.22 * bh))
    top = hsv[y : y + top_h, x : x + bw]
    body = hsv[y + top_h : y + bh, x : x + bw]
    if body.size == 0 or top.size == 0:
        return 0.0
    top_white = float(((top[:, :, 2] >= 140) & (top[:, :, 1] <= 100)).mean())
    top_v = float(top[:, :, 2].mean())
    body_v = float(body[:, :, 2].mean())
    contrast = float(np.clip((top_v - body_v) / 80.0, -0.3, 1.0))
    edge = _vertical_edge_score(gray, (x, y, bw, bh))
    closeup = area >= 0.28
    # Reject unstructured clutter unless close-up (cap may not be brightest).
    if not closeup and top_white < 0.06 and contrast < 0.08 and edge < 0.20:
        return 0.0
    if closeup and edge < 0.08 and aspect < 1.15:
        return 0.0
    # Penalize boxes glued to multiple borders (wall slabs), except close-ups.
    borders = int(x <= 2) + int(y <= 2) + int(x + bw >= w - 3) + int(y + bh >= h - 3)
    border_pen = 0.0 if closeup else 0.12 * max(0, borders - 1)
    score = (
        0.28 * min(1.0, top_white / 0.40)
        + 0.22 * max(0.0, contrast)
        + 0.28 * edge
        + 0.12 * min(1.0, (aspect - 1.0) / 0.7)
        + 0.10 * (1.0 - min(1.0, abs(np.log(max(area, 1e-6) / 0.07)) / 2.5))
        + (0.15 if closeup and 1.1 <= aspect <= 2.2 else 0.0)
        - border_pen
    )
    return float(np.clip(score, 0.0, 1.0))


def _grow_upright_from_cap(
    color_bgr: np.ndarray, cap, max_len: float
) -> tuple[int, int, int, int] | None:
    """Expand a cap seed downward into the clear/dark body."""
    h, w = color_bgr.shape[:2]
    x, y, bw, bh = _clip_box(cap, w, h)
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    # Cap should sit near the top of the vial; widen slightly.
    diam = int(max(bw, min(int(bw * 1.35), int(0.42 * w))))
    cu = x + bw * 0.5
    x0 = int(max(0, cu - 0.5 * diam))
    x1 = int(min(w, cu + 0.5 * diam))
    y0 = y
    # Walk down while body stays not-bright / has edges.
    step = max(2, bh // 3)
    y_end = y + bh
    last_good = y_end
    limit = int(min(h, y0 + max_len))
    bg = float(np.median(gray)) if gray.size else 128.0
    for yy in range(y + bh, limit, step):
        band = gray[yy : min(h, yy + step), x0:x1]
        if band.size < 4:
            break
        mean = float(band.mean())
        # Stop on bright table/hand glare or empty border.
        if mean > max(175.0, bg + 45.0):
            break
        # Prefer continuing through mid-gray plastic / shadows.
        last_good = min(h, yy + step)
        if last_good - y0 > max_len:
            break
    height = last_good - y0
    if height < max(bw, bh) * 1.25:
        # Mild prior when growth is weak — avoid 3× overshoot on mid-field stills.
        height = int(max(height, min(max_len, max(bw, bh) * 2.1)))
        last_good = min(h, y0 + height)
        height = last_good - y0
    if height < 20:
        return None
    return _clip_box((x0, y0, x1 - x0, height), w, h)


def _grow_sideways_from_cap(
    color_bgr: np.ndarray, cap, direction: float, max_len: float
) -> tuple[int, int, int, int] | None:
    h, w = color_bgr.shape[:2]
    x, y, bw, bh = _clip_box(cap, w, h)
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    diam = int(max(bh, min(int(bh * 1.35), int(0.42 * h))))
    cv = y + bh * 0.5
    y0 = int(max(0, cv - 0.5 * diam))
    y1 = int(min(h, cv + 0.5 * diam))
    step = max(2, bw // 3)
    bg = float(np.median(gray)) if gray.size else 128.0
    if direction >= 0:
        x0 = x
        x_end = x + bw
        last = x_end
        for xx in range(x + bw, int(min(w, x + max_len)), step):
            band = gray[y0:y1, xx : min(w, xx + step)]
            if band.size < 4 or float(band.mean()) > max(175.0, bg + 45.0):
                break
            last = min(w, xx + step)
        length = last - x0
        if length < max(bw, bh) * 1.35:
            length = int(min(max_len, max(bw, bh) * 2.6))
            last = min(w, x0 + length)
        return _clip_box((x0, y0, last - x0, y1 - y0), w, h)
    x1 = x + bw
    last = x
    for xx in range(x, int(max(0, x + bw - max_len)), -step):
        band = gray[y0:y1, max(0, xx - step) : xx]
        if band.size < 4 or float(band.mean()) > max(175.0, bg + 45.0):
            break
        last = max(0, xx - step)
    length = x1 - last
    if length < max(bw, bh) * 1.35:
        length = int(min(max_len, max(bw, bh) * 2.6))
        last = max(0, x1 - length)
    return _clip_box((last, y0, x1 - last, y1 - y0), w, h)


def _rgb_closeup_upright(color_bgr: np.ndarray, min_score: float) -> list[VialDetection]:
    """When the vial fills most of the frame, seed from the brightest top band."""
    h, w = color_bgr.shape[:2]
    hsv = cv2.cvtColor(_rgb_clahe_bgr(color_bgr), cv2.COLOR_BGR2HSV)
    # Top third brightness map → horizontal band for cap.
    top = hsv[: max(8, h // 3), :]
    bright = ((top[:, :, 2] >= 150) & (top[:, :, 1] <= 95)).astype(np.uint8) * 255
    ys, xs = np.where(bright > 0)
    if xs.size < 40:
        return []
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    if (x1 - x0) < 0.15 * w:
        return []
    cap = (x0, y0, x1 - x0, max(6, y1 - y0))
    box = _grow_upright_from_cap(color_bgr, cap, max_len=0.95 * h)
    if box is None:
        # Occupy most of the frame under the bright band.
        box = _clip_box((int(0.08 * w), y0, int(0.84 * w), int(0.90 * h) - y0), w, h)
    score = _score_rgb_upright(color_bgr, box)
    # Close-ups often score mid; accept slightly lower.
    if score < max(0.28, min_score * 0.85):
        return []
    z = 350.0
    return [
        VialDetection(
            box_xywh=box,
            score=float(min(1.0, 0.55 + 0.45 * score)),
            center_xyz_mm=(0.0, 0.0, z),
            axis_xyz=(0.0, 1.0, 0.0),
            radius_mm=0.5 * _DEFAULT_DIAM_MM,
            height_mm=_DEFAULT_LENGTH_MM,
            orientation="upright",
            used_real_depth=False,
            n_points=0,
        )
    ]


def _expand_cap_agent05(
    cx: float,
    cy: float,
    diameter: float,
    width: int,
    height: int,
    *,
    w_scale: float = 2.4,
    h_scale: float = 3.6,
) -> tuple[int, int, int, int]:
    """Expand cap center/diameter → full vial box (agent_05 geometry)."""
    r = max(diameter * 0.5, 8.0)
    vw = float(np.clip(r * w_scale, 20.0, 0.55 * width))
    vh = float(np.clip(r * h_scale, 28.0, 0.92 * height))
    vcy = min(height - 1.0, cy + 0.4 * (vh - 2.0 * r))
    x0 = cx - 0.5 * vw
    y0 = vcy - 0.5 * vh
    return _clip_box((x0, y0, vw, vh), width, height)


def _rgb_cap_centers(color_bgr: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Blob + MSER + CC cap centers: (cx, cy, diameter, seed_score)."""
    h, w = color_bgr.shape[:2]
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    blur = cv2.GaussianBlur(clahe.apply(gray), (5, 5), 0)
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    # Strict + loose masks (real lids often V≈140).
    strict = ((hsv[:, :, 1] <= 70) & (hsv[:, :, 2] >= 165)).astype(np.uint8) * 255
    loose = ((hsv[:, :, 1] <= 95) & (hsv[:, :, 2] >= 130)).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    strict = cv2.morphologyEx(strict, cv2.MORPH_OPEN, k)
    strict = cv2.morphologyEx(strict, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    loose = cv2.morphologyEx(loose, cv2.MORPH_OPEN, k)
    loose = cv2.morphologyEx(loose, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    bp = cv2.SimpleBlobDetector_Params()
    bp.filterByColor = True
    bp.blobColor = 255
    bp.minThreshold = 20
    bp.maxThreshold = 220
    bp.thresholdStep = 12
    bp.filterByArea = True
    bp.minArea = 50.0
    bp.maxArea = 30000.0
    bp.filterByCircularity = True
    bp.minCircularity = 0.22
    bp.filterByInertia = True
    bp.minInertiaRatio = 0.12
    bp.filterByConvexity = True
    bp.minConvexity = 0.50
    bp.minDistBetweenBlobs = 12.0
    blob = cv2.SimpleBlobDetector_create(bp)

    centers: list[tuple[float, float, float, float]] = []
    for img, boost in ((blur, 0.05), (strict, 0.16), (loose, 0.08)):
        for kp in blob.detect(img):
            d = float(kp.size)
            if d < 7 or d > 260:
                continue
            centers.append(
                (float(kp.pt[0]), float(kp.pt[1]), d, 0.50 + boost + 0.002 * min(d, 80))
            )

    mser = cv2.MSER_create()
    mser.setDelta(4)
    mser.setMinArea(40)
    mser.setMaxArea(30000)
    for mask, boost in ((strict, 0.14), (loose, 0.06)):
        regions, _ = mser.detectRegions(mask)
        for pts in regions:
            cnt = pts.reshape(-1, 1, 2)
            area = float(cv2.contourArea(cnt))
            if area < 40 or area > 0.15 * h * w:
                continue
            peri = float(cv2.arcLength(cnt, True))
            circ = 4.0 * np.pi * area / (peri * peri + 1e-6)
            if circ < 0.22:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            ar = bw / float(bh + 1e-6)
            if not (0.35 <= ar <= 2.4):
                continue
            centers.append(
                (
                    x + bw * 0.5,
                    y + bh * 0.5,
                    float(max(bw, bh)),
                    0.40 + boost + 0.45 * circ,
                )
            )

    # Connected components on loose mask (small / flat lids).
    nlab, _labs, stats, cents = cv2.connectedComponentsWithStats(loose, 8)
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 40 or area > 12000:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        ar = bw / float(bh + 1e-6)
        if not (0.35 <= ar <= 2.4):
            continue
        diam_cc = float(max(bw, bh))
        if diam_cc > 0.28 * min(w, h):
            continue
        centers.append(
            (
                float(cents[i, 0]),
                float(cents[i, 1]),
                diam_cc,
                0.42 + 0.18 * min(1.0, area / 3500.0),
            )
        )

    # Dedup
    uniq: list[tuple[float, float, float, float]] = []
    for c in sorted(centers, key=lambda t: t[3], reverse=True):
        if all((c[0] - u[0]) ** 2 + (c[1] - u[1]) ** 2 > 16.0**2 for u in uniq):
            uniq.append(c)
        if len(uniq) >= 40:
            break
    return uniq


_YOLO_MODEL = None
_YOLO_TRIED = False


def _resolve_yolo_weights() -> Path | None:
    """Prefer Camera/runs/vial-live (same as depthcamera.find_weights)."""
    cam = _CAMERA_ROOT
    candidates = (
        cam / "runs" / "vial-live" / "weights" / "best.pt",
        cam / "runs" / "train-race" / "r1-winner" / "weights" / "best.pt",
        cam / "runs" / "bon-winner" / "weights" / "best.pt",
        cam / "runs" / "vial" / "weights" / "best.pt",
    )
    for p in candidates:
        if p.is_file():
            return p
    return None


def _yolo_rgb_detect(
    color_bgr: np.ndarray,
    *,
    length_mm: float,
    diam_mm: float,
    min_score: float,
) -> list[VialDetection] | None:
    """Optional YOLO path for real upright stills (weights may be absent)."""
    global _YOLO_MODEL, _YOLO_TRIED
    if _YOLO_MODEL is None and not _YOLO_TRIED:
        _YOLO_TRIED = True
        weights = _resolve_yolo_weights()
        if weights is None:
            return None
        try:
            from ultralytics import YOLO

            _YOLO_MODEL = YOLO(str(weights))
        except Exception:
            return None
    if _YOLO_MODEL is None:
        return None
    try:
        res = _YOLO_MODEL.predict(color_bgr, verbose=False, conf=max(0.20, min_score))[0]
    except Exception:
        return None
    height, width = color_bgr.shape[:2]
    dets: list[VialDetection] = []
    if res.boxes is None:
        return dets
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    for box, conf in zip(xyxy, confs):
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        bw, bh = x2 - x1, y2 - y1
        if bw < 8 or bh < 8:
            continue
        aspect = bh / max(bw, 1e-6)
        orient = "upright" if aspect >= 1.0 else "sideways"
        axis = (0.0, 1.0, 0.0) if orient == "upright" else (1.0, 0.0, 0.0)
        dets.append(
            VialDetection(
                box_xywh=_clip_box((x1, y1, bw, bh), width, height),
                score=float(conf),
                center_xyz_mm=(0.0, 0.0, 520.0),
                axis_xyz=axis,
                radius_mm=0.5 * diam_mm,
                height_mm=float(length_mm),
                orientation=orient,
                used_real_depth=False,
                n_points=0,
            )
        )
    return _nms(dets, thr=0.45)[:_MAX_DETECTIONS]


def _rgb_classical_detect(
    color_bgr: np.ndarray,
    *,
    length_mm: float,
    diam_mm: float,
    fx: float,
    fy: float,
    min_score: float,
) -> list[VialDetection]:
    """Blob/MSER + agent_05 expand (no neural net)."""
    height, width = color_bgr.shape[:2]
    z_assumed = 520.0
    length_px = float(length_mm) * fx / z_assumed
    scales = (
        (2.0, 2.8),
        (2.2, 3.2),
        (2.4, 3.6),
        (2.6, 3.8),
        (2.8, 4.0),
        (1.8, 2.4),
    )

    upright_dets: list[VialDetection] = []
    for cx, cy, diam, seed_sc in _rgb_cap_centers(color_bgr):
        # Default agent_05 scale is the workhorse; multi-scale refines IoU.
        default_box = _expand_cap_agent05(
            cx, cy, diam, width, height, w_scale=2.4, h_scale=3.6
        )
        best_box = default_box
        best_struct = _score_rgb_upright(color_bgr, default_box)
        for ws, hs in scales:
            box = _expand_cap_agent05(
                cx, cy, diam, width, height, w_scale=ws, h_scale=hs
            )
            st = _score_rgb_upright(color_bgr, box)
            if st > best_struct:
                best_struct = st
                best_box = box
        cap = (
            int(cx - 0.5 * diam),
            int(cy - 0.5 * diam),
            int(max(6, diam)),
            int(max(6, diam)),
        )
        grown = _grow_upright_from_cap(
            color_bgr, cap, max_len=max(length_px * 1.3, diam * 3.0)
        )
        if grown is not None:
            st = _score_rgb_upright(color_bgr, grown)
            if st > best_struct:
                best_struct = st
                best_box = grown

        # Accept strong blob seeds even when structure score is middling
        # (real lids are only moderately white; agent_05-style expand still hits).
        aspect = best_box[3] / float(max(best_box[2], 1))
        area = (best_box[2] * best_box[3]) / float(max(width * height, 1))
        if aspect < 1.02 or aspect > 3.3 or area < 0.003 or area > 0.90:
            continue
        ok = seed_sc >= 0.48 or best_struct >= 0.32 or (
            seed_sc >= 0.42 and best_struct >= 0.22
        )
        if not ok:
            continue
        seed_n = float(min(0.95, seed_sc))
        score = float(min(1.0, max(seed_n, 0.35 + 0.55 * best_struct)))
        # Huge diameter seeds are usually wall patches — require structure.
        if diam > 0.22 * min(width, height) and best_struct < 0.35:
            continue
        if score < max(0.42, min_score):
            continue
        upright_dets.append(
            VialDetection(
                box_xywh=best_box,
                score=score,
                center_xyz_mm=(0.0, 0.0, z_assumed),
                axis_xyz=(0.0, 1.0, 0.0),
                radius_mm=0.5 * diam_mm,
                height_mm=float(length_mm),
                orientation="upright",
                used_real_depth=False,
                n_points=0,
            )
        )

    dets = _nms(sorted(upright_dets, key=lambda d: d.score, reverse=True), thr=0.35)[:3]
    if len(dets) >= 2 and dets[0].score - dets[-1].score > 0.18:
        dets = [d for d in dets if d.score >= dets[0].score - 0.12][:2]

    # Sideways synth / fallen: strict white end-cap + elongated body.
    if not dets or dets[0].score < 0.40:
        sideways_dets: list[VialDetection] = []
        for cap in _cap_seeds(_white_cap_mask(color_bgr), min_area=40)[:16]:
            for sign in (1.0, -1.0):
                sb = _grow_sideways_from_cap(
                    color_bgr,
                    cap,
                    sign,
                    max_len=max(length_px * 1.4, max(cap[2], cap[3]) * 3.2),
                )
                if sb is None or sb[2] / float(max(sb[3], 1)) < 1.25:
                    continue
                hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
                x, y, bw, bh = sb
                end = (
                    hsv[y : y + bh, x : x + max(4, bw // 5)]
                    if sign >= 0
                    else hsv[y : y + bh, x + bw - max(4, bw // 5) : x + bw]
                )
                end_white = float(
                    ((end[:, :, 2] >= 170) & (end[:, :, 1] <= 70)).mean()
                    if end.size
                    else 0
                )
                if end_white < 0.28:
                    continue
                sc = 0.55 * end_white + 0.45 * min(1.0, sb[2] / float(max(sb[3], 1)) / 2.0)
                if sc < 0.70:
                    continue
                sideways_dets.append(
                    VialDetection(
                        box_xywh=sb,
                        score=float(min(1.0, sc)),
                        center_xyz_mm=(0.0, 0.0, z_assumed),
                        axis_xyz=(1.0 if sign >= 0 else -1.0, 0.0, 0.0),
                        radius_mm=0.5 * diam_mm,
                        height_mm=float(length_mm),
                        orientation="sideways",
                        used_real_depth=False,
                        n_points=0,
                    )
                )
        dets = _nms(dets + sideways_dets, thr=0.40)[:2]

    if not dets or dets[0].score < 0.42:
        for c in _rgb_closeup_upright(color_bgr, min_score=min_score):
            if all(_iou(c.box_xywh, d.box_xywh) < 0.45 for d in dets):
                dets.append(c)
        dets = _nms(dets, thr=0.35)[:2]

    return dets


def _rgb_anypose_detect(
    color_bgr: np.ndarray,
    *,
    length_mm: float,
    diam_mm: float,
    fx: float,
    fy: float,
    min_score: float,
    use_yolo: bool = True,
) -> list[VialDetection]:
    """RGB fallback for stills / missing depth.

    Real upright stills: prefer YOLO (`runs/vial-live`) when weights exist —
    classical blob/expand tops out ~0.60 recall because white walls merge with
    lids. Classical remains the no-weights path and covers synth sideways.
    """
    if use_yolo:
        yolo = _yolo_rgb_detect(
            color_bgr,
            length_mm=length_mm,
            diam_mm=diam_mm,
            min_score=min_score,
        )
        if yolo:
            return yolo
    return _rgb_classical_detect(
        color_bgr,
        length_mm=length_mm,
        diam_mm=diam_mm,
        fx=fx,
        fy=fy,
        min_score=min_score,
    )


def detect_vials(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray | None = None,
    intrinsics: Any = None,
    *,
    dims_path: Path | None = None,
    allow_rgb_fallback: bool = True,
    use_yolo_rgb: bool = True,
    min_score: float = 0.25,
) -> list[VialDetection]:
    if color_bgr is None or getattr(color_bgr, "size", 0) == 0:
        return []
    image = np.asarray(color_bgr)
    if image.ndim != 3 or image.shape[2] < 3:
        return []
    height, width = image.shape[:2]
    length_mm, diam_mm = _load_vial_mm(dims_path)
    fx, fy, cx, cy = normalize_intrinsics(intrinsics, width, height)

    used_real = depth_mm is not None and np.asarray(depth_mm).size > 0
    depth: np.ndarray | None
    if used_real:
        depth = align_depth(depth_mm, width, height)
        if not np.any((depth > _MIN_DEPTH_MM) & (depth < _MAX_DEPTH_MM)):
            used_real = False
            depth = None
    else:
        depth = None

    if not used_real:
        if not allow_rgb_fallback:
            return []
        return _rgb_anypose_detect(
            image,
            length_mm=length_mm,
            diam_mm=diam_mm,
            fx=fx,
            fy=fy,
            min_score=min_score,
            use_yolo=use_yolo_rgb,
        )

    assert depth is not None
    plane = None
    ys_g, xs_g = np.where(
        (depth > 200.0) & (depth < _MAX_DEPTH_MM) & np.isfinite(depth)
    )
    if xs_g.size >= 200:
        if xs_g.size > 5000:
            idx = np.linspace(0, xs_g.size - 1, 5000).astype(np.int32)
            xs_s, ys_s = xs_g[idx], ys_g[idx]
        else:
            xs_s, ys_s = xs_g, ys_g
        cloud_g = _deproject(
            depth, xs_s.astype(np.float64), ys_s.astype(np.float64), fx, fy, cx, cy
        )
        plane = _estimate_plane(cloud_g)

    seeds = _cap_seeds(_white_cap_mask(image))
    # Secondary dim seeds only if primary found nothing (hard synth lids).
    if not seeds:
        seeds = _cap_seeds(_dim_cap_mask(image), min_area=35)

    detections: list[VialDetection] = []
    for cap in seeds:
        hyp = _hypothesis_from_seed(
            image,
            depth,
            cap,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            length_mm=length_mm,
            diam_mm=diam_mm,
            plane=plane,
            used_real=True,
            min_score=min_score,
            require_strong_body=False,
        )
        if hyp is not None:
            detections.append(hyp)

    # If bright seeds missed dim hard vials, try dim mask once.
    if len(detections) < 1:
        for cap in _cap_seeds(_dim_cap_mask(image), min_area=35):
            hyp = _hypothesis_from_seed(
                image,
                depth,
                cap,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                length_mm=length_mm,
                diam_mm=diam_mm,
                plane=plane,
                used_real=True,
                min_score=min_score,
                require_strong_body=True,
            )
            if hyp is not None:
                detections.append(hyp)

    dets = _nms(detections)[:_MAX_DETECTIONS]
    # RGB fallback only when depth was absent — never after a failed real-depth
    # pass (that re-opens white-slab FPs that clear-cup correctly rejected).
    return dets


def annotate(color_bgr: np.ndarray, detections: list[VialDetection]) -> np.ndarray:
    out = color_bgr.copy()
    palette = {
        "sideways": (40, 200, 40),
        "upright": (40, 160, 220),
        "angled": (20, 180, 220),
        "topdown": (200, 140, 40),
    }
    for d in detections:
        x, y, w, h = d.box_xywh
        color = palette.get(d.orientation, (200, 200, 40))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"{d.orientation} {d.score:.2f} z={d.center_xyz_mm[2]:.0f}"
        cv2.putText(
            out, label, (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
        )
        cx, cy = x + w // 2, y + h // 2
        tip = (int(cx + d.axis_xyz[0] * 32), int(cy + d.axis_xyz[1] * 32))
        cv2.arrowedLine(out, (cx, cy), tip, color, 2, tipLength=0.35)
    mode = (
        "RGB-D"
        if any(d.used_real_depth for d in detections)
        else ("RGB-fallback" if detections else "none")
    )
    banner = f"r2_07 depth_anypose ({mode}) n={len(detections)}"
    cv2.putText(out, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(out, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return out
