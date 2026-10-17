"""Cap → base axis skeleton and oriented box construction."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .cap_finder import CapBlob
from .types import Det, Keypoint


def _unit(dx: float, dy: float) -> Tuple[float, float]:
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return (1.0, 0.0)
    return (dx / n, dy / n)


def oriented_corners(
    cx: float, cy: float, length: float, diameter: float, angle_deg: float
) -> np.ndarray:
    axis = np.array(
        [math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))],
        dtype=np.float32,
    )
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    c = np.array([cx, cy], dtype=np.float32)
    return np.stack(
        (
            c - 0.5 * length * axis - 0.5 * diameter * normal,
            c + 0.5 * length * axis - 0.5 * diameter * normal,
            c + 0.5 * length * axis + 0.5 * diameter * normal,
            c - 0.5 * length * axis + 0.5 * diameter * normal,
        )
    )


def aabb_from_corners(corners: np.ndarray, w: int, h: int) -> Tuple[float, float, float, float]:
    xs = corners[:, 0]
    ys = corners[:, 1]
    x1 = float(np.clip(np.floor(xs.min()), 0, w - 1))
    y1 = float(np.clip(np.floor(ys.min()), 0, h - 1))
    x2 = float(np.clip(np.ceil(xs.max()), 0, w))
    y2 = float(np.clip(np.ceil(ys.max()), 0, h))
    return x1, y1, x2, y2


def pose_from_angle(angle_deg: float) -> str:
    a = abs(((angle_deg + 90.0) % 180.0) - 90.0)  # 0 = horizontal, 90 = vertical
    # Convert to distance from horizontal.
    from_h = min(abs(angle_deg % 180.0), abs((angle_deg % 180.0) - 180.0))
    from_h = min(from_h, 180.0 - from_h)
    if from_h <= 30.0:
        return "sideways"
    if from_h >= 60.0:
        return "upright"
    return "angled"


def det_from_keypoints(
    cap_xy: Tuple[float, float],
    base_xy: Tuple[float, float],
    diameter: float,
    *,
    conf: float,
    image_shape: Tuple[int, int],
    source: str = "keypoint_axis",
) -> Det:
    h, w = image_shape
    cxy = np.array(cap_xy, dtype=np.float32)
    bxy = np.array(base_xy, dtype=np.float32)
    vec = bxy - cxy
    span = float(np.linalg.norm(vec))
    # Cap-to-base span already covers lid through cup; small pad only.
    length = max(span + 0.20 * diameter, diameter * 1.05)
    diam = float(max(diameter * 0.95, 4.0))
    angle_deg = float(math.degrees(math.atan2(float(vec[1]), float(vec[0]))))
    center = 0.5 * (cxy + bxy)
    corners = oriented_corners(
        float(center[0]), float(center[1]), length, diam, angle_deg
    )
    x1, y1, x2, y2 = aabb_from_corners(corners, w, h)
    return Det(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        conf=float(conf),
        cx=float(center[0]),
        cy=float(center[1]),
        length=float(length),
        diameter=diam,
        angle_deg=angle_deg,
        cap=Keypoint(x=float(cap_xy[0]), y=float(cap_xy[1]), name="cap", conf=conf),
        base=Keypoint(x=float(base_xy[0]), y=float(base_xy[1]), name="base", conf=conf),
        pose=pose_from_angle(angle_deg),
        source=source,
    )


def candidate_axes(cap: CapBlob) -> List[Tuple[float, float]]:
    """Unit directions to search for the far end / base.

    Priors bias the search but never drop the opposite family entirely —
    Round-1 failed by assuming upright-only growth.
    """
    if cap.kind == "side_on":
        degs = (
            90,
            270,
            75,
            105,
            255,
            285,
            60,
            120,
            0,
            180,
            15,
            -15,
            165,
            195,
            30,
            -30,
        )
    elif cap.kind == "end_cap":
        degs = (
            0,
            180,
            15,
            -15,
            165,
            195,
            30,
            -30,
            150,
            210,
            45,
            -45,
            90,
            270,
            75,
            105,
        )
    else:
        degs = tuple(range(0, 360, 15))
    return [(math.cos(math.radians(d)), math.sin(math.radians(d))) for d in degs]


def _bg_level(gray: np.ndarray, origin: Tuple[float, float], diameter: float) -> float:
    h, w = gray.shape
    r = int(max(8, diameter * 1.8))
    x0, y0 = int(origin[0]), int(origin[1])
    ys, xs = [], []
    for ang in range(0, 360, 45):
        px = int(x0 + r * math.cos(math.radians(ang)))
        py = int(y0 + r * math.sin(math.radians(ang)))
        if 0 <= px < w and 0 <= py < h:
            xs.append(px)
            ys.append(py)
    if not xs:
        return float(np.median(gray))
    return float(np.median(gray[ys, xs]))


def _march_axis(
    gray: np.ndarray,
    edges: np.ndarray,
    sat: np.ndarray,
    val: np.ndarray,
    depth: Optional[np.ndarray],
    origin: Tuple[float, float],
    axis: Tuple[float, float],
    diameter: float,
    expected_aspect: float,
    cap_z: Optional[float],
    bg: float,
) -> Tuple[float, float, Tuple[float, float]]:
    """Ray-march from cap; return (score, length, base_xy)."""
    h, w = gray.shape
    ux, uy = axis
    nx, ny = -uy, ux
    step = max(2.0, diameter * 0.10)
    # Keep length near physical vial aspect; avoid runaway into desk.
    max_len = diameter * min(1.85, expected_aspect * 1.25)
    min_len = diameter * 0.85
    start = diameter * 0.35
    t = start
    good = 0
    hole = 0
    dark = 0
    white = 0
    total = 0
    last_good_t = start
    bad_streak = 0
    while t <= max_len:
        cx = origin[0] + ux * t
        cy = origin[1] + uy * t
        if not (0 <= cx < w and 0 <= cy < h):
            break
        slice_ok = 0
        for s in (-0.28, 0.0, 0.28):
            px = int(round(cx + nx * diameter * s))
            py = int(round(cy + ny * diameter * s))
            if not (0 <= px < w and 0 <= py < h):
                continue
            total += 1
            g = float(gray[py, px])
            s_val = float(sat[py, px])
            v_val = float(val[py, px])
            if v_val >= 210 and s_val <= 50:
                white += 1
                continue
            is_hole = False
            if depth is not None:
                z = float(depth[py, px])
                is_hole = (not np.isfinite(z)) or z <= 0
            # Clear cup is darker than nearby desk OR a depth hole.
            is_dark = g <= min(bg - 12.0, 95.0)
            if is_hole:
                hole += 1
            if is_dark:
                dark += 1
            if is_dark or is_hole:
                slice_ok += 1
        if slice_ok >= 2:
            good += 1
            last_good_t = t
            bad_streak = 0
        else:
            bad_streak += 1
            if t >= min_len and bad_streak >= 2:
                break
        t += step

    length = float(np.clip(last_good_t, min_len, max_len))
    n_slices = max(1, int(round((length - start) / step)) + 1)
    if good < max(3, int(0.40 * n_slices)):
        return -1.0, length, (origin[0] + ux * length, origin[1] + uy * length)

    outside = 0.0
    inside = 0.0
    n_cmp = 0
    for tt in np.linspace(start, length, num=max(4, int(length / 6))):
        for s_in, s_out in ((0.0, 0.75), (0.0, -0.75)):
            ix = int(round(origin[0] + ux * tt + nx * diameter * s_in))
            iy = int(round(origin[1] + uy * tt + ny * diameter * s_in))
            ox = int(round(origin[0] + ux * tt + nx * diameter * s_out))
            oy = int(round(origin[1] + uy * tt + ny * diameter * s_out))
            if 0 <= ix < w and 0 <= iy < h and 0 <= ox < w and 0 <= oy < h:
                inside += float(gray[iy, ix])
                outside += float(gray[oy, ox])
                n_cmp += 1
    contrast = 0.0
    if n_cmp:
        contrast = (outside - inside) / max(n_cmp, 1) / 40.0

    hole_frac = hole / max(total, 1)
    dark_frac = dark / max(total, 1)
    white_frac = white / max(total, 1)
    aspect = length / max(diameter, 1.0)
    score = (
        1.1 * dark_frac
        + 1.6 * hole_frac
        + 0.9 * float(np.clip(contrast, -0.5, 1.5))
        - 1.6 * white_frac
        + (0.30 if 0.90 <= aspect <= 1.70 else -0.20)
    )
    # Depth available but almost no holes → likely opaque clutter, not clear cup.
    if depth is not None and hole_frac < 0.18 and dark_frac < 0.45:
        score -= 0.8
    # RGB-only: lean on darkness + lateral contrast (no hole cue).
    if depth is None:
        score = (
            1.35 * dark_frac
            + 1.1 * float(np.clip(contrast, -0.5, 1.5))
            - 1.6 * white_frac
            + (0.25 if 0.90 <= aspect <= 1.70 else -0.15)
        )
    base = (float(origin[0] + ux * length), float(origin[1] + uy * length))
    return float(score), length, base


def estimate_base(
    bgr: np.ndarray,
    cap: CapBlob,
    *,
    depth_mm: Optional[np.ndarray] = None,
    expected_aspect: float = 1.34,
    min_score: float = 0.85,
    min_margin: float = 0.04,
) -> Optional[Tuple[Tuple[float, float], float, Tuple[float, float]]]:
    """Return (base_xy, score, axis_unit) for best far-end estimate.

    Requires the winning axis to beat the runner-up by ``min_margin`` so
    isotropic white blobs (walls / top-down lids) do not invent a vial axis.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 110)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    depth = None
    if depth_mm is not None:
        d = np.asarray(depth_mm, dtype=np.float32)
        if d.shape[:2] != (h, w):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
        depth = d
    if cap.kind == "side_on":
        diam = float(max(cap.w, 8))
    else:
        diam = float(max(cap.h if cap.w < cap.h else cap.w, 8))
        # End-cap: diameter ≈ the larger lateral extent of the lid face.
        diam = float(max(cap.h, cap.w * 0.9, 8))
    origin = (cap.cx, cap.cy)
    bg = _bg_level(gray, origin, diam)
    cap_z = None
    if depth is not None:
        r = max(2, int(diam // 4))
        x0, y0 = int(cap.cx), int(cap.cy)
        patch = depth[max(0, y0 - r) : y0 + r + 1, max(0, x0 - r) : x0 + r + 1]
        valid = patch[(patch > 80) & (patch < 5000)]
        if valid.size >= 4:
            cap_z = float(np.median(valid))

    scored: List[Tuple[float, Tuple[float, float], Tuple[float, float], float]] = []
    for axis in candidate_axes(cap):
        score, length, base = _march_axis(
            gray, edges, sat, val, depth, origin, axis, diam, expected_aspect, cap_z, bg
        )
        if score < 0:
            continue
        scored.append((score, base, axis, length))

    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)
    best_score, best_base, best_axis, best_len = scored[0]
    second = scored[1][0] if len(scored) > 1 else -1.0
    margin = best_score - second

    if best_score < min_score:
        return None
    # Top-down lids: nearly equal scores in many directions.
    if margin < min_margin and best_score < 1.35:
        return None
    # Reject absurdly short / long bodies relative to diameter.
    aspect = best_len / max(diam, 1.0)
    if aspect < 0.65 or aspect > 2.3:
        return None
    return best_base, float(best_score), best_axis


def far_end_from_box(
    cap_xy: Tuple[float, float],
    xyxy: Sequence[float],
) -> Tuple[float, float]:
    """Weak base label: opposite corner of AABB from the cap along the long axis."""
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    if bh >= bw:
        # Upright-ish: base is the vertical end opposite the cap.
        base_y = y2 - 1.0 if cap_xy[1] < cy else y1 + 1.0
        return (float(np.clip(cap_xy[0], x1 + 1, x2 - 1)), float(base_y))
    base_x = x2 - 1.0 if cap_xy[0] < cx else x1 + 1.0
    return (float(base_x), float(np.clip(cap_xy[1], y1 + 1, y2 - 1)))


def nms_dets(dets: List[Det], iou_thr: float = 0.4) -> List[Det]:
    from .types import box_iou

    ordered = sorted(dets, key=lambda d: d.conf, reverse=True)
    kept: List[Det] = []
    for d in ordered:
        if all(box_iou(d, k) < iou_thr for k in kept):
            kept.append(d)
    return kept
