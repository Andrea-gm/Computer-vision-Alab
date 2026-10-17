"""Track pale baby-blue / cyan UR joint plastic via HSV color mask (no NN)."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# Seeded from user baby-blue joint crops (OpenCV HSV: H 0-179).
# Slightly relaxed for pale / highlighted caps; mean-S gate still rejects white panels.
DEFAULT_HSV_LO = (88, 36, 140)
DEFAULT_HSV_HI = (104, 180, 255)
DEFAULT_MIN_AREA = 220
DEFAULT_MIN_MEAN_S = 42.0
DEFAULT_MIN_FILL = 0.18
DEFAULT_MAX_BOXES = 12
DEFAULT_PROXIMITY_MM = 100.0  # 10 cm
# Lab equipment obstacle colors (mask is never drawn):
# - matte black / near-black
# - cool light-grey / aluminum extrusion (seeded from user crop)
DEFAULT_BLACK_V_MAX = 58
DEFAULT_BLACK_S_MAX = 255
DEFAULT_SILVER_HSV_LO = (85, 22, 105)
DEFAULT_SILVER_HSV_HI = (115, 100, 220)
BOX_COLOR = (255, 220, 80)
MASK_TINT = (255, 200, 60)
ALERT_COLOR = (0, 0, 255)  # BGR red


def baby_blue_mask(
    bgr: np.ndarray,
    *,
    hsv_lo: tuple[int, int, int] = DEFAULT_HSV_LO,
    hsv_hi: tuple[int, int, int] = DEFAULT_HSV_HI,
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8))
    # Prefer slightly open to kill speckles on pale equipment.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def median_depth_mm(
    depth_mm: Optional[np.ndarray],
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    mask: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Median depth (mm) in box; ignores zeros / invalid. Depth must be mm."""
    if depth_mm is None or depth_mm.size == 0:
        return None
    dh, dw = depth_mm.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dw, x + w), min(dh, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = depth_mm[y0:y1, x0:x1].astype(np.float32)
    if mask is not None:
        m = mask[y0:y1, x0:x1]
        if m.shape[:2] == patch.shape[:2]:
            vals = patch[m > 0]
        else:
            vals = patch.reshape(-1)
    else:
        # Center crop is more stable than full box edges
        ch, cw = patch.shape[:2]
        vals = patch[ch // 4 : 3 * ch // 4, cw // 4 : 3 * cw // 4].reshape(-1)
    vals = vals[np.isfinite(vals) & (vals > 50.0) & (vals < 8000.0)]
    if vals.size < 8:
        return None
    return float(np.median(vals))


def format_distance_mm(depth_mm: Optional[float]) -> str:
    """Human label: centimetres if < 1000 mm, else metres."""
    if depth_mm is None or depth_mm <= 0:
        return "?"
    if depth_mm < 1000.0:
        return f"{depth_mm / 10.0:.0f}cm"
    return f"{depth_mm / 1000.0:.2f}m"


def black_equipment_mask(
    bgr: np.ndarray,
    *,
    v_max: int = DEFAULT_BLACK_V_MAX,
    s_max: int = DEFAULT_BLACK_S_MAX,
) -> np.ndarray:
    """HSV mask for matte black / near-black lab equipment (not drawn on screen)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([0, 0, 0], np.uint8),
        np.array([180, int(s_max), int(v_max)], np.uint8),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def silver_equipment_mask(
    bgr: np.ndarray,
    *,
    hsv_lo: tuple[int, int, int] = DEFAULT_SILVER_HSV_LO,
    hsv_hi: tuple[int, int, int] = DEFAULT_SILVER_HSV_HI,
) -> np.ndarray:
    """Cool light-grey / aluminum lab frames (not drawn). Seeded from user crop."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def equipment_obstacle_mask(
    bgr: np.ndarray,
    *,
    black_v_max: int = DEFAULT_BLACK_V_MAX,
) -> np.ndarray:
    """Union of black + silver/aluminum equipment colors (hidden; for proximity)."""
    mask = cv2.bitwise_or(
        black_equipment_mask(bgr, v_max=black_v_max),
        silver_equipment_mask(bgr),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def _robot_body_mask(bgr: np.ndarray) -> np.ndarray:
    """Rough UR grey / dark metal so we don't treat the arm as an obstacle."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Desaturated grey / charcoal arm links (+ slightly wider S for brushed metal)
    grey = cv2.inRange(
        hsv, np.array([0, 0, 30], np.uint8), np.array([180, 70, 220], np.uint8)
    )
    grey = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    grey = cv2.dilate(grey, np.ones((9, 9), np.uint8), iterations=1)
    return grey


def is_base_blue_box(
    x: int,
    y: int,
    w: int,
    h: int,
    frame_h: int,
    *,
    y_frac: float = 0.62,
) -> bool:
    """True for the pedestal / base joint (lower part of the frame).

    That joint sits on the table forever — never use it for proximity alarms.
    """
    if frame_h <= 0:
        return False
    cy = y + 0.5 * h
    return cy >= float(y_frac) * float(frame_h)


def _unproject_mm(
    u: float,
    v: float,
    z_mm: float,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float]:
    """Pixel + depth → camera-frame XYZ (mm) via similar triangles / pinhole."""
    z = float(z_mm)
    return ((u - cx) * z / fx, (v - cy) * z / fy, z)


def _dist3d_mm(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    return float(
        np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
    )


def _blue_point_mm(
    depth_mm: np.ndarray,
    blue_mask: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    d_blue: Optional[float],
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Optional[tuple[tuple[float, float, float], float]]:
    if d_blue is None:
        d_blue = median_depth_mm(depth_mm, x, y, w, h, mask=blue_mask)
    if d_blue is None:
        return None
    u = x + 0.5 * w
    v = y + 0.5 * h
    return _unproject_mm(u, v, float(d_blue), fx=fx, fy=fy, cx=cx, cy=cy), float(
        d_blue
    )


def proximity_alerts(
    depth_mm: Optional[np.ndarray],
    boxes: list[tuple[int, int, int, int, Optional[float]]],
    blue_mask: np.ndarray,
    *,
    bgr: Optional[np.ndarray] = None,
    thresh_mm: float = DEFAULT_PROXIMITY_MM,
    fx_px: float = 520.0,
    skip_base: bool = True,
    black_v_max: int = DEFAULT_BLACK_V_MAX,
    search_px: int = 180,
) -> list[tuple[int, int, int, int, float, str]]:
    """Alert when a non-base blue joint is within ``thresh_mm`` of equipment.

    Hidden obstacle mask = black + cool silver/aluminum. Distance uses each
    pixel's own depth (similar triangles / pinhole). Arm body is carved out.

    Returns (x, y, w, h, gap_mm, label).
    """
    if depth_mm is None or depth_mm.size == 0 or not boxes or bgr is None:
        return []
    dh, dw = depth_mm.shape[:2]
    if bgr.shape[:2] != (dh, dw):
        return []

    fx = max(50.0, float(fx_px))
    fy = fx
    cx_f, cy_f = 0.5 * dw, 0.5 * dh

    equip = equipment_obstacle_mask(bgr, black_v_max=black_v_max)
    body = _robot_body_mask(bgr)
    # Carve out the arm (grey + blue) so cables / joint plastic don't alarm
    arm = cv2.bitwise_or(body, blue_mask)
    arm = cv2.dilate(arm, np.ones((21, 21), np.uint8), iterations=1)
    obstacle = cv2.bitwise_and(equip, cv2.bitwise_not(arm))
    if cv2.countNonZero(obstacle) < 40:
        return []

    alerts: list[tuple[int, int, int, int, float, str]] = []
    rad = max(40, int(search_px))
    for x, y, w, h, d_blue in boxes:
        if skip_base and is_base_blue_box(x, y, w, h, dh):
            continue
        bp = _blue_point_mm(
            depth_mm, blue_mask, x, y, w, h, d_blue, fx=fx, fy=fy, cx=cx_f, cy=cy_f
        )
        if bp is None:
            continue
        p_blue, _d_blue_f = bp
        x0 = max(0, x - rad)
        y0 = max(0, y - rad)
        x1 = min(dw, x + w + rad)
        y1 = min(dh, y + h + rad)
        d_roi = depth_mm[y0:y1, x0:x1].astype(np.float32)
        o_roi = obstacle[y0:y1, x0:x1]
        ys, xs = np.where(
            (o_roi > 0)
            & np.isfinite(d_roi)
            & (d_roi > 50.0)
            & (d_roi < 8000.0)
        )
        if xs.size < 12:
            continue
        step = max(1, xs.size // 250)
        xs_s = xs[::step]
        ys_s = ys[::step]
        best: Optional[float] = None
        for i in range(xs_s.size):
            u = float(xs_s[i] + x0)
            v = float(ys_s[i] + y0)
            z = float(d_roi[ys_s[i], xs_s[i]])
            p_obs = _unproject_mm(u, v, z, fx=fx, fy=fy, cx=cx_f, cy=cy_f)
            gap = _dist3d_mm(p_blue, p_obs)
            if best is None or gap < best:
                best = gap
        if best is not None and best <= float(thresh_mm):
            alerts.append((int(x), int(y), int(w), int(h), float(best), "equip"))
    return alerts


def draw_proximity_error(
    bgr: np.ndarray,
    alerts: list,
    *,
    thresh_cm: float = 10.0,
) -> np.ndarray:
    """Paint a big red error banner + highlight offending blue boxes."""
    if not alerts:
        return bgr
    out = bgr.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), ALERT_COLOR, 14)
    cv2.rectangle(out, (10, 10), (w - 11, h - 11), ALERT_COLOR, 4)
    banner_h = max(72, h // 7)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), ALERT_COLOR, -1)
    out = cv2.addWeighted(overlay, 0.72, out, 0.28, 0)
    gap_cm = min(float(a[4]) for a in alerts) / 10.0
    lines = [
        "ERROR: PROXIMITY",
        f"BLUE < {thresh_cm:.0f}cm OF EQUIPMENT",
        f"closest gap ~ {gap_cm:.1f}cm",
    ]
    y = 28
    for i, text in enumerate(lines):
        scale = 1.15 if i == 0 else 0.75
        cv2.putText(
            out,
            text,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            3 if i == 0 else 2,
            cv2.LINE_AA,
        )
        y += 28 if i == 0 else 24
    for item in alerts:
        x, y0, bw, bh, gap = item[0], item[1], item[2], item[3], item[4]
        cv2.rectangle(out, (x, y0), (x + bw, y0 + bh), ALERT_COLOR, 3)
        cv2.putText(
            out,
            f"NEAR EQUIP {gap / 10.0:.1f}cm",
            (x, max(banner_h + 20, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            ALERT_COLOR,
            2,
            cv2.LINE_AA,
        )
    return out


def find_baby_blue_boxes(
    bgr: np.ndarray,
    *,
    min_area: int = DEFAULT_MIN_AREA,
    max_boxes: int = DEFAULT_MAX_BOXES,
    hsv_lo: tuple[int, int, int] = DEFAULT_HSV_LO,
    hsv_hi: tuple[int, int, int] = DEFAULT_HSV_HI,
    min_mean_s: float = DEFAULT_MIN_MEAN_S,
    min_fill: float = DEFAULT_MIN_FILL,
    depth_mm: Optional[np.ndarray] = None,
) -> list[tuple[int, int, int, int, Optional[float]]]:
    """Return list of (x, y, w, h, depth_mm_or_None) for strict baby-blue blobs."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = baby_blue_mask(bgr, hsv_lo=hsv_lo, hsv_hi=hsv_hi)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int, Optional[float], float]] = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < float(min_area):
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 10 or h < 10:
            continue
        # Reject long thin pale strips (cabinet edges)
        aspect = max(w, h) / max(1.0, float(min(w, h)))
        if aspect > 4.5:
            continue
        roi_m = mask[y : y + h, x : x + w]
        fill = float(cv2.countNonZero(roi_m)) / max(1.0, float(w * h))
        if fill < float(min_fill):
            continue
        roi_s = hsv[y : y + h, x : x + w, 1]
        mean_s = float(cv2.mean(roi_s, mask=roi_m)[0]) if roi_m is not None else 0.0
        if mean_s < float(min_mean_s):
            continue
        quality = min(0.99, 0.45 * fill + 0.55 * (mean_s / 120.0))
        if quality < 0.35:
            continue
        d_mm = median_depth_mm(depth_mm, x, y, w, h, mask=mask)
        boxes.append((int(x), int(y), int(w), int(h), d_mm, quality))
    boxes.sort(key=lambda b: (b[5], -(b[2] * b[3])), reverse=True)
    return [(x, y, w, h, d) for x, y, w, h, d, _q in boxes[: max(1, int(max_boxes))]]


def draw_baby_blue_overlay(
    bgr: np.ndarray,
    *,
    min_area: int = DEFAULT_MIN_AREA,
    max_boxes: int = DEFAULT_MAX_BOXES,
    show_mask: bool = True,
    hsv_lo: tuple[int, int, int] = DEFAULT_HSV_LO,
    hsv_hi: tuple[int, int, int] = DEFAULT_HSV_HI,
    boxes: Optional[list[tuple[int, int, int, int, Optional[float]]]] = None,
    depth_mm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Draw boxes labeled with camera distance (not confidence)."""
    out = bgr.copy()
    mask = baby_blue_mask(bgr, hsv_lo=hsv_lo, hsv_hi=hsv_hi)
    if boxes is None:
        boxes = find_baby_blue_boxes(
            bgr,
            min_area=min_area,
            max_boxes=max_boxes,
            hsv_lo=hsv_lo,
            hsv_hi=hsv_hi,
            depth_mm=depth_mm,
        )
    # Only tint pixels inside accepted boxes (not the whole loose mask)
    if show_mask and boxes:
        keep = np.zeros_like(mask)
        for x, y, w, h, _d in boxes:
            keep[y : y + h, x : x + w] = cv2.bitwise_and(
                mask[y : y + h, x : x + w], 255
            )
        tint = np.zeros_like(out)
        tint[:] = MASK_TINT
        overlay = cv2.addWeighted(out, 1.0, tint, 0.40, 0)
        out = np.where(keep[..., None] > 0, overlay, out)
    for i, (x, y, w, h, d_mm) in enumerate(boxes):
        # Refresh depth if caller passed a fresh depth map
        if depth_mm is not None:
            d_mm = median_depth_mm(depth_mm, x, y, w, h, mask=mask)
        cv2.rectangle(out, (x, y), (x + w, y + h), BOX_COLOR, 2)
        cv2.putText(
            out,
            f"blue {i} {format_distance_mm(d_mm)}",
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            BOX_COLOR,
            2,
            cv2.LINE_AA,
        )
    return out
