"""Bridge sideways_race winner (I_hard_robust) → oriented boxes.

Round-1 AABB YOLO misses side views. The sideways race winner already finds
lid+cup with high recall; we keep its geometry but emit OBB (rotated rect)
instead of only an axis-aligned cover.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .obb_geometry import OBBDet, aabb_xywh_to_obb, corners_to_obb


def _camera_root() -> Path:
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / "sideways_race" / "methods").is_dir():
            return parent
    for parent in here.parents:
        if parent.name == "ros2_ws":
            sibling = parent.parent / "robotics"
            if sibling.is_dir():
                return sibling
            break
    return here.parents[4]


def _load_method_i():
    path = _camera_root() / "sideways_race" / "methods" / "method_i_hard_robust.py"
    if not path.is_file():
        return None
    cam = str(_camera_root())
    methods = str(path.parent)
    for p in (cam, methods):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("method_i_hard_robust_obb", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "detect_sideways", None)


_DETECT_FN = None


def _refine_crop_obb(bgr: np.ndarray, x: float, y: float, w: float, h: float, conf: float) -> OBBDet:
    H, W = bgr.shape[:2]
    xa, ya = max(0, int(x)), max(0, int(y))
    xb, yb = min(W, int(x + w)), min(H, int(y + h))
    if xb - xa < 6 or yb - ya < 6:
        return aabb_xywh_to_obb(x, y, w, h, angle_deg=0.0 if w >= h else None, conf=conf, source="sideways_race_i")
    crop = bgr[ya:yb, xa:xb]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 140)
    ys, xs = np.where(edges > 0)
    if xs.size >= 12:
        pts = np.stack([xs + xa, ys + ya], axis=1).astype(np.float32)
        det = corners_to_obb(pts, conf=conf, source="sideways_race_i")
        # If refine flipped to upright on a clearly wide AABB, keep horizontal OBB.
        if w >= h * 1.15 and det.orientation == "upright":
            return aabb_xywh_to_obb(x, y, w, h, angle_deg=0.0, conf=conf, source="sideways_race_i")
        return det
    return aabb_xywh_to_obb(x, y, w, h, angle_deg=0.0 if w >= h else None, conf=conf, source="sideways_race_i")


def detect_sideways_obb(
    color_bgr: np.ndarray,
    depth_mm: Optional[np.ndarray] = None,
    intrinsics=None,
) -> List[OBBDet]:
    """Return oriented vials for sideways / fallen poses."""
    global _DETECT_FN
    if color_bgr is None or getattr(color_bgr, "size", 0) == 0:
        return []
    if _DETECT_FN is None:
        _DETECT_FN = _load_method_i()
    if _DETECT_FN is None:
        return _fallback_sideways_obb(color_bgr)

    try:
        aabbs = _DETECT_FN(color_bgr, depth_mm, intrinsics)
    except Exception:  # noqa: BLE001
        return _fallback_sideways_obb(color_bgr)

    out: List[OBBDet] = []
    for box in aabbs or []:
        x, y, w, h = [float(v) for v in box]
        # Wide AABB → horizontal OBB (matches synth GT). Near-square → minAreaRect θ.
        if w >= h * 1.25:
            out.append(
                aabb_xywh_to_obb(x, y, w, h, angle_deg=0.0, conf=0.72, source="sideways_race_i")
            )
        else:
            out.append(_refine_crop_obb(color_bgr, x, y, w, h, conf=0.72))
    return out


def _fallback_sideways_obb(color_bgr: np.ndarray) -> List[OBBDet]:
    h, w = color_bgr.shape[:2]
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] >= 185) & (hsv[:, :, 1] <= 55)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[OBBDet] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 40 or area > 0.15 * h * w:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        grow = int(max(bw, bh) * 1.8)
        x0 = max(0, x - grow)
        x1 = min(w, x + bw + grow)
        y0 = max(0, y - int(0.15 * bh))
        y1 = min(h, y + bh + int(0.15 * bh))
        ww, hh = x1 - x0, y1 - y0
        if ww < hh * 1.1:
            continue
        out.append(aabb_xywh_to_obb(x0, y0, ww, hh, angle_deg=0.0, conf=0.4, source="sideways_fallback"))
    return out[:3]
