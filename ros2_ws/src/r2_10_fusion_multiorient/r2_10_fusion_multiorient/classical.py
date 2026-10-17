"""Orientation-aware classical proposals: upright cap-grow + sideways cap-axis.

Self-contained (no import of other r2_* packages). Optionally loads
Camera/sideways_race Method-I if present for extra recall.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .types import Det, classify_orientation, xywh_to_det

_MIN_DEPTH_MM = 80.0
_MAX_DEPTH_MM = 5000.0


def _align_depth(depth_mm: Optional[np.ndarray], w: int, h: int) -> Optional[np.ndarray]:
    if depth_mm is None:
        return None
    d = np.asarray(depth_mm, dtype=np.float32)
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
    return d


def _white_mask(bgr: np.ndarray, vmin: int = 185, smax: int = 60) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 2] >= vmin) & (hsv[:, :, 1] <= smax)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m


def _iou_xywh(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Upright: white cap on top → grow clear body downward
# ---------------------------------------------------------------------------


def propose_upright(
    bgr: np.ndarray,
    depth_mm: Optional[np.ndarray] = None,
) -> List[Det]:
    h, w = bgr.shape[:2]
    depth = _align_depth(depth_mm, w, h)
    white = _white_mask(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 90)
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets: List[Det] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 80 or area > 0.12 * h * w:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 10 or bh < 8:
            continue
        # Cap should be relatively squat (not a long horizontal strip alone).
        if bw / max(bh, 1) > 3.5 or bh / max(bw, 1) > 2.2:
            continue
        # Grow body below cap.
        reach = int(max(bw, bh) * 2.6)
        y0 = y + bh
        y1 = min(h, y0 + reach)
        if y1 - y0 < max(12, int(bh * 0.8)):
            continue
        band = gray[y0:y1, max(0, x - 2) : min(w, x + bw + 2)]
        if band.size == 0:
            continue
        # Stop when we leave mid-tone / edge body.
        body_h = 0
        gap = 0
        for row in range(band.shape[0]):
            row_vals = band[row]
            mean_v = float(row_vals.mean())
            edge_row = edges[y0 + row, max(0, x) : min(w, x + bw)]
            edge_frac = float((edge_row > 0).mean()) if edge_row.size else 0.0
            invalid = 0.0
            if depth is not None:
                drow = depth[y0 + row, max(0, x) : min(w, x + bw)]
                invalid = float((drow <= _MIN_DEPTH_MM).mean()) if drow.size else 0.0
            ok = (40 <= mean_v <= 175) or edge_frac >= 0.04 or invalid >= 0.2
            if ok:
                body_h = row + 1
                gap = 0
            else:
                gap += 1
                if body_h > 0 and gap >= 3:
                    break
        if body_h < max(14, int(bw * 0.55)):
            continue
        total_h = bh + body_h
        # Inset x slightly from lid width for cup.
        pad = max(0, int(bw * 0.05))
        bx = max(0, x - pad)
        bw2 = min(w - bx, bw + 2 * pad)
        hw = total_h / max(bw2, 1)
        if not (0.95 <= hw <= 2.4):
            continue
        conf = min(0.85, 0.35 + 0.01 * body_h + 0.15 * min(1.0, area / 800.0))
        det = xywh_to_det(bx, y, bw2, total_h, conf=conf, source="classical_upright")
        det.orientation = "upright"
        dets.append(det)
    dets.sort(key=lambda d: d.conf, reverse=True)
    return _nms_dets(dets, 0.4)[:5]


# ---------------------------------------------------------------------------
# Sideways: compact white end-cap → oriented body hypothesis
# ---------------------------------------------------------------------------


def _polygon_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(points).astype(np.int32), 255)
    return mask


def _oriented_points(
    center: np.ndarray, length: float, diameter: float, axis: np.ndarray
) -> np.ndarray:
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    return np.stack(
        (
            center - 0.5 * length * axis - 0.5 * diameter * normal,
            center + 0.5 * length * axis - 0.5 * diameter * normal,
            center + 0.5 * length * axis + 0.5 * diameter * normal,
            center - 0.5 * length * axis + 0.5 * diameter * normal,
        )
    )


def _median_at(image: np.ndarray, mask: np.ndarray) -> float:
    values = image[mask > 0]
    return float(np.median(values)) if values.size else 0.0


def _body_hypothesis(
    gray: np.ndarray,
    edges: np.ndarray,
    depth: Optional[np.ndarray],
    cap_center: np.ndarray,
    axis: np.ndarray,
    cap_length: float,
    diameter: float,
) -> tuple[float, Optional[tuple[int, int, int, int]]]:
    h, w = gray.shape
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    best_score = -1.0
    best_box = None
    for body_factor in np.linspace(0.95, 1.90, 8):
        body_length = float(diameter * body_factor)
        body_center = cap_center + axis * (0.5 * cap_length + 0.5 * body_length)
        body_pts = _oriented_points(body_center, body_length, diameter * 0.92, axis)
        body_mask = _polygon_mask((h, w), body_pts)
        if cv2.countNonZero(body_mask) < 100:
            continue
        outer_a = _polygon_mask(
            (h, w),
            _oriented_points(
                body_center + normal * diameter * 0.78, body_length, diameter * 0.28, axis
            ),
        )
        outer_b = _polygon_mask(
            (h, w),
            _oriented_points(
                body_center - normal * diameter * 0.78, body_length, diameter * 0.28, axis
            ),
        )
        outer = cv2.bitwise_or(outer_a, outer_b)
        contrast = _median_at(gray, outer) - _median_at(gray, body_mask)
        rail = np.zeros((h, w), np.uint8)
        p = np.rint(body_pts).astype(np.int32)
        cv2.line(rail, tuple(p[0]), tuple(p[1]), 255, 4, cv2.LINE_AA)
        cv2.line(rail, tuple(p[3]), tuple(p[2]), 255, 4, cv2.LINE_AA)
        rail_pixels = max(cv2.countNonZero(rail), 1)
        edge_fraction = cv2.countNonZero(cv2.bitwise_and(edges, rail)) / rail_pixels
        invalid = 0.0
        near_flat = 0.0
        if depth is not None:
            d = depth[body_mask > 0]
            if d.size:
                invalid = float((d <= _MIN_DEPTH_MM).mean())
                valid = d[(d > _MIN_DEPTH_MM) & (d < _MAX_DEPTH_MM)]
                if valid.size > 20:
                    near_flat = float(np.std(valid) < 14.0)
        rgb_ok = edge_fraction >= 0.07 and contrast >= 3.5
        depth_ok = invalid >= 0.20 and edge_fraction >= 0.03
        if depth is not None and invalid < 0.10:
            continue
        if not (rgb_ok or depth_ok) or (near_flat and invalid < 0.08):
            continue
        score = 2.8 * edge_fraction + 0.025 * max(contrast, 0.0) + 0.75 * min(invalid, 0.8)
        far_center = cap_center + axis * (cap_length * 0.5 + body_length)
        far = _polygon_mask((h, w), _oriented_points(far_center, 5.0, diameter, axis))
        score += 0.5 * cv2.countNonZero(cv2.bitwise_and(edges, far)) / max(
            cv2.countNonZero(far), 1
        )
        total_length = cap_length + body_length
        vial_center = cap_center + axis * (0.5 * body_length)
        vial_pts = _oriented_points(vial_center, total_length, diameter, axis)
        x, y, bw, bh = cv2.boundingRect(np.rint(vial_pts).astype(np.int32))
        x, y = max(0, x), max(0, y)
        bw, bh = min(w - x, bw), min(h - y, bh)
        if score > best_score and bw >= 18 and bh >= 10:
            best_score = score
            best_box = (x, y, bw, bh)
    return best_score, best_box


def propose_sideways(
    bgr: np.ndarray,
    depth_mm: Optional[np.ndarray] = None,
) -> List[Det]:
    """RGB(+depth) sideways proposals; also tries race Method-I if available."""
    h, w = bgr.shape[:2]
    depth = _align_depth(depth_mm, w, h)

    # Race winner Method-I: 0.95 recall / 0 FP on synth. Prefer exclusively when
    # available so weaker local heuristics cannot inject empty-bench FPs.
    race_boxes = _try_race_method_i(bgr, depth)
    if race_boxes is not None:
        dets: List[Det] = []
        kept: List[tuple[int, int, int, int]] = []
        for box in race_boxes:
            if any(_iou_xywh(box, other) >= 0.35 for other in kept):
                continue
            kept.append(box)
            det = xywh_to_det(*box, conf=0.88, source="classical_sideways")
            det.orientation = "sideways"
            dets.append(det)
            # Single-vial scenes: keep the strongest Method-I box to protect precision.
            if len(dets) >= 1:
                break
        return dets

    boxes: List[tuple[float, tuple[int, int, int, int]]] = []

    # Path A: angled end-cap → body (Method-I style, self-contained).
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bright = ((hsv[:, :, 2] >= 140) & (hsv[:, :, 1] <= 75)).astype(np.uint8) * 255
    bright = cv2.morphologyEx(
        bright, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    edges = cv2.Canny(gray, 22, 70)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 70 or area > 0.025 * h * w:
            continue
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
        if min(rw, rh) < 5 or max(rw, rh) < 14 or max(rw, rh) > 60:
            continue
        solidity = area / max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
        if solidity < 0.60:
            continue
        if rw <= rh:
            theta = np.deg2rad(angle)
            cap_length, diameter = float(rw), float(rh)
        else:
            theta = np.deg2rad(angle + 90.0)
            cap_length, diameter = float(rh), float(rw)
        if not (0.16 <= cap_length / max(diameter, 1.0) <= 0.85):
            continue
        axis0 = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        center = np.array([cx, cy], dtype=np.float32)
        for sign in (-1.0, 1.0):
            axis = axis0 * sign
            # Prefer near-horizontal axes (sideways / ±30°).
            if abs(float(axis[0])) < 0.55:
                continue
            score, box = _body_hypothesis(
                gray, edges, depth, center, axis, cap_length, diameter
            )
            if box is not None and score >= 0.40:
                boxes.append((score + 0.25 * solidity, box))

    # Path B: horizontal white-cap grow left/right (simple, high-precision).
    white = _white_mask(bgr, vmin=175, smax=65)
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 60 or area > 0.15 * h * w:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 6 or bh < 10:
            continue
        # End-cap: taller than wide-ish, or square-ish.
        if bh < bw * 0.7:
            continue
        for side, step in (("left", -1), ("right", 1)):
            max_reach = int(max(bw, bh) * 2.5)
            extent = 0
            gap = 0
            hole_hits = 0
            start = x - 1 if side == "left" else x + bw
            inset = max(1, bh // 6)
            y0, y1 = y + inset, y + bh - inset
            if y1 <= y0:
                y0, y1 = y, y + bh
            for i in range(max_reach):
                col = start + i * step
                if col < 0 or col >= w:
                    break
                band_v = hsv[y0:y1, col, 2]
                band_s = hsv[y0:y1, col, 1]
                if band_v.size == 0:
                    break
                white_frac = float(((band_v >= 190) & (band_s <= 55)).mean())
                if white_frac >= 0.7 and extent >= 8:
                    break
                tone = float(((band_v >= 35) & (band_v <= 165) & (band_s <= 95)).mean())
                invalid = 0.0
                if depth is not None:
                    dcol = depth[y0:y1, col]
                    invalid = float((dcol <= _MIN_DEPTH_MM).mean())
                edge = float(edges[y0:y1, col].mean() / 255.0) if y1 > y0 else 0.0
                ok = invalid >= 0.22 or edge >= 0.08 or (tone >= 0.5 and extent > 0)
                if ok:
                    extent += 1
                    gap = 0
                    if invalid >= 0.22:
                        hole_hits += 1
                else:
                    gap += 1
                    if extent > 0 and gap >= 2:
                        break
            min_cup = max(10, int(max(bw, bh) * 0.4))
            if extent < min_cup:
                continue
            if hole_hits < 2 and depth is not None:
                # RGB-only allow if extent strong.
                if extent < int(min_cup * 1.3):
                    continue
            if side == "left":
                bx = max(0, start - extent + 1)
                total_w = (x + bw) - bx
            else:
                bx = x
                total_w = extent + bw
            by, bh2 = y, bh
            hw = bh2 / max(total_w, 1)
            if not (0.32 <= hw <= 0.90):
                continue
            score = 0.45 + 0.01 * extent + 0.02 * hole_hits
            boxes.append((score, (bx, by, total_w, bh2)))

    boxes.sort(key=lambda t: t[0], reverse=True)
    kept: List[tuple[int, int, int, int]] = []
    dets: List[Det] = []
    for score, box in boxes:
        if any(_iou_xywh(box, other) >= 0.35 for other in kept):
            continue
        kept.append(box)
        det = xywh_to_det(*box, conf=min(0.9, float(score)), source="classical_sideways")
        det.orientation = "sideways"
        dets.append(det)
        if len(dets) >= 5:
            break
    return dets


_METHOD_I = None  # None=unresolved, False=unavailable, callable=ready


def _load_method_i():
    global _METHOD_I
    if _METHOD_I is not None:
        return _METHOD_I
    try:
        cam = None
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "sideways_race" / "methods" / "method_i_hard_robust.py").is_file():
                cam = parent
                break
        if cam is None:
            _METHOD_I = False
            return _METHOD_I
        if str(cam) not in sys.path:
            sys.path.insert(0, str(cam))
        from sideways_race.methods.method_i_hard_robust import detect_sideways

        _METHOD_I = detect_sideways
        return _METHOD_I
    except Exception:  # noqa: BLE001
        try:
            here = Path(__file__).resolve()
            methods_dir = None
            for parent in here.parents:
                p = parent / "sideways_race" / "methods" / "method_i_hard_robust.py"
                if p.is_file():
                    methods_dir = p.parent
                    break
            if methods_dir is None:
                _METHOD_I = False
                return _METHOD_I
            if str(methods_dir) not in sys.path:
                sys.path.insert(0, str(methods_dir))
            path = methods_dir / "method_i_hard_robust.py"
            spec = importlib.util.spec_from_file_location("r2_method_i", path)
            if spec is None or spec.loader is None:
                _METHOD_I = False
                return _METHOD_I
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _METHOD_I = mod.detect_sideways
            return _METHOD_I
        except Exception:  # noqa: BLE001
            _METHOD_I = False
            return _METHOD_I


def _try_race_method_i(
    bgr: np.ndarray, depth: Optional[np.ndarray]
) -> Optional[List[tuple[int, int, int, int]]]:
    """Return Method-I boxes, or None if the race module is unavailable.

    Empty list means Method-I ran and found nothing (trust that — do not fall
    back to noisier local heuristics on the same frame).
    """
    fn = _load_method_i()
    if fn is False or fn is None:
        return None
    try:
        return list(fn(bgr, depth))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Cap-axis refine: snap YOLO boxes toward white-cap + body extent
# ---------------------------------------------------------------------------


def refine_cap_axis(
    bgr: np.ndarray,
    det: Det,
    depth_mm: Optional[np.ndarray] = None,
) -> Det:
    """Light refine: expand/shrink box using white-cap locus + orientation."""
    h, w = bgr.shape[:2]
    x, y, bw, bh = det.xywh
    # Pad ROI
    pad = int(0.15 * max(bw, bh))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return det
    white = _white_mask(roi, vmin=180, smax=65)
    if white.mean() < 1.0:
        return det
    ys, xs = np.where(white > 0)
    if xs.size < 20:
        return det
    # Cap centroid in full image
    cx = float(xs.mean()) + x0
    cy = float(ys.mean()) + y0
    orient = det.orientation if det.orientation != "unknown" else classify_orientation(det)
    out = Det(
        det.x1,
        det.y1,
        det.x2,
        det.y2,
        det.conf,
        det.cls_id,
        det.label,
        det.source + "+refine",
        orient,
    )
    if orient == "sideways":
        # Ensure box covers from leftmost to rightmost white + body stretch.
        wx0, wx1 = float(xs.min()) + x0, float(xs.max()) + x0
        # Expand horizontally toward image center of mass away from cap.
        mid = 0.5 * (det.x1 + det.x2)
        if cx < mid:
            # Cap on left → body to the right; keep left at cap, extend right.
            out.x1 = min(out.x1, wx0 - 2)
            out.x2 = max(out.x2, mid + (mid - cx))
        else:
            out.x2 = max(out.x2, wx1 + 2)
            out.x1 = min(out.x1, mid - (cx - mid))
        out.y1 = min(out.y1, float(ys.min()) + y0 - 2)
        out.y2 = max(out.y2, float(ys.max()) + y0 + 2)
    else:
        # Upright: cap near top of box.
        out.y1 = min(out.y1, float(ys.min()) + y0 - 2)
        out.x1 = min(out.x1, float(xs.min()) + x0 - 2)
        out.x2 = max(out.x2, float(xs.max()) + x0 + 2)
    out.clamp(w, h)
    out.orientation = classify_orientation(out)
    return out


def _nms_dets(dets: Sequence[Det], iou_thr: float) -> List[Det]:
    ordered = sorted(dets, key=lambda d: d.conf, reverse=True)
    kept: List[Det] = []
    from .types import box_iou

    for d in ordered:
        if any(box_iou(d, k) >= iou_thr for k in kept):
            continue
        kept.append(d)
    return kept
