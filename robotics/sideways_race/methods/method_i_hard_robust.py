"""Method I: conservative multi-angle cap/body detector for degraded stills.

B2 remains the primary path.  The fallback uses a bright compact end cap to
anchor an oriented body hypothesis, then requires parallel body-edge evidence
and either RGB contrast or clear-plastic depth holes.  Depth is supporting
evidence, not mandatory, so missing-depth hard cases remain detectable.
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    from sideways_race.methods.method_b2_improved import (
        _align_depth,
        _iou,
        detect_sideways as _detect_b2,
    )
except ModuleNotFoundError:
    from method_b2_improved import _align_depth, _iou, detect_sideways as _detect_b2

NAME = "I_hard_robust"

_MIN_DEPTH_MM = 80.0
_MAX_DEPTH_MM = 5000.0


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
    depth: np.ndarray | None,
    cap_center: np.ndarray,
    axis: np.ndarray,
    cap_length: float,
    diameter: float,
) -> tuple[float, tuple[int, int, int, int] | None]:
    """Return best evidence score and oriented vial AABB for one cap side."""
    h, w = gray.shape
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    best_score = -1.0
    best_box = None

    for body_factor in np.linspace(0.95, 1.85, 7):
        body_length = float(diameter * body_factor)
        body_center = cap_center + axis * (0.5 * cap_length + 0.5 * body_length)
        body_pts = _oriented_points(body_center, body_length, diameter * 0.92, axis)
        body_mask = _polygon_mask((h, w), body_pts)
        if cv2.countNonZero(body_mask) < 120:
            continue

        # Compare the body with strips immediately outside its two long edges.
        outer_a = _polygon_mask(
            (h, w),
            _oriented_points(body_center + normal * diameter * 0.78, body_length, diameter * 0.28, axis),
        )
        outer_b = _polygon_mask(
            (h, w),
            _oriented_points(body_center - normal * diameter * 0.78, body_length, diameter * 0.28, axis),
        )
        outer = cv2.bitwise_or(outer_a, outer_b)
        contrast = _median_at(gray, outer) - _median_at(gray, body_mask)

        # Parallel long edges are the strongest RGB-only signal.
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

        # Reject untextured solid slabs. Missing depth (invalid ~= 1) is okay
        # only with convincing parallel edges and local contrast.
        rgb_ok = edge_fraction >= 0.075 and contrast >= 4.0
        depth_ok = invalid >= 0.22 and edge_fraction >= 0.035
        # With a depth frame present, a transparent/missing-depth body must
        # contain holes. This rejects detached white blobs beside dark tools.
        if depth is not None and invalid < 0.12:
            continue
        if not (rgb_ok or depth_ok) or (near_flat and invalid < 0.08):
            continue

        score = 2.8 * edge_fraction + 0.025 * max(contrast, 0.0) + 0.75 * min(invalid, 0.8)
        # Prefer an extent whose far end still has an edge, preventing a tiny
        # cap-adjacent fragment or excessive growth into the desk.
        far_center = cap_center + axis * (cap_length * 0.5 + body_length)
        far = _polygon_mask((h, w), _oriented_points(far_center, 5.0, diameter, axis))
        score += 0.5 * cv2.countNonZero(cv2.bitwise_and(edges, far)) / max(cv2.countNonZero(far), 1)

        total_length = cap_length + body_length
        vial_center = cap_center + axis * (0.5 * body_length)
        vial_pts = _oriented_points(vial_center, total_length, diameter, axis)
        x, y, bw, bh = cv2.boundingRect(np.rint(vial_pts).astype(np.int32))
        x, y = max(0, x), max(0, y)
        bw, bh = min(w - x, bw), min(h - y, bh)
        if score > best_score and bw >= 20 and bh >= 12:
            best_score = score
            best_box = (x, y, bw, bh)
    return best_score, best_box


def _angled_fallback(
    color_bgr: np.ndarray, depth: np.ndarray | None
) -> list[tuple[int, int, int, int]]:
    h, w = color_bgr.shape[:2]
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    # The low threshold includes dim lids; local geometry filters the clutter.
    bright = ((hsv[:, :, 2] >= 140) & (hsv[:, :, 1] <= 75)).astype(np.uint8) * 255
    bright = cv2.morphologyEx(
        bright, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    edges = cv2.Canny(gray, 22, 70)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 70 or area > 0.025 * h * w:
            continue
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
        if min(rw, rh) < 5 or max(rw, rh) < 14 or max(rw, rh) > 55:
            continue
        solidity = area / max(float(cv2.contourArea(cv2.convexHull(contour))), 1.0)
        if solidity < 0.62:
            continue

        # The cap's diameter is its long side; the vial axis follows its short side.
        if rw <= rh:
            theta = np.deg2rad(angle)
            cap_length, diameter = float(rw), float(rh)
        else:
            theta = np.deg2rad(angle + 90.0)
            cap_length, diameter = float(rh), float(rw)
        if not (0.18 <= cap_length / max(diameter, 1.0) <= 0.82):
            continue
        axis0 = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        center = np.array([cx, cy], dtype=np.float32)

        for sign in (-1.0, 1.0):
            axis = axis0 * sign
            # Sideways includes the requested +/-30-degree range, not upright caps.
            if abs(float(axis[0])) < 0.72:
                continue
            score, box = _body_hypothesis(
                gray, edges, depth, center, axis, cap_length, diameter
            )
            if box is not None and score >= 0.42:
                candidates.append((score + 0.25 * solidity, box))

    candidates.sort(key=lambda item: item[0], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for _score, box in candidates:
        if not any(_iou(box, other) >= 0.35 for other in kept):
            kept.append(box)
    return kept[:4]


def detect_sideways(
    color_bgr,
    depth_mm,
    intrinsics=None,
) -> list[tuple[int, int, int, int]]:
    """Detect original and degraded sideways vials with unchanged harness API."""
    if color_bgr is None or getattr(color_bgr, "size", 0) == 0:
        return []
    h, w = color_bgr.shape[:2]
    depth = _align_depth(depth_mm, w, h)
    primary = _detect_b2(color_bgr, depth, intrinsics)
    fallback = _angled_fallback(color_bgr, depth)
    result = list(primary)
    for box in fallback:
        if not any(_iou(box, old) >= 0.35 for old in result):
            result.append(box)
    return result

