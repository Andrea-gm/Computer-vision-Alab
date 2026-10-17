"""Sideways race method A: white end-cap → grow clear cup horizontally.

Find opaque white lid blobs, then extend left/right into the translucent body
using Canny / Sobel walls, mid-tone value continuity, and depth holes through
clear plastic. Rank candidates by lid solidity + grown body length.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

NAME = "A_cap_horizontal_grow"

# Match live depthcamera sideways band (h/w) without importing the live module.
_SIDEWAYS_ASPECT_MIN = 0.40
_SIDEWAYS_ASPECT_MAX = 0.71
_FIT_SIDEWAYS_LENGTH_RATIO = 1.65
_CLEAR_FAR_DELTA_MM = 200
_MAX_DEPTH_MM = 5000
_MIN_SURFACE_MM = 80.0


def detect_sideways(
    color_bgr,
    depth_mm,
    intrinsics=None,
) -> list[tuple[int, int, int, int]]:
    """Detect sideways (fallen) white-cap vials; return AABB boxes (x, y, w, h).

    ``intrinsics`` is accepted for API compatibility (fx, fy, cx, cy) but not
    required — ranking uses image-space body length when Z scale is unavailable.
    """
    del intrinsics  # reserved for future mm-scale ranking
    if color_bgr is None or color_bgr.size == 0:
        return []
    image = np.asarray(color_bgr)
    if image.ndim != 3 or image.shape[2] < 3:
        return []
    height, width = image.shape[:2]
    depth = None
    if depth_mm is not None:
        depth = np.asarray(depth_mm)
        if depth.shape[:2] != (height, width):
            depth = cv2.resize(
                depth.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 35, 100)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    lids = _white_lid_blobs(value, saturation, width, height)
    scored: list[tuple[float, tuple[int, int, int, int]]] = []
    for lid, solidity in lids:
        grown = _grow_from_lid(image, value, saturation, edges, gy, depth, lid)
        if grown is None:
            continue
        gx, gy_, gw, gh = grown
        if not _is_sideways_box((gx, gy_, gw, gh)):
            continue
        if _end_cap_score(image, grown) < 0.10:
            continue
        body_len = max(0, gw - lid[2])
        rank = float(solidity) * 0.55 + min(1.0, body_len / max(float(gh), 1.0)) * 0.45
        scored.append((rank, grown))

    scored.sort(key=lambda item: item[0], reverse=True)
    # Deduplicate heavily overlapping proposals (keep higher rank).
    out: list[tuple[int, int, int, int]] = []
    for _, box in scored:
        if any(_iou(box, kept) >= 0.55 for kept in out):
            continue
        out.append(box)
        if len(out) >= 8:
            break
    return out


def _clip_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = (int(round(v)) for v in box)
    w = max(1, w)
    h = max(1, h)
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    return x, y, w, h


def _is_sideways_box(box) -> bool:
    _, _, w, h = box
    if w < 12 or h < 8:
        return False
    aspect = h / float(w)
    return _SIDEWAYS_ASPECT_MIN <= aspect <= _SIDEWAYS_ASPECT_MAX and w >= h * 1.15


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _white_mask(value, saturation, minimum_value: float) -> np.ndarray:
    mask = ((value >= minimum_value) & (saturation <= 55)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _cap_mask(value, saturation) -> np.ndarray:
    best = _white_mask(value, saturation, 200)
    best_fraction = float(best.mean() / 255)
    if 0.002 <= best_fraction <= 0.22:
        return best
    for minimum in (185, 170, 210, 220, 230):
        trial = _white_mask(value, saturation, minimum)
        fraction = float(trial.mean() / 255)
        if 0.002 <= fraction <= 0.22 and abs(fraction - 0.06) < abs(best_fraction - 0.06):
            best = trial
            best_fraction = fraction
    return best


def _white_lid_blobs(
    value, saturation, width: int, height: int
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Return (box, solidity) for compact white end-cap candidates."""
    cap_mask = _cap_mask(value, saturation)
    cap_mask = cv2.morphologyEx(
        cap_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)),
    )
    # Vertical close so side-on end caps stay as one blob.
    cap_mask = cv2.bitwise_or(
        cap_mask,
        cv2.morphologyEx(
            cap_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11)),
        ),
    )
    contours, _ = cv2.findContours(cap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_end = max(10, int(min(width, height) * 0.025))
    lids: list[tuple[tuple[int, int, int, int], float]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw > 0.72 * width or bh > 0.72 * height:
            continue
        if bw * bh > 0.4 * width * height:
            continue
        aspect = bw / max(bh, 1)
        # End-cap: compact; often taller than wide (Ø × lid depth). Skip wide upright lids.
        end_cap = (
            bw >= min_end
            and bh >= min_end
            and 0.22 <= aspect <= 1.55
            and max(bw, bh) <= min(width, height) * 0.40
        )
        if not end_cap:
            continue
        area = float(cv2.contourArea(contour))
        rect_area = float(max(bw * bh, 1))
        solidity = area / rect_area
        if solidity < 0.35:
            continue
        # Expand slightly into remaining white pixels.
        x, y, bw, bh = _expand_lid(value, saturation, x, y, bw, bh, width, height)
        if bw / max(bh, 1) >= 1.35:
            # Wide strip → upright lid; not for horizontal grow.
            continue
        lids.append(((x, y, bw, bh), solidity))
    lids.sort(key=lambda item: item[1] * item[0][2] * item[0][3], reverse=True)
    return lids[:24]


def _expand_lid(value, saturation, x, y, bw, bh, width, height):
    white = (value >= 170) & (saturation <= 55)
    for _ in range(8):
        grew = False
        if x > 0 and float(white[y : y + bh, x - 1].mean()) > 0.55:
            x -= 1
            bw += 1
            grew = True
        if x + bw < width and float(white[y : y + bh, x + bw].mean()) > 0.55:
            bw += 1
            grew = True
        if y > 0 and float(white[y - 1, x : x + bw].mean()) > 0.55:
            y -= 1
            bh += 1
            grew = True
        if y + bh < height and float(white[y + bh, x : x + bw].mean()) > 0.55:
            bh += 1
            grew = True
        if not grew:
            break
    return _clip_box((x, y, bw, bh), width, height)


def _blob_depth_mm(depth_mm: np.ndarray, x: int, y: int, bw: int, bh: int) -> float | None:
    height, width = depth_mm.shape[:2]
    cx = x + bw // 2
    cy = y + max(1, bh // 2)
    radius = max(3, min(bw, bh) // 3)
    x0, y0 = max(0, cx - radius), max(0, cy - radius)
    x1, y1 = min(width, cx + radius + 1), min(height, cy + radius + 1)
    patch = depth_mm[y0:y1, x0:x1]
    usable = patch[
        (patch > _MIN_SURFACE_MM) & (patch < _MAX_DEPTH_MM) & np.isfinite(patch)
    ]
    if usable.size < 4:
        return None
    return float(np.median(usable))


def _cup_cols(
    value,
    saturation,
    edges,
    y: int,
    bh: int,
    start: int,
    step: int,
    max_cols: int,
    height: int,
    width: int,
    depth_mm=None,
    cap_depth=None,
    gy=None,
) -> int:
    """Walk columns left/right while clear-cup cues continue."""
    traveled = 0
    gap = 0
    side = max(2, bh // 8)
    for index in range(max_cols):
        col = start + index * step
        if col < 0 or col >= width:
            break
        segment_v = value[y : y + bh, col]
        segment_s = saturation[y : y + bh, col]
        if segment_v.size == 0:
            break
        if depth_mm is not None and cap_depth is not None:
            strip = depth_mm[y : y + bh, col]
            near = strip[(strip > _MIN_SURFACE_MM) & (strip < _MAX_DEPTH_MM)]
            if near.size >= 4 and float(np.median(near)) > float(cap_depth) + 350:
                break
        margin = max(4, bh // 3)
        outside_white = []
        if y - margin >= 0:
            outside_v = value[y - margin : y, col]
            outside_s = saturation[y - margin : y, col]
            outside_white.append(float(((outside_v >= 170) & (outside_s <= 55)).mean()))
        if y + bh + margin <= height:
            outside_v = value[y + bh : y + bh + margin, col]
            outside_s = saturation[y + bh : y + bh + margin, col]
            outside_white.append(float(((outside_v >= 170) & (outside_s <= 55)).mean()))
        walls = False
        if edges is not None:
            top_wall = edges[y : y + side, col]
            bot_wall = edges[y + bh - side : y + bh, col]
            walls = bool(top_wall.any() or bot_wall.any())
        if not walls and gy is not None:
            top_gy = float(np.mean(np.abs(gy[y : y + side, col]))) if side > 0 else 0.0
            bot_gy = (
                float(np.mean(np.abs(gy[y + bh - side : y + bh, col]))) if side > 0 else 0.0
            )
            walls = top_gy >= 12.0 or bot_gy >= 12.0
        mean_v = float(segment_v.mean())
        bodyish = mean_v >= 55 and mean_v <= 210 and float(segment_s.mean()) <= 90
        depth_body = False
        if depth_mm is not None and cap_depth is not None:
            strip = depth_mm[y : y + bh, col]
            finite = np.isfinite(strip)
            valid = finite & (strip > _MIN_SURFACE_MM) & (strip < _MAX_DEPTH_MM)
            invalid = ~valid
            far = valid & (strip > float(cap_depth) + _CLEAR_FAR_DELTA_MM)
            hole = float(np.count_nonzero(invalid | far)) / max(float(strip.size), 1.0)
            depth_body = hole >= 0.35
        center_bright = mean_v >= 90
        if outside_white and max(outside_white) > 0.7 and not walls and not center_bright:
            gap += 1
            if gap > 3:
                break
            continue
        if mean_v < 40 and not walls and not depth_body:
            gap += 1
            if gap > 2:
                break
            continue
        if not walls and not bodyish and not depth_body and traveled > 4:
            gap += 1
            if gap > 4:
                break
            continue
        if not walls and traveled > 10 and not (bodyish or depth_body):
            gap += 1
            if gap > 3:
                break
            continue
        gap = 0
        traveled += 1
    return traveled


def _end_cap_score(image, box) -> float:
    height, width = image.shape[:2]
    x, y, w, h = _clip_box(box, width, height)
    if w < 12 or h < 8:
        return 0.0
    hsv = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 150)
    vband = max(3, int(w * 0.34))
    if w <= vband * 2:
        return 0.0
    left = float(white[:, :vband].mean())
    right = float(white[:, -vband:].mean())
    mid = float(white[:, vband : w - vband].mean())
    hband = max(3, int(h * 0.34))
    top = float(white[:hband].mean())
    bot = float(white[-hband:].mean())
    end = max(left, right)
    if end < 0.28 or mid > end - 0.10:
        return 0.0
    return float(max(0.0, min(1.0, end - max(mid, top, bot) + 0.15)))


def _grow_from_lid(
    image,
    value,
    saturation,
    edges,
    gy,
    depth_mm,
    lid_box,
) -> tuple[int, int, int, int] | None:
    height, width = image.shape[:2]
    x, y, bw, bh = _clip_box(lid_box, width, height)
    if bw < 8 or bh < 8:
        return None
    if bw / max(bh, 1) >= 1.35:
        return None
    cap_depth = None
    if depth_mm is not None:
        cap_depth = _blob_depth_mm(depth_mm, x, y, bw, bh)
    inset_y = max(1, bh // 5)
    core_y = y + inset_y
    core_h = max(1, bh - 2 * inset_y)
    max_cup_x = int(max(bh, bw) * _FIT_SIDEWAYS_LENGTH_RATIO)
    left = _cup_cols(
        value,
        saturation,
        edges,
        core_y,
        core_h,
        x - 1,
        -1,
        max_cup_x,
        height,
        width,
        depth_mm=depth_mm,
        cap_depth=cap_depth,
        gy=gy,
    )
    right = _cup_cols(
        value,
        saturation,
        edges,
        core_y,
        core_h,
        x + bw,
        1,
        max_cup_x,
        height,
        width,
        depth_mm=depth_mm,
        cap_depth=cap_depth,
        gy=gy,
    )
    pad = max(2, int(max(bw, bh) * 0.04))
    min_cup = max(8, int(max(bh, bw) * 0.22))
    # Prefer the side with clearer cup evidence (depth holes / rim edges), not
    # just raw travel into mid-tone table behind the lid.
    left_q = _side_quality(
        value, saturation, edges, gy, depth_mm, cap_depth, x - 1, -1, left, y, bh
    )
    right_q = _side_quality(
        value, saturation, edges, gy, depth_mm, cap_depth, x + bw, 1, right, y, bh
    )
    pick_left = False
    if left >= min_cup and right >= min_cup:
        pick_left = (left_q, left) >= (right_q, right)
    elif left >= min_cup:
        pick_left = True
    elif right >= min_cup:
        pick_left = False
    else:
        return None
    if pick_left:
        grown = (x - left, y - pad, bw + left + pad, bh + pad * 2)
    else:
        grown = (x - pad, y - pad, bw + right + pad, bh + pad * 2)
    return _clip_box(grown, width, height)


def _side_quality(
    value,
    saturation,
    edges,
    gy,
    depth_mm,
    cap_depth,
    start: int,
    step: int,
    traveled: int,
    y: int,
    bh: int,
) -> float:
    """How cup-like the grown span is (0..1): holes + rim + mid body tone."""
    if traveled <= 0:
        return 0.0
    height, width = value.shape[:2]
    hole_acc = 0.0
    wall_acc = 0.0
    body_acc = 0.0
    n = 0
    side = max(2, bh // 8)
    for index in range(traveled):
        col = start + index * step
        if col < 0 or col >= width:
            break
        n += 1
        mean_v = float(value[y : y + bh, col].mean())
        mean_s = float(saturation[y : y + bh, col].mean())
        if 55 <= mean_v <= 210 and mean_s <= 90:
            body_acc += 1.0
        top_e = edges[y : y + side, col]
        bot_e = edges[y + bh - side : y + bh, col]
        walls = bool(top_e.any() or bot_e.any())
        if not walls and gy is not None:
            top_gy = float(np.mean(np.abs(gy[y : y + side, col])))
            bot_gy = float(np.mean(np.abs(gy[y + bh - side : y + bh, col])))
            walls = top_gy >= 12.0 or bot_gy >= 12.0
        if walls:
            wall_acc += 1.0
        if depth_mm is not None and cap_depth is not None:
            strip = depth_mm[y : y + bh, col]
            finite = np.isfinite(strip)
            valid = finite & (strip > _MIN_SURFACE_MM) & (strip < _MAX_DEPTH_MM)
            invalid = ~valid
            far = valid & (strip > float(cap_depth) + _CLEAR_FAR_DELTA_MM)
            hole = float(np.count_nonzero(invalid | far)) / max(float(strip.size), 1.0)
            if hole >= 0.35:
                hole_acc += 1.0
    if n == 0:
        return 0.0
    return (hole_acc * 0.5 + wall_acc * 0.3 + body_acc * 0.2) / float(n)


def _synthetic_sideways(w: int = 320, h: int = 240):
    """Left white lid + mid-tone body with top/bottom edges + depth holes."""
    image = np.full((h, w, 3), 90, dtype=np.uint8)
    depth = np.full((h, w), 900.0, dtype=np.float32)
    vx, vy, vw, vh = 110, 90, 70, 40
    lid_w = 14
    # Clear body (slightly darker) + horizontal rim.
    image[vy : vy + vh, vx + lid_w : vx + vw] = (70, 70, 65)
    image[vy, vx + lid_w : vx + vw] = (40, 40, 40)
    image[vy + vh - 1, vx + lid_w : vx + vw] = (40, 40, 40)
    image[vy : vy + vh, vx : vx + lid_w] = (245, 245, 245)
    # Opaque lid nearer than table; invalid depth through clear cup (holes).
    depth[vy : vy + vh, vx : vx + lid_w] = 520.0
    depth[vy : vy + vh, vx + lid_w : vx + vw] = 0.0
    gt = (vx, vy, vw, vh)
    return image, depth, gt


if __name__ == "__main__":
    # Allow `python sideways_race/methods/method_a_cap_horizontal_grow.py` from repo root.
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    color, depth, gt = _synthetic_sideways()
    boxes = detect_sideways(color, depth, intrinsics=(500.0, 500.0, 160.0, 120.0))
    print(f"NAME={NAME}")
    print(f"gt={gt}")
    print(f"detections={boxes}")
    if boxes:
        best = max(boxes, key=lambda b: _iou(b, gt))
        print(f"best_iou={_iou(best, gt):.3f} best={best}")
        ok = _iou(best, gt) >= 0.5
    else:
        ok = False
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
