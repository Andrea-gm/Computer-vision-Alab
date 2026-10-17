"""Sideways vial detector D: Sobel edge capsule template.

Looks for a horizontal capsule: long opposing |Gy| walls (top/bottom), plus a
white end-cap with a vertical |Gx| rim. Scores sliding windows around white
blobs and elongated |Gy|-contour groups.

API for the sideways_race harness:
    detect_sideways(color_bgr, depth_mm, intrinsics=None) -> list[tuple[x,y,w,h]]
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

NAME = "D_edge_template"

# Sideways lid+cup geometry (h/w), matching live Camera convention.
_ASPECT_MIN = 0.38
_ASPECT_MAX = 0.75
_MIN_W = 28
_MIN_H = 14
_MAX_FRAC_W = 0.85
_MAX_FRAC_H = 0.55
_SCORE_FLOOR = 0.28
_NMS_IOU = 0.45
_MAX_BOXES = 8


def detect_sideways(
    color_bgr,
    depth_mm,
    intrinsics=None,
) -> list[tuple[int, int, int, int]]:
    """Detect sideways (fallen) vials as axis-aligned boxes (x, y, w, h).

    ``depth_mm`` and ``intrinsics`` are accepted for harness compatibility; this
    method is primarily RGB edge/template based (depth only soft-rejects walls).
    """
    if color_bgr is None or getattr(color_bgr, "size", 0) == 0:
        return []
    image = np.asarray(color_bgr)
    if image.ndim != 3 or image.shape[2] < 3:
        return []
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    gx, gy, mag = _sobel_derivatives(gray)
    white = _white_mask(val, sat)
    depth = None
    if depth_mm is not None:
        d = np.asarray(depth_mm)
        if d.shape[:2] == (height, width):
            depth = d

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []

    # Path 1: white end-cap seeds → horizontal capsule windows.
    for seed in _white_blobs(white, width, height):
        for box in _windows_from_seed(seed, width, height):
            score = _capsule_score(image, gray, gx, gy, mag, white, box, depth)
            if score >= _SCORE_FLOOR:
                candidates.append((score, box))

    # Path 2: elongated contour groups from strong horizontal |Gy| edges.
    for box in _gy_contour_windows(gy, mag, width, height):
        score = _capsule_score(image, gray, gx, gy, mag, white, box, depth)
        if score >= _SCORE_FLOOR:
            candidates.append((score, box))

    # Path 3: coarse sliding windows near white pixels (recall backup).
    for box in _slide_near_white(white, width, height):
        score = _capsule_score(image, gray, gx, gy, mag, white, box, depth)
        if score >= _SCORE_FLOOR + 0.05:
            candidates.append((score, box))

    if not candidates:
        return []

    candidates.sort(key=lambda t: t[0], reverse=True)
    return _nms(candidates, _NMS_IOU, _MAX_BOXES)


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------


def _sobel_derivatives(gray: np.ndarray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return gx, gy, mag


def _white_mask(val: np.ndarray, sat: np.ndarray) -> np.ndarray:
    mask = ((val >= 155) & (sat <= 70)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def _mean_abs(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr)))


def _clip_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = [int(v) for v in box]
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    w = max(1, min(width - x, w))
    h = max(1, min(height - y, h))
    return x, y, w, h


def _aspect_ok(w: int, h: int) -> bool:
    if w < _MIN_W or h < _MIN_H:
        return False
    aspect = h / max(w, 1)
    return _ASPECT_MIN <= aspect <= _ASPECT_MAX


def _white_end_score(white: np.ndarray, box) -> tuple[float, str]:
    """Return (score, side) where side is 'left' or 'right'."""
    x, y, w, h = box
    roi = white[y : y + h, x : x + w] > 0
    if roi.size == 0 or w < 12:
        return 0.0, "left"
    vband = max(3, int(w * 0.32))
    if w <= vband * 2:
        return 0.0, "left"
    left = float(roi[:, :vband].mean())
    right = float(roi[:, -vband:].mean())
    mid = float(roi[:, vband : w - vband].mean())
    hband = max(3, int(h * 0.34))
    top = float(roi[:hband, :].mean())
    bot = float(roi[-hband:, :].mean())
    if left >= right:
        end, side = left, "left"
    else:
        end, side = right, "right"
    if end < 0.22 or mid > end - 0.08:
        return 0.0, side
    # Prefer end lid over top/bottom white (rejects upright / top-down).
    score = end - max(mid, top * 0.85, bot * 0.85) + 0.12
    return float(max(0.0, min(1.0, score))), side


def _horizontal_wall_score(gy: np.ndarray, box) -> float:
    """Opposing |Gy| on top/bottom bands vs quieter middle (horizontal cup)."""
    x, y, w, h = box
    roi = gy[y : y + h, x : x + w]
    if roi.size == 0 or h < 6:
        return 0.0
    band = max(1, min(4, h // 5))
    top = _mean_abs(roi[:band, :])
    bot = _mean_abs(roi[-band:, :])
    mid = _mean_abs(roi[band : max(band + 1, h - band), :])
    opposing = min(top, bot) / max(mid, 1.0)
    balance = min(top, bot) / max(max(top, bot), 1.0)
    strength = min(1.0, (top + bot) * 0.5 / 35.0)
    return float(0.55 * min(1.6, opposing) / 1.6 + 0.25 * balance + 0.20 * strength)


def _end_rim_gx_score(gx: np.ndarray, box, side: str) -> float:
    """Vertical |Gx| rim energy on the white-cap end."""
    x, y, w, h = box
    roi = gx[y : y + h, x : x + w]
    if roi.size == 0 or w < 8:
        return 0.0
    vband = max(2, int(w * 0.22))
    if side == "left":
        end = _mean_abs(roi[:, :vband])
        far = _mean_abs(roi[:, -vband:])
    else:
        end = _mean_abs(roi[:, -vband:])
        far = _mean_abs(roi[:, :vband])
    mid_w0, mid_w1 = vband, max(vband + 1, w - vband)
    mid = _mean_abs(roi[:, mid_w0:mid_w1])
    # Cap rim should show stronger vertical edge than the clear mid-body.
    contrast = end / max(mid, 1.0)
    prefer = end / max(far, 1.0)
    return float(
        0.65 * min(1.5, contrast) / 1.5
        + 0.20 * min(1.5, prefer) / 1.5
        + 0.15 * min(1.0, end / 40.0)
    )


def _body_elongation(box) -> float:
    x, y, w, h = box
    aspect = h / max(w, 1)
    # Peak near ~0.52 (typical sideways lid+cup); prefer longer bodies.
    target = 0.52
    dist = abs(aspect - target) / 0.22
    shape = max(0.0, 1.0 - dist)
    length = min(1.0, w / max(2.4 * h, 1))
    return float(0.55 * shape + 0.45 * length)


def _depth_soft_ok(depth: Optional[np.ndarray], box) -> float:
    """Soft prior: reject huge near walls; clear plastic may be sparse — do not hard-fail."""
    if depth is None:
        return 0.5
    x, y, w, h = box
    roi = depth[y : y + h, x : x + w]
    valid = roi[(roi > 80) & (roi < 4000)]
    if valid.size < 8:
        # Many zeros through clear cup is fine for sideways.
        return 0.55
    med = float(np.median(valid))
    if med < 120:
        return 0.15  # likely wall / contact
    if med > 3200:
        return 0.25
    return 0.55 + 0.45 * min(1.0, (med - 120) / 800.0)


def _capsule_score(
    image,
    gray,
    gx,
    gy,
    mag,
    white,
    box,
    depth,
) -> float:
    height, width = gray.shape[:2]
    box = _clip_box(box, width, height)
    x, y, w, h = box
    if not _aspect_ok(w, h):
        return 0.0
    if w > _MAX_FRAC_W * width or h > _MAX_FRAC_H * height:
        return 0.0

    white_score, side = _white_end_score(white, box)
    if white_score < 0.12:
        return 0.0
    wall = _horizontal_wall_score(gy, box)
    if wall < 0.12:
        return 0.0
    rim = _end_rim_gx_score(gx, box, side)
    elong = _body_elongation(box)
    depth_s = _depth_soft_ok(depth, box)

    # Quiet interior vs rim (capsule silhouette).
    roi_mag = mag[y : y + h, x : x + w]
    pad = max(1, min(3, min(w, h) // 6))
    ring = np.ones((h, w), dtype=bool)
    if h > 2 * pad and w > 2 * pad:
        ring[pad : h - pad, pad : w - pad] = False
    ring_m = float(roi_mag[ring].mean()) if ring.any() else 0.0
    inner_m = float(roi_mag[~ring].mean()) if (~ring).any() else 0.0
    silhouette = min(1.5, ring_m / max(inner_m, 1.0)) / 1.5

    score = (
        0.26 * wall
        + 0.24 * white_score
        + 0.16 * rim
        + 0.20 * elong
        + 0.08 * silhouette
        + 0.06 * depth_s
    )
    return float(score)


# ---------------------------------------------------------------------------
# Proposal generators
# ---------------------------------------------------------------------------


def _white_blobs(white: np.ndarray, width: int, height: int):
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    min_area = max(40, int(0.0002 * width * height))
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area < min_area or w < 8 or h < 6:
            continue
        if w > 0.45 * width or h > 0.45 * height:
            continue
        # Cap-like: not extremely elongated vertically (upright lid strip).
        if h > 2.2 * w:
            continue
        blobs.append((x, y, w, h))
    return blobs


def _windows_from_seed(seed, width: int, height: int):
    """Grow horizontal capsule windows with lid on left or right of the seed."""
    sx, sy, sw, sh = seed
    cy = sy + sh // 2
    # Body height ≈ lid height; length ≈ 1.3–2.2 × height (sideways vial).
    heights = sorted({max(_MIN_H, int(sh * s)) for s in (0.85, 1.0, 1.2, 1.45)})
    boxes = []
    for bh in heights:
        if bh > int(_MAX_FRAC_H * height):
            continue
        y0 = int(cy - bh // 2)
        y0 = max(0, min(height - bh, y0))
        for aspect in (0.45, 0.52, 0.58, 0.65):
            bw = int(bh / aspect)
            if bw < _MIN_W or bw > int(_MAX_FRAC_W * width):
                continue
            # Lid on left / right of capsule relative to seed center.
            x_left = sx + sw // 2 - int(bw * 0.18)
            x_right = sx + sw // 2 - int(bw * 0.82)
            for x0 in (x_left, x_right, sx - 4, sx + sw - bw + 4):
                x0 = max(0, min(width - bw, int(x0)))
                if _aspect_ok(bw, bh):
                    boxes.append((x0, y0, bw, bh))
    # Dedup
    uniq = []
    seen = set()
    for b in boxes:
        key = (b[0] // 4, b[1] // 4, b[2] // 4, b[3] // 4)
        if key not in seen:
            seen.add(key)
            uniq.append(b)
    return uniq


def _gy_contour_windows(gy: np.ndarray, mag: np.ndarray, width: int, height: int):
    """Group strong |Gy| edges into elongated horizontal candidates."""
    thr = max(18.0, float(np.percentile(np.abs(gy), 72)))
    strong = (np.abs(gy) >= thr).astype(np.uint8) * 255
    # Prefer horizontal runs: dilate along x.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
    strong = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(strong, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < _MIN_W or h < 4:
            continue
        # Thin horizontal edge strip → expand to capsule height.
        if h <= 10 and w >= 2.2 * max(h, 1):
            bh = max(_MIN_H, int(w * 0.55))
            bh = min(bh, int(_MAX_FRAC_H * height))
            cy = y + h // 2
            y0 = max(0, min(height - bh, cy - bh // 2))
            # Slightly widen.
            pad = max(4, w // 12)
            x0 = max(0, x - pad)
            bw = min(width - x0, w + 2 * pad)
            if _aspect_ok(bw, bh):
                boxes.append((x0, y0, bw, bh))
        elif _aspect_ok(w, h):
            boxes.append((x, y, w, h))
        elif w > h and h >= _MIN_H:
            # Clip / pad toward sideways aspect.
            target_h = int(np.clip(w * 0.55, _MIN_H, _MAX_FRAC_H * height))
            cy = y + h // 2
            y0 = max(0, min(height - target_h, cy - target_h // 2))
            if _aspect_ok(w, target_h):
                boxes.append((x, y0, w, target_h))
    return boxes[:40]


def _slide_near_white(white: np.ndarray, width: int, height: int):
    """Sparse sliding windows centered on white mass (recall)."""
    ys, xs = np.where(white > 0)
    if xs.size == 0:
        return []
    # Subsample white pixels.
    step = max(1, xs.size // 60)
    boxes = []
    for i in range(0, xs.size, step):
        cx, cy = int(xs[i]), int(ys[i])
        for aspect in (0.48, 0.58):
            for bh in (18, 28, 40):
                if bh > int(_MAX_FRAC_H * height):
                    continue
                bw = int(bh / aspect)
                if bw < _MIN_W or bw > int(_MAX_FRAC_W * width):
                    continue
                for lid_frac in (0.15, 0.85):
                    x0 = int(cx - lid_frac * bw)
                    y0 = int(cy - bh // 2)
                    x0 = max(0, min(width - bw, x0))
                    y0 = max(0, min(height - bh, y0))
                    boxes.append((x0, y0, bw, bh))
    # Cap count
    if len(boxes) > 80:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(boxes), size=80, replace=False)
        boxes = [boxes[i] for i in idx]
    return boxes


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


def _nms(
    scored: list[tuple[float, tuple[int, int, int, int]]],
    iou_thr: float,
    max_boxes: int,
) -> list[tuple[int, int, int, int]]:
    keep: list[tuple[int, int, int, int]] = []
    for score, box in scored:
        if any(_iou(box, k) >= iou_thr for k in keep):
            continue
        keep.append(box)
        if len(keep) >= max_boxes:
            break
    return keep


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _synth_sideways(lid_on_left: bool = True, size=(240, 320)):
    """Synthetic sideways vial: translucent body + opaque white end-cap."""
    h, w = size
    img = np.full((h, w, 3), 42, dtype=np.uint8)
    # Table gradient
    for y in range(h):
        img[y, :] = (42 + y // 8, 44 + y // 10, 48 + y // 9)

    body_h = 36
    body_w = 72
    lid_w = 22
    total_w = body_w + lid_w
    y0 = h // 2 - body_h // 2
    x0 = w // 2 - total_w // 2
    if lid_on_left:
        lid = (x0, y0, lid_w, body_h)
        body = (x0 + lid_w, y0 + 2, body_w, body_h - 4)
    else:
        body = (x0, y0 + 2, body_w, body_h - 4)
        lid = (x0 + body_w, y0, lid_w, body_h)

    # Clear-ish body (darker teal-gray) with edge walls.
    bx, by, bw, bh = body
    img[by : by + bh, bx : bx + bw] = (70, 85, 78)
    # Top/bottom horizontal edges (Gy response).
    img[by : by + 2, bx : bx + bw] = (20, 25, 22)
    img[by + bh - 2 : by + bh, bx : bx + bw] = (20, 25, 22)
    # Soft interior mottling
    noise = np.random.default_rng(1).integers(0, 18, (bh, bw, 3), dtype=np.uint8)
    img[by : by + bh, bx : bx + bw] = np.clip(
        img[by : by + bh, bx : bx + bw].astype(np.int16) + noise - 6, 0, 255
    ).astype(np.uint8)

    # White opaque lid with vertical rim.
    lx, ly, lw, lh = lid
    img[ly : ly + lh, lx : lx + lw] = (235, 235, 238)
    if lid_on_left:
        img[ly : ly + lh, lx + lw - 2 : lx + lw] = (180, 180, 185)
    else:
        img[ly : ly + lh, lx : lx + 2] = (180, 180, 185)

    depth = np.full((h, w), 0, dtype=np.uint16)
    # Lid has solid depth; body mostly holes (clear plastic).
    depth[ly : ly + lh, lx : lx + lw] = 900
    depth[by : by + bh, bx : bx + bw : 5] = 920

    gt = (min(lx, bx), min(ly, by), total_w, body_h)
    return img, depth, gt


def _iou_box(a, b) -> float:
    return _iou(a, b)


if __name__ == "__main__":
    img, depth, gt = _synth_sideways(lid_on_left=True)
    boxes = detect_sideways(img, depth, None)
    print(f"NAME={NAME}")
    print(f"gt={gt}")
    print(f"detections={boxes}")
    if boxes:
        best = max(boxes, key=lambda b: _iou_box(b, gt))
        print(f"best={best} IoU={_iou_box(best, gt):.3f}")
    else:
        print("best=None IoU=0.000")

    img2, depth2, gt2 = _synth_sideways(lid_on_left=False)
    boxes2 = detect_sideways(img2, depth2, None)
    print(f"gt_right={gt2}")
    print(f"detections_right={boxes2}")
    if boxes2:
        best2 = max(boxes2, key=lambda b: _iou_box(b, gt2))
        print(f"best_right={best2} IoU={_iou_box(best2, gt2):.3f}")
    else:
        print("best_right=None IoU=0.000")
