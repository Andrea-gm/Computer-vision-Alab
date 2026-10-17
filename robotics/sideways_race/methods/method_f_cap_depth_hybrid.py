"""Detector F: white-cap seed -> depth-failure body with RGB wall support.

This method deliberately starts from an opaque, side-on cap and grows only
left/right.  Clear plastic is represented by invalid or unstable depth, while
horizontal Sobel walls bridge short runs of plausible see-through depth.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

NAME = "F_cap_depth_hybrid"

_MIN_Z = 80.0
_MAX_Z = 5000.0
_DEFAULT_FX = 575.0
_DIMS = Path(__file__).resolve().parents[2] / "vial_3d_dims.json"


def _physical_dims() -> tuple[float, float]:
    try:
        data = json.loads(_DIMS.read_text(encoding="utf-8"))
        return float(data["total_height_mm"]), float(data["lid_diameter_mm"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 72.69, 54.22


def _fx_fy(intrinsics) -> tuple[float, float]:
    if isinstance(intrinsics, dict):
        return float(intrinsics.get("fx", _DEFAULT_FX)), float(
            intrinsics.get("fy", intrinsics.get("fx", _DEFAULT_FX))
        )
    if intrinsics is not None:
        try:
            return float(intrinsics[0]), float(intrinsics[1])
        except (IndexError, TypeError, ValueError):
            pass
    return _DEFAULT_FX, _DEFAULT_FX


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _cap_depth(depth: np.ndarray, box) -> float | None:
    x, y, w, h = box
    inset_x, inset_y = max(1, w // 4), max(1, h // 5)
    patch = depth[
        y + inset_y : y + h - inset_y,
        x + inset_x : x + w - inset_x,
    ]
    valid = patch[np.isfinite(patch) & (patch > _MIN_Z) & (patch < _MAX_Z)]
    return float(np.median(valid)) if valid.size >= 6 else None


def _cap_candidates(image: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
    """Strict compact white components; no horizontal closing into wall slabs."""
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 2] >= 195) & (hsv[:, :, 1] <= 55)).astype(np.uint8) * 255
    white = cv2.morphologyEx(
        white, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = float(cv2.contourArea(cnt))
        fill = area / max(float(bw * bh), 1.0)
        # A fallen cap is an end strip, usually taller than wide.  The loose
        # upper ratio retains mildly oblique caps but rejects upright lids.
        if (
            area >= 30
            and bw >= 5
            and bh >= 10
            and 0.12 <= bw / max(bh, 1) <= 1.15
            and fill >= 0.55
            and bw < 0.25 * w
            and bh < 0.35 * h
        ):
            out.append(((x, y, bw, bh), fill))
    return out


def _column_cues(
    depth: np.ndarray,
    value: np.ndarray,
    saturation: np.ndarray,
    gy: np.ndarray,
    x: int,
    y: int,
    h: int,
    cap_z: float,
) -> tuple[bool, bool, float]:
    """Return strong-depth-failure, RGB-wall bridge, and a quality score."""
    inset = max(1, h // 8)
    y0, y1 = y + inset, y + h - inset
    d = depth[y0:y1, x]
    valid = np.isfinite(d) & (d > _MIN_Z) & (d < _MAX_Z)
    invalid_frac = float((~valid).mean())
    usable = d[valid]
    noisy = float(np.std(usable)) >= 45.0 if usable.size >= 4 else False
    mixed = (
        float((usable > cap_z + 180.0).mean()) >= 0.35
        and float((usable < cap_z + 100.0).mean()) >= 0.12
        if usable.size >= 5
        else False
    )
    strong = invalid_frac >= 0.38 or noisy or mixed

    rim = max(2, h // 9)
    top = float(np.mean(np.abs(gy[y : y + rim, x])))
    bot = float(np.mean(np.abs(gy[y + h - rim : y + h, x])))
    mean_v = float(np.mean(value[y0:y1, x]))
    mean_s = float(np.mean(saturation[y0:y1, x]))
    body_tone = 28.0 <= mean_v <= 190.0 and mean_s <= 115.0
    wall = body_tone and (top >= 18.0 or bot >= 18.0)
    quality = 1.8 * invalid_frac + 0.35 * float(noisy) + 0.25 * float(wall)
    return strong, wall, quality


def _grow_side(
    depth: np.ndarray,
    value: np.ndarray,
    saturation: np.ndarray,
    gy: np.ndarray,
    cap,
    cap_z: float,
    step: int,
) -> tuple[int, float]:
    """Grow from one cap edge, bridging isolated valid-depth columns."""
    frame_w = depth.shape[1]
    x, y, bw, bh = cap
    start = x - 1 if step < 0 else x + bw
    max_len = min(int(2.8 * bh), frame_w)
    extent = 0
    last_strong = 0
    gap = 0
    strong_count = 0
    wall_count = 0
    score = 0.0
    for i in range(max_len):
        col = start + i * step
        if not 0 <= col < frame_w:
            break
        strong, wall, q = _column_cues(
            depth, value, saturation, gy, col, y, bh, cap_z
        )
        if strong:
            strong_count += 1
            last_strong = i + 1
            gap = 0
            score += q
        elif wall and strong_count >= 2:
            wall_count += 1
            gap += 1
            score += q
        else:
            gap += 1
        # One valid stripe is common in transparent depth.  Stop before desk
        # pixels once two consecutive unsupported columns appear.
        if gap >= 2:
            break
        extent = i + 1

    extent = max(last_strong, extent - gap)
    minimum = max(8, int(0.25 * bh))
    if extent < minimum or strong_count < max(4, int(0.35 * extent)):
        return 0, 0.0
    return extent, score + 0.2 * wall_count


def _scale_plausible(box, z: float, intrinsics) -> bool:
    """Generous optional H/diameter-at-Z gate, mainly excluding wall blobs."""
    _, _, bw, bh = box
    fx, fy = _fx_fy(intrinsics)
    if fx <= 1 or fy <= 1:
        return True
    total_h, lid_d = _physical_dims()
    long_mm, short_mm = bw * z / fx, bh * z / fy
    return (
        0.32 * total_h <= long_mm <= 2.5 * total_h
        and 0.28 * lid_d <= short_mm <= 2.1 * lid_d
    )


def detect_sideways(
    color_bgr, depth_mm, intrinsics=None
) -> list[tuple[int, int, int, int]]:
    """Return cap-through-body AABBs for fallen clear vials."""
    if color_bgr is None or depth_mm is None:
        return []
    image = np.asarray(color_bgr)
    depth = np.asarray(depth_mm, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] < 3 or depth.ndim != 2:
        return []
    h, w = image.shape[:2]
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value, saturation = hsv[:, :, 2], hsv[:, :, 1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    ranked: list[tuple[float, tuple[int, int, int, int]]] = []

    for cap, solidity in _cap_candidates(image):
        x, y, bw, bh = cap
        z = _cap_depth(depth, cap)
        if z is None:
            continue  # opaque anchor must itself have trustworthy depth
        left_len, left_score = _grow_side(
            depth, value, saturation, gy, cap, z, -1
        )
        right_len, right_score = _grow_side(
            depth, value, saturation, gy, cap, z, 1
        )
        if max(left_score, right_score) <= 0:
            continue
        if right_score >= left_score:
            box = (x, y, bw + right_len, bh)
            body_len, cue_score = right_len, right_score
        else:
            box = (x - left_len, y, bw + left_len, bh)
            body_len, cue_score = left_len, left_score
        bx, by, bwidth, bheight = box
        if bx < 0 or bx + bwidth > w:
            continue
        aspect = bheight / max(float(bwidth), 1.0)
        if not (0.32 <= aspect <= 0.82 and bwidth >= 1.25 * bheight):
            continue
        if not _scale_plausible(box, z, intrinsics):
            continue
        body_ratio = body_len / max(float(bh), 1.0)
        rank = 0.45 * solidity + 0.35 * min(body_ratio, 1.5) + 0.20 * min(
            cue_score / max(body_len, 1), 1.0
        )
        ranked.append((rank, tuple(int(v) for v in box)))

    ranked.sort(reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for _, box in ranked:
        if not any(_iou(box, other) >= 0.40 for other in kept):
            kept.append(box)
    return kept


def _self_test():
    image = np.full((240, 320, 3), 90, np.uint8)
    depth = np.full((240, 320), 1100.0, np.float32)
    gt = (105, 90, 82, 42)
    image[90:132, 105:187] = (67, 67, 62)
    image[90, 121:187] = image[131, 121:187] = (35, 35, 35)
    image[90:132, 105:121] = (245, 245, 245)
    depth[90:132, 105:121] = 520.0
    depth[90:132, 121:187] = 0.0
    depth[90:132, 130:187:9] = 1120.0
    boxes = detect_sideways(image, depth, {"fx": 575.0, "fy": 575.0})
    best = max((_iou(gt, b) for b in boxes), default=0.0)
    print(f"NAME={NAME} gt={gt} detections={boxes} best_iou={best:.3f}")
    upright = np.full_like(image, 90)
    upright_d = np.full_like(depth, 1100.0)
    upright[70:82, 140:178] = 245
    upright[82:135, 142:176] = 65
    upright_d[70:82, 140:178] = 520.0
    upright_d[82:135, 142:176] = 0.0
    rejects = detect_sideways(upright, upright_d)
    print(f"upright_detections={rejects}")
    return best >= 0.5 and not rejects


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
