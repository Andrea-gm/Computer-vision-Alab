"""Method B2: B plus a conservative depth-hole/cap geometry fallback.

The original B is retained as the primary detector.  B2 additionally bridges
the narrow valid-depth stripes commonly returned through a clear vial, finds
the horizontal body component, and accepts it only when an opaque white,
near-depth cap is attached to exactly one end.
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    from sideways_race.methods.method_b_rotated_white_cap import (
        _align_depth,
        _iou,
        detect_sideways as _detect_b,
    )
except ModuleNotFoundError:  # Direct ``python methods/method_b2_improved.py``.
    from method_b_rotated_white_cap import (  # type: ignore[no-redef]
        _align_depth,
        _iou,
        detect_sideways as _detect_b,
    )

NAME = "B2_improved"

_MIN_DEPTH_MM = 80.0
_MAX_DEPTH_MM = 5000.0
_MIN_BODY_AREA = 180


def _cap_run(
    white: np.ndarray,
    depth: np.ndarray,
    body_box: tuple[int, int, int, int],
    side: str,
) -> tuple[int, float]:
    """Return attached cap width and evidence score on one body end."""
    frame_h, frame_w = white.shape
    x, y, bw, bh = body_box
    # Include a little vertical context, but keep the scan tied to the body.
    pad_y = max(1, bh // 8)
    y0, y1 = max(0, y - pad_y), min(frame_h, y + bh + pad_y)
    max_run = max(10, int(round(max(bh, 12) * 0.75)))
    step = -1 if side == "left" else 1
    start = x - 1 if side == "left" else x + bw

    run = 0
    score = 0.0
    gaps = 0
    for i in range(max_run):
        col = start + i * step
        if col < 0 or col >= frame_w:
            break
        wfrac = float(white[y0:y1, col].mean())
        dcol = depth[y0:y1, col]
        valid = dcol[(dcol > _MIN_DEPTH_MM) & (dcol < _MAX_DEPTH_MM)]
        # The cap is both bright and depth-valid.  A weak-white allowance
        # handles exposure/noise, while depth prevents accepting RGB clutter.
        vfrac = float(valid.size / max(dcol.size, 1))
        near = float(np.median(valid)) if valid.size else np.inf
        is_cap = (wfrac >= 0.30 and vfrac >= 0.30) or (
            wfrac >= 0.16 and vfrac >= 0.70 and near < 750.0
        )
        if is_cap:
            run = i + 1
            score += wfrac + 0.5 * vfrac
            gaps = 0
        else:
            gaps += 1
            if gaps >= 2:
                break
    return max(0, run - max(0, gaps)), score


def _depth_hole_fallback(
    color_bgr: np.ndarray,
    depth: np.ndarray | None,
) -> list[tuple[int, int, int, int]]:
    if depth is None:
        return []
    frame_h, frame_w = color_bgr.shape[:2]

    # Bridge 1--2 pixel valid-depth stripes through transparent plastic, but
    # do not join unrelated holes vertically.
    holes = (depth <= _MIN_DEPTH_MM).astype(np.uint8)
    holes = cv2.morphologyEx(
        holes,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)),
    )
    holes = cv2.morphologyEx(
        holes,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)),
    )

    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 2] >= 180) & (hsv[:, :, 1] <= 65)).astype(np.float32)
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(holes, 8)
    out: list[tuple[int, int, int, int]] = []

    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if area < _MIN_BODY_AREA or bw < 18 or bh < 8:
            continue
        if bw / max(bh, 1) < 1.05:
            continue
        if bw > 0.30 * frame_w or bh > 0.18 * frame_h:
            continue
        fill = area / max(bw * bh, 1)
        if fill < 0.55:
            continue

        left_w, left_s = _cap_run(white, depth, (x, y, bw, bh), "left")
        right_w, right_s = _cap_run(white, depth, (x, y, bw, bh), "right")
        min_cap = max(3, int(round(max(bh, 12) * 0.16)))
        left_ok = left_w >= min_cap and left_s >= 2.0
        right_ok = right_w >= min_cap and right_s >= 2.0
        # A sideways vial has one lid. Requiring exactly one cap also rejects
        # holes between two bright slabs.
        if left_ok == right_ok:
            continue
        cap_w = left_w if left_ok else right_w

        total_w = bw + cap_w
        # If the body is partly hidden by a slab, recover its expected height
        # from the observed vial width. Clean components retain measured h.
        inferred_h = int(round(total_w * 0.62))
        target_h = max(bh, inferred_h if bh < 0.48 * total_w else bh)
        target_h = min(target_h, int(round(total_w * 0.72)))
        target_h = max(target_h, 10)

        x0 = x - cap_w if left_ok else x
        # Detect which vertical edge is occluded and grow only that way.
        extra = max(0, target_h - bh)
        above = white[max(0, y - 2) : y, x : x + bw]
        below = white[y + bh : min(frame_h, y + bh + 2), x : x + bw]
        above_score = float(above.mean()) if above.size else 0.0
        below_score = float(below.mean()) if below.size else 0.0
        if extra and above_score >= below_score:
            y0 = y - extra
        else:
            y0 = y

        x0 = max(0, x0)
        y0 = max(0, y0)
        total_w = min(total_w, frame_w - x0)
        target_h = min(target_h, frame_h - y0)
        hw = target_h / max(total_w, 1)
        if 0.40 <= hw <= 0.71:
            out.append((x0, y0, total_w, target_h))
    return out


def _merge_boxes(
    primary: list[tuple[int, int, int, int]],
    fallback: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    # Prefer the depth-derived box when it overlaps B: it normally follows the
    # full body silhouette and fixes B's occasional short grow.
    result = list(fallback)
    for box in primary:
        if not any(_iou(box, other) >= 0.45 for other in result):
            result.append(box)
    return result


def detect_sideways(
    color_bgr,
    depth_mm,
    intrinsics=None,
) -> list[tuple[int, int, int, int]]:
    """Detect sideways vials with B and depth-hole/cap recovery."""
    if color_bgr is None or getattr(color_bgr, "size", 0) == 0:
        return []
    frame_h, frame_w = color_bgr.shape[:2]
    depth = _align_depth(depth_mm, frame_w, frame_h)
    primary = _detect_b(color_bgr, depth, intrinsics)
    fallback = _depth_hole_fallback(color_bgr, depth)
    return _merge_boxes(primary, fallback)


if __name__ == "__main__":
    try:
        from sideways_race.methods.method_b_rotated_white_cap import _synth_sideways
    except ModuleNotFoundError:
        from method_b_rotated_white_cap import _synth_sideways

    for lid_left in (True, False):
        image, depth_image, gt = _synth_sideways(lid_left=lid_left)
        detections = detect_sideways(image, depth_image)
        best = max((_iou(box, gt) for box in detections), default=0.0)
        print(f"lid_left={lid_left} gt={gt} detections={detections} best_iou={best:.3f}")
        if best < 0.5:
            raise SystemExit(1)

    wall = np.full((240, 320, 3), 200, dtype=np.uint8)
    wall_depth = np.full((240, 320), 800.0, dtype=np.float32)
    if detect_sideways(wall, wall_depth):
        raise SystemExit("flat wall produced a false positive")
    print("self-test OK")
