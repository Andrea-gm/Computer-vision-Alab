"""Baseline: current depthcamera cap-first + sideways geometry filter.

Does not open cameras. Safe to import for eval only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

NAME = "baseline_depthcamera"

_CAMERA = Path(__file__).resolve().parents[2]
if str(_CAMERA) not in sys.path:
    sys.path.insert(0, str(_CAMERA))


def detect_sideways(color_bgr, depth_mm, intrinsics=None) -> list[tuple[int, int, int, int]]:
    """Return sideways lid+cup boxes using live depthcamera helpers if importable."""
    try:
        import depthcamera as dc
    except Exception:
        return []

    dc.set_allow_sideways(True)
    dc.set_sideways_assist(True)

    depth = None if depth_mm is None else np.asarray(depth_mm, dtype=np.float32)
    K = None
    if isinstance(intrinsics, dict):
        try:
            K = {
                "fx": float(intrinsics["fx"]),
                "fy": float(intrinsics["fy"]),
                "cx": float(intrinsics["cx"]),
                "cy": float(intrinsics["cy"]),
            }
        except Exception:
            K = None

    found: list[tuple[int, int, int, int]] = []
    try:
        raw = dc.detect_vial_by_cap(color_bgr, depth, K)
    except Exception:
        return []

    for box in raw or []:
        try:
            if dc._is_sideways_box(box) and dc._sideways_pose(box, image=color_bgr):
                found.append(tuple(int(v) for v in box))
        except Exception:
            continue

    # Soft fallback: geometry-only sideways if pose helper is strict on synth.
    if not found:
        for box in raw or []:
            try:
                if dc._is_sideways_box(box):
                    found.append(tuple(int(v) for v in box))
            except Exception:
                continue
    return found


if __name__ == "__main__":
    import cv2

    # Minimal synthetic self-check
    img = np.full((240, 320, 3), 90, dtype=np.uint8)
    vx, vy, vw, vh = 110, 90, 70, 40
    img[vy : vy + vh, vx + 14 : vx + vw] = (70, 70, 65)
    img[vy : vy + vh, vx : vx + 14] = (245, 245, 245)
    depth = np.full((240, 320), 900.0, dtype=np.float32)
    depth[vy : vy + vh, vx : vx + 14] = 520.0
    depth[vy : vy + vh, vx + 14 : vx + vw] = 0.0
    boxes = detect_sideways(img, depth, None)
    print(NAME, "boxes=", boxes)
    # Optional visualize
    vis = img.copy()
    for x, y, bw, bh in boxes:
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
    print("ok" if boxes else "no detections (depthcamera import/path may have failed)")
