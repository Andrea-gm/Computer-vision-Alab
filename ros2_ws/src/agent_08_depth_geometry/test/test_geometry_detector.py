"""Unit checks for Agent 08 geometry detector (no ROS)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from agent_08_depth_geometry.geometry_detector import detect_vials, pseudo_depth_from_rgb


def _synth_sideways(width=640, height=400, lid_on_left=True, z_mm=350.0):
    image = np.full((height, width, 3), 95, dtype=np.uint8)
    length_mm, diam_mm = 72.69, 54.22
    fx = fy = float(width) / (2.0 * np.tan(np.radians(45.0)))
    length_px = int(round(length_mm * fx / z_mm))
    diam_px = int(round(diam_mm * fy / z_mm))
    length_px = max(40, min(width - 40, length_px))
    diam_px = max(24, min(height - 40, diam_px))
    lid_w = max(12, min(int(round(diam_px * 0.35)), length_px // 3))
    x0 = (width - length_px) // 2
    y0 = (height - diam_px) // 2
    image[y0 : y0 + diam_px, x0 : x0 + length_px] = (70, 72, 68)
    if lid_on_left:
        image[y0 : y0 + diam_px, x0 : x0 + lid_w] = (245, 245, 245)
    else:
        image[y0 : y0 + diam_px, x0 + length_px - lid_w : x0 + length_px] = (
            245,
            245,
            245,
        )
    gt = (x0, y0, length_px, diam_px)
    depth = np.full((height, width), 1800.0, dtype=np.float32)
    bx, by, bw, bh = gt
    if lid_on_left:
        depth[by : by + bh, bx : bx + lid_w] = z_mm
        body_x0, body_x1 = bx + lid_w, bx + bw
    else:
        depth[by : by + bh, bx + bw - lid_w : bx + bw] = z_mm
        body_x0, body_x1 = bx, bx + bw - lid_w
    depth[by : by + 3, body_x0:body_x1] = z_mm + 4.0
    depth[by + bh - 3 : by + bh, body_x0:body_x1] = z_mm + 4.0
    cup = depth[by + 3 : by + bh - 3, body_x0 + 2 : body_x1 - 2]
    if cup.size:
        cup[:] = 0.0
    K = (fx, fy, width * 0.5, height * 0.5)
    return image, depth, K, gt


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return float(inter / max(union, 1))


def test_sideways_detection_iou():
    image, depth, K, gt = _synth_sideways(lid_on_left=True)
    dets = detect_vials(image, depth, K, allow_pseudo_depth=False)
    assert dets, "expected at least one detection"
    best = max(_iou(gt, d.box_xywh) for d in dets)
    assert best >= 0.40, best


def test_pseudo_depth_shape():
    image, _, _, _ = _synth_sideways()
    d = pseudo_depth_from_rgb(image)
    assert d.shape == image.shape[:2]
    assert np.any(d > 0)


if __name__ == "__main__":
    test_sideways_detection_iou()
    test_pseudo_depth_shape()
    print("ok")
