"""Offline smoke for rack overlay drawing (no live ROS / cameras)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from orbbec_dual_sync.rack_overlay_draw import draw_rack_overlay, load_rack_rois


def _robotics() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "ros2_ws":
            return parent.parent / "robotics"
    raise FileNotFoundError("robotics root")


def test_draw_overlay_on_probe_still():
    robotics = _robotics()
    rack_json = robotics / "rack_lid_state" / "rack.json"
    still = robotics / "rack_lid_state" / "captures" / "probe_cam1.png"
    assert rack_json.is_file()
    assert still.is_file()
    classify, display, holes = load_rack_rois(rack_json)
    assert display[2] > classify[2]
    assert display[3] > classify[3]
    assert holes
    bgr = cv2.imread(str(still))
    assert bgr is not None
    view = draw_rack_overlay(bgr, display, "OPEN", classify_xywh=classify)
    assert view.shape == bgr.shape
    out = robotics / "rack_lid_state" / "captures" / "smoke_rack_overlay_open.png"
    assert cv2.imwrite(str(out), view)
    dx, dy, dw, dh = display
    crop = view[dy : dy + dh, dx : dx + dw]
    assert crop.size > 0
    assert float(np.mean(crop[:, :, 0])) > 0
