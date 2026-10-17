"""Offline smoke for cam0 furnace overlay drawing (no live ROS / cameras)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from orbbec_dual_sync.furnace_overlay_draw import (
    default_furnace_json,
    paint_furnace_overlay,
    resolve_slots_for_frame,
)


def _robotics() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "ros2_ws":
            return parent.parent / "robotics"
    raise FileNotFoundError("robotics root")


def test_paint_furnace_overlay_on_capture():
    robotics = _robotics()
    furnace_json = default_furnace_json(robotics)
    caps = sorted(
        (robotics / "furnace_state" / "captures").glob("cam0_*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    assert furnace_json.is_file()
    assert caps, "need at least one furnace_state/captures/cam0_*.png"
    still = caps[0]
    bgr = cv2.imread(str(still))
    assert bgr is not None
    slots = resolve_slots_for_frame(
        bgr, robotics_root=robotics, furnace_json=furnace_json, match_live=True
    )
    assert len(slots) >= 1
    view = paint_furnace_overlay(
        bgr,
        robotics_root=robotics,
        furnace_json=furnace_json,
        match_live=True,
    )
    assert view.shape == bgr.shape
    out = robotics / "furnace_state" / "overlays" / "smoke_furnace_overlay_cam0.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(out), view)
    # Boxes should add non-black pixels near first slot.
    s0 = slots[0]
    crop = view[s0.y : s0.y + s0.h, s0.x : s0.x + s0.w]
    assert crop.size > 0
    assert float(np.mean(crop)) > 0
