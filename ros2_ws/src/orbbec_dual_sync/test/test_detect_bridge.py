"""Regression checks for the ROS-to-depthcamera detection bridge."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMERA_ROOT = ROOT.parents[2]
for path in (ROOT, CAMERA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from orbbec_dual_sync.detect_bridge import DetectionWorkerClient, DepthcameraDetectSession
from sideways_race.methods.method_b_rotated_white_cap import _synth_sideways
from sideways_race.methods.method_i_hard_robust import detect_sideways as promoted_winner


def test_ros_bridge_uses_promoted_sideways_winner(monkeypatch):
    session = DepthcameraDetectSession(model=CAMERA_ROOT / "__missing_ros_model__.pt")
    assert session.sideways_detector == "I_hard_robust"
    assert session.dc._detect_sideways_race_winner is promoted_winner

    # Force the primary cap proposal stage to miss so this synthetic frame
    # exercises the same promoted fallback used by depthcamera.locate.
    monkeypatch.setattr(session.dc, "detect_vial_by_cap", lambda *_args: [])
    color, depth, _expected = _synth_sideways()
    result = session.locate(0, color, depth)

    assert result.has_vial
    assert result.detection_source == "sideways-race-winner"


def test_ros_yolo_stability_profile_is_symmetric():
    session = DepthcameraDetectSession(
        model=CAMERA_ROOT / "__missing_ros_model__.pt",
        use_yolo=True,
        weak_camera=0,
        symmetric_yolo_gates=True,
    )

    assert session.gate_map[0].name == "weak"
    assert session.gate_map[1].name == "weak"
    assert session.gate_map[0].yolo_conf == session.gate_map[1].yolo_conf == 0.05


def test_classical_fallback_keeps_symmetric_stability_profiles():
    session = DepthcameraDetectSession(
        model=CAMERA_ROOT / "__missing_ros_model__.pt",
        use_yolo=False,
        weak_camera=0,
        symmetric_yolo_gates=True,
    )

    assert session.gate_map[0].name == "weak"
    assert session.gate_map[1].name == "weak"


def test_isolated_worker_survives_multiple_detect_cycles():
    color, depth, _expected = _synth_sideways()
    worker = DetectionWorkerClient(
        model=CAMERA_ROOT / "__missing_ros_model__.pt",
        use_yolo=False,
        weak_camera=0,
    )
    try:
        completed = 0
        deadline = time.monotonic() + 45.0
        while completed < 3 and time.monotonic() < deadline:
            for event in worker.poll():
                assert event.kind != "crash", event.detail
                assert event.kind not in ("error", "startup_error"), event.detail
                if event.kind == "result":
                    assert event.result is not None
                    completed += 1
            if not worker.busy and completed < 3:
                worker.submit(completed, color, depth, color, depth)
            time.sleep(0.05)
        assert completed == 3
    finally:
        worker.close()
