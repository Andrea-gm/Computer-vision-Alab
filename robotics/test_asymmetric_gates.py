"""Unit tests for asymmetric per-camera gate profiles (BoN strategy #3)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import depthcamera as dc


class AsymmetricGateTests(unittest.TestCase):
    def test_weak_more_permissive_than_strong(self):
        strong = dc.strong_gate_profile(yolo_conf=0.25)
        weak = dc.weak_gate_profile(base_conf=0.25)
        self.assertLess(weak.yolo_conf, strong.yolo_conf)
        self.assertLess(weak.clear_hole_margin, strong.clear_hole_margin)
        self.assertGreater(weak.lid_hole_max, strong.lid_hole_max)
        self.assertLess(weak.cup_hole_min, strong.cup_hole_min)
        self.assertGreater(weak.opaque_cup_near_frac, strong.opaque_cup_near_frac)
        self.assertLess(weak.aspect_min, strong.aspect_min)
        self.assertGreater(weak.aspect_max, strong.aspect_max)
        self.assertLess(weak.interior_edge_min, strong.interior_edge_min)

    def test_build_map_defaults_cam0_weak(self):
        """Default / file: cam0 permissive (weak), cam1 strong/tighter."""
        dc.load_camera_gates_file(reload=True)
        gate_map = dc.build_camera_gate_map(base_conf=dc.DEFAULT_YOLO_CONF)
        self.assertEqual(gate_map[0].name, "weak")
        self.assertEqual(gate_map[1].name, "strong")
        self.assertAlmostEqual(gate_map[0].yolo_conf, 0.05)
        self.assertAlmostEqual(gate_map[1].yolo_conf, dc.DEFAULT_YOLO_CONF)
        line = dc.format_gate_assignment_line(gate_map)
        self.assertIn("weak=cam0", line)
        self.assertIn("strong=cam1", line)

    def test_build_map_override_cam1_weak(self):
        """--weak-camera 1 still assigns permissive gates to cam1."""
        gate_map = dc.build_camera_gate_map(weak_camera=1, base_conf=dc.DEFAULT_YOLO_CONF)
        self.assertEqual(gate_map[0].name, "strong")
        self.assertEqual(gate_map[1].name, "weak")
        self.assertAlmostEqual(gate_map[0].yolo_conf, dc.DEFAULT_YOLO_CONF)
        self.assertAlmostEqual(gate_map[1].yolo_conf, 0.05)

    def test_build_map_explicit_base_025_still_works(self):
        gate_map = dc.build_camera_gate_map(weak_camera=0, base_conf=0.25)
        self.assertAlmostEqual(gate_map[0].yolo_conf, 0.05)
        self.assertAlmostEqual(gate_map[1].yolo_conf, 0.25)

    def test_explicit_conf_overrides(self):
        gate_map = dc.build_camera_gate_map(
            weak_camera=0, base_conf=0.25, cam0_conf=0.12, cam1_conf=0.30
        )
        self.assertAlmostEqual(gate_map[0].yolo_conf, 0.12)
        self.assertAlmostEqual(gate_map[1].yolo_conf, 0.30)
        self.assertEqual(gate_map[0].name, "weak")
        self.assertEqual(gate_map[1].name, "strong")

    def test_weak_aspect_accepts_borderline_box(self):
        # Below strong aspect_min from camera_gates (~0.95); weak allows ~0.90.
        box = (10, 10, 40, 36)  # h/w = 0.90
        strong = dc.strong_gate_profile()
        weak = dc.weak_gate_profile()
        self.assertFalse(strong.is_full_vial_aspect(box))
        self.assertTrue(weak.is_full_vial_aspect(box))

    def test_opaque_wall_still_rejected_on_weak(self):
        """Blank wall / opaque slab: no white cap → both profiles reject."""
        color = np.full((120, 160, 3), 40, dtype=np.uint8)
        depth = np.full((120, 160), 800.0, dtype=np.float32)
        box = (40, 30, 40, 55)
        weak = dc.weak_gate_profile()
        self.assertFalse(dc.accepts_vial_box(depth, color, box, gates=weak))
        self.assertFalse(
            dc.accepts_vial_box(depth, color, box, peer_confirmed=True, gates=weak)
        )

    def test_locate_uses_per_camera_conf(self):
        detector = dc.VialDetector(None, 0.25)
        weak = dc.weak_gate_profile(yolo_conf=0.12)
        cam = dc.OrbbecDepthCamera(
            detector=detector, camera_index=1, confidence=0.12, gates=weak, window="T"
        )
        color = np.zeros((80, 100, 3), dtype=np.uint8)
        depth = np.full((80, 100), 500.0, dtype=np.float32)
        # Inject a fake YOLO model path via spy on detect conf kwarg.
        seen: list[float | None] = []

        def fake_detect(image, conf=None):
            seen.append(conf)
            return []

        with mock.patch.object(cam.detector, "model", object()), mock.patch.object(
            cam.detector, "detect", side_effect=fake_detect
        ):
            cam.locate(color, depth, peer_confirmed=False)
        # Full-frame discovery intentionally asks YOLO for slightly weaker sibling
        # proposals; the camera-specific profile still controls the base threshold.
        self.assertEqual(
            seen,
            [
                max(
                    dc.MIN_DISCOVERY_YOLO_CONF,
                    weak.yolo_conf - dc.DISCOVERY_YOLO_CONF_DELTA,
                )
            ],
        )

    def test_locate_pair_still_syncs_with_asymmetric_gates(self):
        detector = dc.VialDetector(None, 0.25)
        cam_a = dc.OrbbecDepthCamera(
            detector=detector,
            camera_index=0,
            gates=dc.strong_gate_profile(),
            window="A",
        )
        cam_b = dc.OrbbecDepthCamera(
            detector=detector,
            camera_index=1,
            gates=dc.weak_gate_profile(),
            window="B",
        )
        fix_a = dc.VialFix((10, 10, 30, 40), 400.0, 0.0, 0.0, 400.0)
        soft_b = dc.VialFix((12, 12, 28, 40), 410.0, 0.0, 0.0, 410.0)
        frame = (
            np.zeros((60, 80, 3), dtype=np.uint8),
            np.full((60, 80), 500.0, dtype=np.float32),
        )

        def locate_a(color, depth_mm, *, peer_confirmed=False):
            return fix_a

        def locate_b(color, depth_mm, *, peer_confirmed=False):
            return soft_b if peer_confirmed else None

        with mock.patch.object(cam_a, "locate", side_effect=locate_a), mock.patch.object(
            cam_b, "locate", side_effect=locate_b
        ):
            pair = dc.locate_pair(cam_a, frame, cam_b, frame)
        self.assertIs(pair.fix_a, fix_a)
        self.assertIs(pair.fix_b, soft_b)
        self.assertTrue(pair.peer_assist)
        # Distances stay independent (not averaged).
        self.assertNotEqual(pair.fix_a.distance_mm, pair.fix_b.distance_mm)

    def test_weak_profile_reads_camera_gates_json(self):
        dc.load_camera_gates_file(reload=True)
        data = dc.load_camera_gates_file()
        weak = dc.weak_gate_profile(base_conf=dc.DEFAULT_YOLO_CONF)
        self.assertAlmostEqual(weak.yolo_conf, float(data["cam1_weak"]["yolo_conf"]))
        self.assertAlmostEqual(weak.yolo_conf, 0.05)
        self.assertAlmostEqual(weak.aspect_min, float(data["cam1_weak"]["aspect_min"]))

    def test_has_white_cap_threads_weak_gates(self):
        """Regression: strong usable_detection defaults must not zero weak-cam boxes."""
        weak = dc.weak_gate_profile()
        color = np.zeros((100, 120, 3), dtype=np.uint8)
        box = (10, 10, 40, 38)  # aspect 0.95 — rejected by strong, allowed by weak
        with mock.patch.object(dc, "usable_detection", return_value=False) as ud:
            self.assertFalse(dc.has_white_cap(color, box, gates=weak))
        self.assertTrue(ud.called)
        _, kwargs = ud.call_args
        self.assertIs(kwargs.get("gates"), weak)

    def test_empty_view_message_honesty(self):
        self.assertEqual(dc.empty_view_message(None), "No vial in view")
        self.assertEqual(dc.empty_view_message("no_vial"), "No vial in view")
        self.assertEqual(dc.empty_view_message("low_confidence"), "Low confidence")

    def test_locate_miss_reason_low_confidence_when_candidates(self):
        detector = dc.VialDetector(None, 0.05)
        weak = dc.weak_gate_profile(yolo_conf=0.05)
        cam = dc.OrbbecDepthCamera(
            detector=detector, camera_index=1, confidence=0.05, gates=weak, window="T"
        )
        color = np.zeros((80, 100, 3), dtype=np.uint8)
        depth = np.full((80, 100), 500.0, dtype=np.float32)
        candidate = (20, 15, 30, 45)

        def fake_detect(image, conf=None):
            return [candidate]

        with mock.patch.object(cam.detector, "model", object()), mock.patch.object(
            cam.detector, "detect", side_effect=fake_detect
        ), mock.patch.object(dc, "usable_detection", return_value=False), mock.patch.object(
            dc, "detect_vial_by_cap", return_value=[]
        ):
            fix = cam.locate(color, depth, peer_confirmed=False)
        self.assertIsNone(fix)
        self.assertEqual(cam.miss_reason, "low_confidence")

    def test_locate_miss_reason_no_vial_when_empty(self):
        detector = dc.VialDetector(None, 0.05)
        weak = dc.weak_gate_profile(yolo_conf=0.05)
        cam = dc.OrbbecDepthCamera(
            detector=detector, camera_index=1, confidence=0.05, gates=weak, window="T"
        )
        color = np.zeros((80, 100, 3), dtype=np.uint8)
        depth = np.full((80, 100), 500.0, dtype=np.float32)

        with mock.patch.object(cam.detector, "model", object()), mock.patch.object(
            cam.detector, "detect", return_value=[]
        ), mock.patch.object(dc, "detect_vial_by_cap", return_value=[]):
            fix = cam.locate(color, depth, peer_confirmed=False)
        self.assertIsNone(fix)
        self.assertEqual(cam.miss_reason, "no_vial")


if __name__ == "__main__":
    unittest.main()
