"""Agent B — cap-first FN fallback when YOLO misses / clear-depth gates reject."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import depthcamera as dc
from test_dual_review import _clear_vial_depth_scene


class CapFallbackGateTests(unittest.TestCase):
    def _weak_hole_vial(self):
        image, depth, wall_box, slab_box, vial_box, jar_box = _clear_vial_depth_scene()
        vx, vy, vw, vh = vial_box
        # Fill cup with near depth so strict clear-cup fails (glare / bad angle).
        depth[vy + 10 : vy + vh, vx : vx + vw] = 520.0
        return image, depth, wall_box, slab_box, vial_box, jar_box

    def test_cap_fallback_accepts_white_lid_body_without_clear_depth(self):
        image, depth, _wall, _slab, vial_box, _jar = self._weak_hole_vial()
        self.assertFalse(dc.looks_like_capped_clear_vial(depth, image, vial_box))
        self.assertFalse(
            dc.accepts_vial_box(depth, image, vial_box, peer_confirmed=False, soft_clear=False)
        )
        self.assertTrue(dc.accepts_cap_fallback_box(image, vial_box))
        self.assertTrue(
            dc.accepts_vial_box(
                depth, image, vial_box, peer_confirmed=False, soft_clear=False, cap_fallback=True
            )
        )

    def test_cap_fallback_rejects_wall_and_slab(self):
        image, depth, wall_box, slab_box, _vial, _jar = self._weak_hole_vial()
        self.assertFalse(dc.accepts_cap_fallback_box(image, wall_box))
        self.assertFalse(dc.accepts_cap_fallback_box(image, slab_box))
        self.assertFalse(
            dc.accepts_vial_box(depth, image, wall_box, cap_fallback=True)
        )
        self.assertFalse(
            dc.accepts_vial_box(depth, image, slab_box, cap_fallback=True)
        )


class CapFallbackLocateTests(unittest.TestCase):
    def test_locate_recovers_via_cap_without_peer_or_soft_clear(self):
        """YOLO miss + weak depth holes → detect_vial_by_cap + cap_fallback accepts."""
        image, depth, _wall, _slab, vial_box, _jar = _clear_vial_depth_scene()
        vx, vy, vw, vh = vial_box
        depth[vy + 10 : vy + vh, vx : vx + vw] = 520.0

        detector = dc.VialDetector(None, 0.25)
        cam = dc.OrbbecDepthCamera(
            detector=detector, camera_index=0, gates=dc.strong_gate_profile(), window="T"
        )
        # Pretend YOLO is loaded but returns nothing (classic miss).
        with mock.patch.object(cam.detector, "model", object()), mock.patch.object(
            cam.detector, "detect", return_value=[]
        ), mock.patch.object(dc, "detect_vial_by_cap", return_value=[vial_box]):
            # Strict path (no peer, no soft_clear) must still recover via cap-first.
            fix = cam.locate(image, depth, peer_confirmed=False, soft_clear=False)

        self.assertIsNotNone(fix)
        assert fix is not None
        self.assertGreaterEqual(dc.box_iou(fix.box, vial_box), 0.3)

    def test_locate_retries_yolo_and_still_checks_cap_for_more_vials(self):
        """A recovered YOLO hit must not suppress cap proposals for sibling vials."""
        image, depth, _wall, _slab, vial_box, _jar = _clear_vial_depth_scene()
        vx, vy, vw, vh = vial_box
        depth[vy + 10 : vy + vh, vx : vx + vw] = 520.0

        detector = dc.VialDetector(None, 0.25)
        cam = dc.OrbbecDepthCamera(
            detector=detector, camera_index=0, gates=dc.strong_gate_profile(), window="T"
        )
        with mock.patch.object(cam.detector, "model", object()), mock.patch.object(
            cam.detector, "detect", return_value=[vial_box]
        ), mock.patch.object(dc, "detect_vial_by_cap", return_value=[]) as cap_detect:
            fix = cam.locate(image, depth, peer_confirmed=False, soft_clear=False)

        self.assertIsNotNone(fix)
        cap_detect.assert_called_once()
        assert fix is not None
        self.assertGreaterEqual(dc.box_iou(fix.box, vial_box), 0.3)

    def test_locate_does_not_invent_box_from_peer(self):
        """Empty RGB-D with peer_confirmed still yields no local invent."""
        color = np.full((120, 160, 3), 40, dtype=np.uint8)
        depth = np.full((120, 160), 800.0, dtype=np.float32)
        detector = dc.VialDetector(None, 0.25)
        cam = dc.OrbbecDepthCamera(detector=detector, camera_index=0, window="T")
        with mock.patch.object(cam.detector, "model", object()), mock.patch.object(
            cam.detector, "detect", return_value=[]
        ), mock.patch.object(dc, "detect_vial_by_cap", return_value=[]):
            fix = cam.locate(color, depth, peer_confirmed=True, soft_clear=False)
        self.assertIsNone(fix)

    def test_locate_standalone_still_independent(self):
        detector = dc.VialDetector(None, 0.25)
        cam = dc.OrbbecDepthCamera(detector=detector, camera_index=1, window="B")
        color = np.zeros((60, 80, 3), dtype=np.uint8)
        depth = np.full((60, 80), 500.0, dtype=np.float32)
        with mock.patch.object(cam, "locate", return_value=None) as locate_mock, mock.patch.object(
            dc, "locate_pair", side_effect=AssertionError("no peer invent")
        ):
            dc.locate_standalone(cam, color, depth)
        locate_mock.assert_called_once()
        self.assertFalse(locate_mock.call_args.kwargs.get("peer_confirmed"))


if __name__ == "__main__":
    unittest.main()
