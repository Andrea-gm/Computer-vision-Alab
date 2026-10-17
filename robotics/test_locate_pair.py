"""Tests for synchronized dual locate_pair (BoN #6)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

# Ensure imports resolve from this worktree
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import depthcamera as dc


def _fix(box=(10, 10, 30, 40), distance_mm: float | None = 400.0) -> dc.VialFix:
    return dc.VialFix(box=box, distance_mm=distance_mm, x_mm=0.0, y_mm=0.0, z_mm=distance_mm)


def _frame(w: int = 80, h: int = 60):
    color = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), 500.0, dtype=np.float32)
    return color, depth


class LocatePairSyncTests(unittest.TestCase):
    def setUp(self):
        detector = dc.VialDetector(None, 0.25)
        self.cam_a = dc.OrbbecDepthCamera(detector=detector, camera_index=0, window="A")
        self.cam_b = dc.OrbbecDepthCamera(detector=detector, camera_index=1, window="B")
        self.frame_a = _frame()
        self.frame_b = _frame()
        self.fix_a = _fix(box=(10, 10, 30, 40))
        self.fix_b = _fix(box=(20, 12, 32, 42))
        self.soft_b = _fix(box=(22, 14, 30, 40), distance_mm=410.0)

    def test_returns_both_fixes_in_one_result(self):
        """UI gets a paired PairLocateResult — never one cam alone mid-tick."""
        calls: list[tuple[int, bool]] = []

        def locate_a(color, depth_mm, *, peer_confirmed=False):
            calls.append((0, peer_confirmed))
            return self.fix_a if not peer_confirmed else self.fix_a

        def locate_b(color, depth_mm, *, peer_confirmed=False):
            calls.append((1, peer_confirmed))
            return self.fix_b if not peer_confirmed else self.fix_b

        with mock.patch.object(self.cam_a, "locate", side_effect=locate_a), mock.patch.object(
            self.cam_b, "locate", side_effect=locate_b
        ):
            pair = dc.locate_pair(self.cam_a, self.frame_a, self.cam_b, self.frame_b)

        self.assertIsInstance(pair, dc.PairLocateResult)
        self.assertIs(pair.fix_a, self.fix_a)
        self.assertIs(pair.fix_b, self.fix_b)
        self.assertTrue(pair.peer_assist)
        # Strict only — soft not needed when both hit.
        self.assertEqual(calls, [(0, False), (1, False)])

    def test_peer_soft_applied_inside_before_return(self):
        """When A is strict and B misses, soft retry for B happens before return."""
        mem = dc.PeerMemory(window=15)
        order: list[str] = []

        def locate_a(color, depth_mm, *, peer_confirmed=False):
            order.append(f"a:strict={not peer_confirmed}")
            return self.fix_a

        def locate_b(color, depth_mm, *, peer_confirmed=False):
            order.append(f"b:soft={peer_confirmed}")
            if peer_confirmed:
                return self.soft_b
            return None

        with mock.patch.object(self.cam_a, "locate", side_effect=locate_a), mock.patch.object(
            self.cam_b, "locate", side_effect=locate_b
        ):
            pair = dc.locate_pair(
                self.cam_a,
                self.frame_a,
                self.cam_b,
                self.frame_b,
                peer_memory=mem,
            )

        self.assertEqual(order, ["a:strict=True", "b:soft=False", "b:soft=True"])
        self.assertIs(pair.fix_a, self.fix_a)
        self.assertIs(pair.fix_b, self.soft_b)
        self.assertTrue(pair.peer_assist)
        self.assertTrue(mem.active)

    def test_sticky_memory_soft_on_peer_blink_same_tick(self):
        """After a prior strict hit, soft assist warms even if both miss strict this tick."""
        mem = dc.PeerMemory(window=15)
        mem.observe(True)  # prior tick had a clear-vial confirm

        def locate_miss(color, depth_mm, *, peer_confirmed=False):
            if peer_confirmed:
                return self.soft_b
            return None

        with mock.patch.object(self.cam_a, "locate", side_effect=locate_miss), mock.patch.object(
            self.cam_b, "locate", side_effect=locate_miss
        ):
            pair = dc.locate_pair(
                self.cam_a,
                self.frame_a,
                self.cam_b,
                self.frame_b,
                peer_memory=mem,
            )

        self.assertIs(pair.fix_a, self.soft_b)
        self.assertIs(pair.fix_b, self.soft_b)
        self.assertTrue(pair.peer_assist)

    def test_missing_frame_still_returns_pair(self):
        """One camera waiting → still a PairLocateResult for the same tick."""
        with mock.patch.object(self.cam_a, "locate", return_value=self.fix_a) as la, mock.patch.object(
            self.cam_b, "locate"
        ) as lb:
            pair = dc.locate_pair(self.cam_a, self.frame_a, self.cam_b, None)

        self.assertIs(pair.fix_a, self.fix_a)
        self.assertIsNone(pair.fix_b)
        la.assert_called_once()
        lb.assert_not_called()

    def test_no_soft_without_peer_memory_warm(self):
        """Cold memory + both miss strict → no soft retry, paired Nones."""
        mem = dc.PeerMemory(window=15)
        calls: list[bool] = []

        def locate_miss(color, depth_mm, *, peer_confirmed=False):
            calls.append(peer_confirmed)
            return None

        with mock.patch.object(self.cam_a, "locate", side_effect=locate_miss), mock.patch.object(
            self.cam_b, "locate", side_effect=locate_miss
        ):
            pair = dc.locate_pair(
                self.cam_a,
                self.frame_a,
                self.cam_b,
                self.frame_b,
                peer_memory=mem,
            )

        self.assertIsNone(pair.fix_a)
        self.assertIsNone(pair.fix_b)
        self.assertFalse(pair.peer_assist)
        self.assertEqual(calls, [False, False])  # strict only


class MultiVialPreviewTests(unittest.TestCase):
    def test_nms_merges_duplicates_and_keeps_distinct_vials(self):
        boxes = [
            (10, 10, 20, 40),
            (11, 11, 20, 40),  # duplicate of first
            (80, 10, 20, 40),
            (140, 10, 20, 40),
        ]
        self.assertEqual(
            dc.nms_vial_boxes(boxes),
            [(10, 10, 20, 40), (80, 10, 20, 40), (140, 10, 20, 40)],
        )

    def test_nms_caps_scene_at_eight_vials(self):
        boxes = [(i * 30, 5, 12, 24) for i in range(12)]
        self.assertEqual(len(dc.nms_vial_boxes(boxes)), dc.MAX_VIALS_PER_CAMERA)

    def test_vial_labels_hidden_by_default(self):
        image = np.zeros((120, 220, 3), dtype=np.uint8)
        fixes = [_fix((10, 10, 20, 40)), _fix((100, 10, 20, 40))]
        with mock.patch.object(dc, "draw_label") as draw:
            dc.annotate(image, fixes[0], fixes=fixes)
        vial_labels = [
            call for call in draw.call_args_list if call.args[2] and call.args[2][0] == "vial"
        ]
        self.assertEqual(len(vial_labels), 0)

    def test_preview_labels_every_vial_when_enabled(self):
        image = np.zeros((120, 220, 3), dtype=np.uint8)
        fixes = [_fix((10, 10, 20, 40)), _fix((100, 10, 20, 40))]
        with mock.patch.object(dc, "draw_label") as draw:
            dc.annotate(image, fixes[0], fixes=fixes, show_vial_labels=True)
        vial_labels = [
            call for call in draw.call_args_list if call.args[2] and call.args[2][0] == "vial"
        ]
        self.assertEqual(len(vial_labels), 2)


if __name__ == "__main__":
    unittest.main()
