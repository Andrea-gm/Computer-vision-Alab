"""Regression tests for the promoted sideways-race winner."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import depthcamera as dc
from sideways_race.methods.method_b_rotated_white_cap import _synth_sideways


class SidewaysWinnerTests(unittest.TestCase):
    def test_promoted_wrapper_finds_synthetic_sideways_vial(self):
        image, depth, gt = _synth_sideways()
        boxes = dc.detect_vial_by_sideways_winner(image, depth)
        self.assertTrue(boxes)
        self.assertGreaterEqual(max(dc.box_iou(box, gt) for box in boxes), 0.5)

    def test_accepted_upright_path_skips_sideways_winner(self):
        image = np.full((120, 160, 3), 80, dtype=np.uint8)
        depth = np.full((120, 160), 600.0, dtype=np.float32)
        upright = (60, 30, 30, 42)
        camera = dc.OrbbecDepthCamera(
            detector=dc.VialDetector(None, dc.DEFAULT_YOLO_CONF), window="T"
        )

        with mock.patch.object(camera.detector, "model", object()), mock.patch.object(
            camera.detector, "detect", return_value=[upright]
        ), mock.patch.object(dc, "usable_detection", return_value=True), mock.patch.object(
            dc, "accepts_vial_box", return_value=True
        ), mock.patch.object(dc, "within_reach", return_value=True), mock.patch.object(
            dc, "vial_scale", return_value=True
        ), mock.patch.object(dc, "has_interior_structure", return_value=True), mock.patch.object(
            dc, "fit_vial_shape", side_effect=lambda _image, _depth, box: box
        ), mock.patch.object(
            dc,
            "detect_vial_by_sideways_winner",
            side_effect=AssertionError("sideways fallback must not run"),
        ), mock.patch.object(dc, "measure_cap_mm", return_value=(600.0, None)):
            fix = camera.locate(image, depth)

        self.assertIsNotNone(fix)
        self.assertIsNone(camera.last_detect_source)

    def test_winner_runs_after_yolo_and_cap_miss(self):
        image, depth, _gt = _synth_sideways()
        winner_box = dc.detect_vial_by_sideways_winner(image, depth)[0]
        camera = dc.OrbbecDepthCamera(
            detector=dc.VialDetector(None, dc.DEFAULT_YOLO_CONF), window="T"
        )

        with mock.patch.object(camera.detector, "model", object()), mock.patch.object(
            camera.detector, "detect", return_value=[]
        ), mock.patch.object(dc, "detect_vial_by_cap", return_value=[]), mock.patch.object(
            dc, "detect_vial_by_sideways_winner", return_value=[winner_box]
        ) as winner, mock.patch.object(
            dc, "measure_cap_mm", return_value=(520.0, None)
        ):
            fix = camera.locate(image, depth)

        winner.assert_called_once()
        self.assertIsNotNone(fix)
        self.assertEqual(camera.last_detect_source, "sideways-race-winner")


if __name__ == "__main__":
    unittest.main()
