"""Synthetic tests for Gx/Gy rim edge assist + far scale gates (no camera / no 3D)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

import depthcamera as dc
from test_dual_review import _clear_vial_depth_scene, _desk_scene


class EdgeAssistDetectTests(unittest.TestCase):
    def test_detect_vial_by_edges_finds_white_lid_with_rim(self):
        image, _wall, _slab, vial_box = _desk_scene()
        proposals = dc.detect_vial_by_edges(image)
        self.assertTrue(proposals, "expected at least one edge proposal")
        best_iou = max(dc.box_iou(p, vial_box) for p in proposals)
        self.assertGreaterEqual(best_iou, 0.25)

    def test_detect_vial_by_edges_rejects_blank_frame(self):
        blank = np.full((120, 160, 3), 40, dtype=np.uint8)
        self.assertEqual(dc.detect_vial_by_edges(blank), [])

    def test_rim_score_higher_on_edged_lid_than_flat_slab(self):
        image, _wall, slab_box, vial_box = _desk_scene()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx, gy, mag = dc._image_derivatives(gray)
        thr = max(22.0, float(np.percentile(mag, 68)))
        strong = (mag >= thr).astype(np.uint8) * 255
        canny = cv2.Canny(gray, 35, 100)
        edge = cv2.bitwise_or(strong, canny)
        vx, vy, vw, vh = vial_box
        sx, sy, sw, sh = slab_box
        vial_rim = dc._rim_edge_score(edge, vx, vy, vw, vh)
        slab_rim = dc._rim_edge_score(edge, sx, sy, sw, sh)
        self.assertGreater(vial_rim, slab_rim)

    def test_dxdy_structure_prefers_vial_over_slab(self):
        image, _wall, slab_box, vial_box = _desk_scene()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx, gy, mag = dc._image_derivatives(gray)
        vial_score = dc.vial_gradient_match_score(
            image, vial_box, gx=gx, gy=gy, mag=mag, gray=gray
        )
        slab_score = dc.vial_gradient_match_score(
            image, slab_box, gx=gx, gy=gy, mag=mag, gray=gray
        )
        self.assertGreater(vial_score, slab_score)
        self.assertGreater(vial_score, 0.15)

    def test_upright_vial_has_strong_opposing_gx(self):
        image, _wall, _slab, vial_box = _desk_scene()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx, gy, mag = dc._image_derivatives(gray)
        x, y, w, h = vial_box
        # Cup body under the white lid — vertical side walls → high |Gx|.
        cup_y = y + 10
        cup_h = h - 10
        side = max(1, w // 5)
        left = float(np.mean(np.abs(gx[cup_y : cup_y + cup_h, x : x + side])))
        right = float(np.mean(np.abs(gx[cup_y : cup_y + cup_h, x + w - side : x + w])))
        mid = float(
            np.mean(np.abs(gx[cup_y : cup_y + cup_h, x + side : x + w - side]))
        )
        self.assertGreater(min(left, right), mid * 0.8)
        structure = dc._vial_dxdy_structure_score(
            gx, gy, mag, x, y, w, h, top_downish=False, gray=gray
        )
        self.assertGreater(structure, 0.2)


class EdgeAssistLocateTests(unittest.TestCase):
    def test_locate_uses_edge_assist_after_yolo_and_cap_miss(self):
        image, depth, _wall, _slab, vial_box, _jar = _clear_vial_depth_scene()
        detector = dc.VialDetector(None, 0.25)
        cam = dc.OrbbecDepthCamera(
            detector=detector,
            camera_index=0,
            gates=dc.strong_gate_profile(),
            window="T",
            use_edge_assist=True,
        )
        with mock.patch.object(cam.detector, "model", object()), mock.patch.object(
            cam.detector, "detect", return_value=[]
        ), mock.patch.object(dc, "detect_vial_by_cap", return_value=[]), mock.patch.object(
            dc, "detect_vial_by_edges", return_value=[vial_box]
        ) as edge_mock:
            fix = cam.locate(image, depth, peer_confirmed=False, soft_clear=False)

        edge_mock.assert_called_once()
        self.assertIsNotNone(fix)
        assert fix is not None
        self.assertGreaterEqual(dc.box_iou(fix.box, vial_box), 0.3)
        self.assertEqual(cam.last_detect_source, "edges")

    def test_no_edge_assist_skips_detect_vial_by_edges(self):
        image, depth, _wall, _slab, vial_box, _jar = _clear_vial_depth_scene()
        detector = dc.VialDetector(None, 0.25)
        cam = dc.OrbbecDepthCamera(
            detector=detector,
            camera_index=0,
            gates=dc.strong_gate_profile(),
            window="T",
            use_edge_assist=False,
        )
        with mock.patch.object(cam.detector, "model", object()), mock.patch.object(
            cam.detector, "detect", return_value=[]
        ), mock.patch.object(dc, "detect_vial_by_cap", return_value=[]), mock.patch.object(
            dc, "detect_vial_by_edges", side_effect=AssertionError("edge assist off")
        ):
            fix = cam.locate(image, depth, peer_confirmed=False, soft_clear=False)
        self.assertIsNone(fix)

    def test_choose_box_edge_rank_prefers_structured_vial(self):
        image, _wall, slab_box, vial_box = _desk_scene()
        # Force a close tie on cap_run so dx/dy structure is the decider.
        with mock.patch.object(dc, "cap_run", return_value=0.9), mock.patch.object(
            dc, "has_white_cap", side_effect=lambda _img, box, **_k: box in (vial_box, slab_box)
        ):
            chosen = dc.choose_box(
                [slab_box, vial_box],
                previous=None,
                image=image,
                use_edge_rank=True,
            )
        self.assertEqual(chosen, vial_box)

    def test_live_3d_size_score_off_reproject_stack_loadable(self):
        self.assertFalse(dc.LIVE_3D_VIAL_ENABLED)
        self.assertTrue(dc.LIVE_3D_REPROJECT_ASSIST_ALLOWED)
        self.assertFalse(dc.DEFAULT_USE_3D_MODEL_SCORE)
        self.assertIsNone(dc._load_3d_identify_stack())
        # Reproject assist stack may load even while size-score stays off.
        stack = dc._load_3d_assist_stack()
        self.assertIsNotNone(stack)
        cam = dc.OrbbecDepthCamera(
            detector=dc.VialDetector(None, 0.25),
            use_3d_model_score=True,
            window="T",
        )
        self.assertFalse(cam.use_3d_model_score)
        line = dc.format_3d_reproject_status_line(want_assist=True)
        self.assertIn("PLACEHOLDER", line)


class FarDistanceScaleTests(unittest.TestCase):
    """No hard 1.5 m cut — far + tiny vial ok; far + huge wall rejected by scale."""

    def _fake_intrinsics(self, fx=600.0, fy=600.0, cx=160.0, cy=120.0):
        return (fx, fy, cx, cy)

    def test_within_reach_allows_far_surface(self):
        image, depth, _wall, _slab, vial_box, _jar = _clear_vial_depth_scene(vial_z=2800.0)
        # Clear-cup scene still has lid at vial_z; within_reach must not 1.5 m-cut.
        self.assertTrue(dc.within_reach(depth, image, vial_box))
        self.assertGreater(2800.0, dc.MAX_RANGE_MM)

    def test_within_reach_rejects_too_near_junk(self):
        image = np.full((120, 160, 3), 40, dtype=np.uint8)
        depth = np.full((120, 160), 40.0, dtype=np.float32)  # below MIN_SURFACE_MM
        box = (40, 40, 40, 40)
        image[40:50, 40:80] = (245, 245, 245)
        self.assertFalse(dc.within_reach(depth, image, box))

    def test_vial_scale_allows_far_tiny_box(self):
        # At Z=3000 mm, fx=600 → 20×30 px ≈ 100×150 mm (inside vial band).
        image = np.full((240, 320, 3), 90, dtype=np.uint8)
        vx, vy, vw, vh = 150, 100, 20, 30
        image[vy : vy + 8, vx : vx + vw] = (245, 245, 245)
        image[vy + 8 : vy + vh, vx : vx + vw] = (70, 70, 65)
        image[vy + 8 : vy + vh, vx] = (40, 40, 40)
        image[vy + 8 : vy + vh, vx + vw - 1] = (40, 40, 40)
        depth = np.full((240, 320), 3000.0, dtype=np.float32)
        depth[vy : vy + 8, vx : vx + vw] = 3000.0
        box = (vx, vy, vw, vh)
        K = self._fake_intrinsics()
        width_mm = vw * 3000.0 / 600.0
        self.assertLessEqual(width_mm, 110.0)
        self.assertTrue(dc.within_reach(depth, image, box))
        self.assertTrue(dc.vial_scale(depth, image, box, K))

    def test_vial_scale_rejects_far_huge_box(self):
        # Same far Z but a wall-sized box → width_mm way above vial band.
        image = np.full((240, 320, 3), 90, dtype=np.uint8)
        bx, by, bw, bh = 20, 20, 200, 180
        image[by : by + 20, bx : bx + bw] = (245, 245, 245)
        depth = np.full((240, 320), 3000.0, dtype=np.float32)
        box = (bx, by, bw, bh)
        K = self._fake_intrinsics()
        self.assertTrue(dc.within_reach(depth, image, box))
        self.assertFalse(dc.vial_scale(depth, image, box, K))


if __name__ == "__main__":
    unittest.main()
