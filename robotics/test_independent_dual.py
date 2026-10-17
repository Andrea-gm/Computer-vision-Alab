"""Tests for independent dual IPC parse/corroboration (no cameras required)."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import depthcamera as dc


def _fix(box=(10, 10, 30, 40), distance_mm: float | None = 400.0) -> dc.VialFix:
    return dc.VialFix(box=box, distance_mm=distance_mm, x_mm=0.0, y_mm=0.0, z_mm=distance_mm)


class PeerReportParseTests(unittest.TestCase):
    def test_roundtrip_payload(self):
        report = dc.PeerReport(
            timestamp=123.5,
            camera_id=0,
            has_vial=True,
            box=(1, 2, 3, 4),
            distance_cm=41.2,
            conf=0.25,
        )
        parsed = dc.parse_peer_report(dc.encode_peer_report(report))
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.camera_id, 0)
        self.assertTrue(parsed.has_vial)
        self.assertEqual(parsed.box, (1, 2, 3, 4))
        self.assertAlmostEqual(parsed.distance_cm, 41.2)
        self.assertAlmostEqual(parsed.conf, 0.25)

    def test_parse_dict_and_reject_garbage(self):
        ok = dc.parse_peer_report(
            {
                "timestamp": 1.0,
                "camera_id": 1,
                "has_vial": False,
                "box": None,
                "distance_cm": None,
                "conf": None,
            }
        )
        self.assertIsNotNone(ok)
        assert ok is not None
        self.assertFalse(ok.has_vial)
        self.assertIsNone(dc.parse_peer_report(None))
        self.assertIsNone(dc.parse_peer_report(b""))
        self.assertIsNone(dc.parse_peer_report(b"{not json"))
        self.assertIsNone(dc.parse_peer_report({"camera_id": 0}))  # missing has_vial


class CorroborationTests(unittest.TestCase):
    def _local(self, cm: float | None = 40.0, has: bool = True) -> dc.PeerReport:
        return dc.PeerReport(
            timestamp=time.time(),
            camera_id=0,
            has_vial=has,
            box=(10, 10, 20, 30) if has else None,
            distance_cm=cm if has else None,
            conf=0.25 if has else None,
        )

    def _peer(self, cm: float | None = 40.5, has: bool = True, age: float = 0.0) -> dc.PeerReport:
        return dc.PeerReport(
            timestamp=time.time() - age,
            camera_id=1,
            has_vial=has,
            box=(12, 12, 20, 30) if has else None,
            distance_cm=cm if has else None,
            conf=0.05 if has else None,
        )

    def test_agree_within_abs(self):
        result = dc.corroborate_distances(self._local(40.0), self._peer(42.0), abs_cm=3.0, rel=0.01)
        self.assertEqual(result.status, "agree")
        self.assertAlmostEqual(result.local_cm, 40.0)
        self.assertAlmostEqual(result.peer_cm, 42.0)

    def test_agree_within_rel_even_if_abs_misses(self):
        # |Δ|=8 on ~80cm → 10% relative; abs threshold 3 fails, rel 0.15 passes.
        result = dc.corroborate_distances(self._local(80.0), self._peer(88.0), abs_cm=3.0, rel=0.15)
        self.assertEqual(result.status, "agree")

    def test_disagree(self):
        result = dc.corroborate_distances(self._local(40.0), self._peer(55.0), abs_cm=3.0, rel=0.1)
        self.assertEqual(result.status, "disagree")
        self.assertIn("disagree", result.overlay_line)

    def test_missing_peer(self):
        result = dc.corroborate_distances(self._local(), None)
        self.assertEqual(result.status, "missing_peer")
        self.assertEqual(result.label, "no peer")
        # Local distance unchanged / still reported.
        self.assertAlmostEqual(result.local_cm, 40.0)

    def test_stale_peer_treated_as_missing(self):
        result = dc.corroborate_distances(
            self._local(), self._peer(age=5.0), stale_s=1.0, now=time.time()
        )
        self.assertEqual(result.status, "missing_peer")

    def test_peer_no_vial(self):
        result = dc.corroborate_distances(self._local(), self._peer(has=False))
        self.assertEqual(result.status, "peer_no_vial")

    def test_local_no_vial_still_notes_peer(self):
        result = dc.corroborate_distances(self._local(has=False), self._peer(41.0))
        self.assertEqual(result.status, "local_no_vial")
        self.assertAlmostEqual(result.peer_cm, 41.0)

    def test_never_averages_distances(self):
        result = dc.corroborate_distances(self._local(40.0), self._peer(50.0), abs_cm=3.0, rel=0.05)
        self.assertEqual(result.status, "disagree")
        self.assertAlmostEqual(result.local_cm, 40.0)
        self.assertAlmostEqual(result.peer_cm, 50.0)
        mid = (40.0 + 50.0) / 2.0
        self.assertNotAlmostEqual(result.local_cm, mid)


class LocateStandaloneTests(unittest.TestCase):
    def test_never_calls_peer_soft_locate(self):
        detector = dc.VialDetector(None, 0.25)
        cam = dc.OrbbecDepthCamera(detector=detector, camera_index=0, window="A")
        color = np.zeros((60, 80, 3), dtype=np.uint8)
        depth = np.full((60, 80), 500.0, dtype=np.float32)
        calls: list[tuple[bool, bool]] = []

        def locate(color, depth_mm, *, peer_confirmed=False, soft_clear=False):
            calls.append((peer_confirmed, soft_clear))
            return _fix()

        with mock.patch.object(cam, "locate", side_effect=locate):
            fix = dc.locate_standalone(cam, color, depth)

        self.assertIsNotNone(fix)
        # No sticky peer assist; local soft-clear on for independent FN relief.
        self.assertEqual(calls, [(False, True)])

    def test_independent_path_does_not_use_locate_pair(self):
        """Worker loop uses locate_standalone; locate_pair must not be required."""
        detector = dc.VialDetector(None, 0.25)
        cam = dc.OrbbecDepthCamera(detector=detector, camera_index=0, window="A")
        color = np.zeros((60, 80, 3), dtype=np.uint8)
        depth = np.full((60, 80), 500.0, dtype=np.float32)

        with mock.patch.object(cam, "locate", return_value=_fix()) as locate_mock, mock.patch.object(
            dc, "locate_pair", side_effect=AssertionError("locate_pair must not run")
        ):
            fix = dc.locate_standalone(cam, color, depth)
            report = dc.peer_report_from_fix(0, fix, conf=0.25)
            corr = dc.corroborate_distances(report, None)

        locate_mock.assert_called_once()
        self.assertEqual(locate_mock.call_args.kwargs.get("peer_confirmed"), False)
        self.assertTrue(locate_mock.call_args.kwargs.get("soft_clear"))
        self.assertEqual(corr.status, "missing_peer")
        self.assertTrue(report.has_vial)
        self.assertAlmostEqual(report.distance_cm, 40.0)

    def test_two_boxes_survive_independent_locate_and_annotate(self):
        """One strict + one fallback hit must both reach the worker preview."""
        first = (10, 10, 20, 40)
        second = (80, 10, 20, 40)
        detector = dc.VialDetector(None, 0.25)
        detector.model = object()
        cam = dc.OrbbecDepthCamera(detector=detector, camera_index=0, window="A")
        color = np.zeros((100, 140, 3), dtype=np.uint8)
        depth = np.full((100, 140), 500.0, dtype=np.float32)

        def accepts(_depth, _color, box, **kwargs):
            return box == first or (box == second and kwargs.get("cap_fallback", False))

        with mock.patch.object(detector, "detect", return_value=[first, second]), mock.patch.object(
            dc, "detect_vial_by_cap", return_value=[]
        ), mock.patch.object(dc, "usable_detection", return_value=True), mock.patch.object(
            dc, "within_reach", return_value=True
        ), mock.patch.object(dc, "vial_scale", return_value=True), mock.patch.object(
            dc, "accepts_vial_box", side_effect=accepts
        # Strict YOLO boxes have already passed the clear-vial gate. A historical
        # second global structure veto collapsed real multi-vial frames.
        ), mock.patch.object(dc, "has_interior_structure", return_value=False), mock.patch.object(
            dc, "fit_vial_shape", side_effect=lambda _color, _depth, box: box
        ), mock.patch.object(
            dc, "measure_cap_mm", side_effect=lambda _depth, _color, box: (500.0, box)
        ):
            primary = dc.locate_standalone(cam, color, depth)

        self.assertIsNotNone(primary)
        self.assertEqual({fix.box for fix in cam.fixes}, {first, second})
        with mock.patch("builtins.print") as output:
            self.assertTrue(dc.log_found_vials(cam.camera_index, cam.fixes))
        output.assert_called_once_with(
            "[cam0] FOUND 2 vials ids=1,2",
            flush=True,
        )
        with mock.patch.object(dc, "draw_label") as draw:
            dc.annotate(color, primary, fixes=cam.fixes, camera_label="camera 0")
        vial_labels = [
            call for call in draw.call_args_list if call.args[2] and call.args[2][0] == "vial"
        ]
        self.assertEqual(len(vial_labels), 2)

    def test_two_tracked_rois_survive_without_full_frame_predict(self):
        """Stable independent worker frame uses both ROI predictions and draws both."""
        first = (10, 10, 20, 40)
        second = (80, 10, 20, 40)
        detector = dc.VialDetector(None, 0.25)
        detector.model = object()
        cam = dc.OrbbecDepthCamera(detector=detector, camera_index=1, window="B")
        cam._tracked_boxes = [first, second]
        color = np.zeros((100, 140, 3), dtype=np.uint8)
        depth = np.full((100, 140), 500.0, dtype=np.float32)

        with mock.patch.object(
            detector, "detect_in_rois", return_value=[[first], [second]]
        ), mock.patch.object(detector, "detect", side_effect=AssertionError("full scan")):
            with mock.patch.object(dc, "usable_detection", return_value=True), mock.patch.object(
                dc, "within_reach", return_value=True
            ), mock.patch.object(dc, "vial_scale", return_value=True), mock.patch.object(
                dc, "accepts_vial_box", return_value=True
            ), mock.patch.object(dc, "detect_vial_by_cap", return_value=[]), mock.patch.object(
                dc, "fit_vial_shape", side_effect=lambda _color, _depth, box: box
            ), mock.patch.object(
                dc, "measure_cap_mm", side_effect=lambda _depth, _color, box: (500.0, box)
            ):
                primary = dc.locate_standalone(cam, color, depth)

        self.assertIsNotNone(primary)
        self.assertEqual({fix.box for fix in cam.fixes}, {first, second})
        with mock.patch.object(dc, "draw_label") as draw:
            dc.annotate(color, primary, fixes=cam.fixes, camera_label="camera 1")
        self.assertEqual(
            sum(
                1
                for call in draw.call_args_list
                if call.args[2] and call.args[2][0] == "vial"
            ),
            2,
        )

    def test_two_tracks_keep_ids_and_update_both_rois_across_frames(self):
        """Both vials own persistent identities and independently moving ROIs."""
        first = (10, 10, 20, 40)
        second = (80, 10, 20, 40)
        moved_first = (14, 12, 20, 40)
        moved_second = (76, 14, 20, 40)
        detector = dc.VialDetector(None, 0.25)
        detector.model = object()
        cam = dc.OrbbecDepthCamera(detector=detector, camera_index=0, window="A")
        color = np.zeros((100, 140, 3), dtype=np.uint8)
        depth = np.full((100, 140), 500.0, dtype=np.float32)

        gate_patches = (
            mock.patch.object(dc, "usable_detection", return_value=True),
            mock.patch.object(dc, "within_reach", return_value=True),
            mock.patch.object(dc, "vial_scale", return_value=True),
            mock.patch.object(dc, "accepts_vial_box", return_value=True),
            mock.patch.object(dc, "detect_vial_by_cap", return_value=[]),
            mock.patch.object(dc, "fit_vial_shape", side_effect=lambda _c, _d, box: box),
            mock.patch.object(
                dc, "measure_cap_mm", side_effect=lambda _d, _c, box: (500.0, box)
            ),
        )
        with mock.patch.object(detector, "detect_in_rois", side_effect=[[], [[moved_first], [moved_second]]]) as roi, mock.patch.object(
            detector, "detect", return_value=[first, second]
        ) as full:
            with gate_patches[0], gate_patches[1], gate_patches[2], gate_patches[3], gate_patches[4], gate_patches[5], gate_patches[6]:
                dc.locate_standalone(cam, color, depth)
                first_ids = {fix.track_id for fix in cam.fixes}
                first_rois = list(cam._tracked_boxes)
                dc.locate_standalone(cam, color, depth)

        self.assertEqual(len(first_ids), 2)
        self.assertNotIn(None, first_ids)
        self.assertEqual({fix.track_id for fix in cam.fixes}, first_ids)
        self.assertEqual(len(cam.tracks), 2)
        self.assertEqual(len(cam._tracked_boxes), 2)
        self.assertNotEqual(cam._tracked_boxes[0], first_rois[0])
        self.assertNotEqual(cam._tracked_boxes[1], first_rois[1])
        self.assertEqual(roi.call_args_list[1].args[1], first_rois)
        full.assert_called_once()

    def test_periodic_full_search_adds_second_track_without_replacing_first(self):
        first = (10, 10, 20, 40)
        second = (80, 10, 20, 40)
        detector = dc.VialDetector(None, 0.25)
        detector.model = object()
        cam = dc.OrbbecDepthCamera(detector=detector, camera_index=0, window="A")
        color = np.zeros((100, 140, 3), dtype=np.uint8)
        depth = np.full((100, 140), 500.0, dtype=np.float32)

        with mock.patch.object(
            detector, "detect_in_rois", side_effect=[[], [[first]]]
        ), mock.patch.object(
            detector, "detect", side_effect=[[first], [first, second]]
        ) as full, mock.patch.object(
            dc, "usable_detection", return_value=True
        ), mock.patch.object(
            dc, "within_reach", return_value=True
        ), mock.patch.object(
            dc, "vial_scale", return_value=True
        ), mock.patch.object(
            dc, "accepts_vial_box", return_value=True
        ), mock.patch.object(
            dc, "detect_vial_by_cap", return_value=[]
        ), mock.patch.object(
            dc, "fit_vial_shape", side_effect=lambda _c, _d, box: box
        ), mock.patch.object(
            dc, "measure_cap_mm", side_effect=lambda _d, _c, box: (500.0, box)
        ):
            dc.locate_standalone(cam, color, depth)
            original_id = cam.fixes[0].track_id
            cam._roi_frame = dc.ROI_FULL_SEARCH_INTERVAL - 1
            dc.locate_standalone(cam, color, depth)

        self.assertEqual(full.call_count, 2)
        expected_conf = max(
            dc.MIN_DISCOVERY_YOLO_CONF,
            cam.gates.yolo_conf - dc.DISCOVERY_YOLO_CONF_DELTA,
        )
        self.assertAlmostEqual(full.call_args_list[1].kwargs["conf"], expected_conf)
        self.assertEqual(len(cam.tracks), 2)
        self.assertIn(original_id, {fix.track_id for fix in cam.fixes})
        self.assertEqual({fix.box for fix in cam.fixes}, {first, second})

    def test_roi_miss_runs_full_search_and_reacquires_both(self):
        first = (10, 10, 20, 40)
        second = (80, 10, 20, 40)
        detector = dc.VialDetector(None, 0.25)
        detector.model = object()
        cam = dc.OrbbecDepthCamera(detector=detector, camera_index=0, window="A")
        cam._tracked_boxes = [first, second]
        color = np.zeros((100, 140, 3), dtype=np.uint8)
        depth = np.full((100, 140), 500.0, dtype=np.float32)

        with mock.patch.object(
            detector, "detect_in_rois", return_value=[[], [second]]
        ), mock.patch.object(detector, "detect", return_value=[first, second]) as full:
            with mock.patch.object(dc, "usable_detection", return_value=True), mock.patch.object(
                dc, "within_reach", return_value=True
            ), mock.patch.object(dc, "vial_scale", return_value=True), mock.patch.object(
                dc, "accepts_vial_box", return_value=True
            ), mock.patch.object(dc, "detect_vial_by_cap", return_value=[]), mock.patch.object(
                dc, "fit_vial_shape", side_effect=lambda _color, _depth, box: box
            ), mock.patch.object(
                dc, "measure_cap_mm", side_effect=lambda _depth, _color, box: (500.0, box)
            ):
                primary = dc.locate_standalone(cam, color, depth)

        self.assertIsNotNone(primary)
        full.assert_called_once()
        self.assertEqual({fix.box for fix in cam.fixes}, {first, second})

    def test_yolo_predict_preserves_multi_proposals_before_local_gates(self):
        detector = dc.VialDetector(None, 0.25)
        detector.model = mock.Mock()
        detector.model.predict.return_value = []
        image = np.zeros((100, 140, 3), dtype=np.uint8)

        self.assertEqual(detector.detect(image, conf=0.12), [])

        kwargs = detector.model.predict.call_args.kwargs
        self.assertGreaterEqual(kwargs["max_det"], 8)
        self.assertGreater(kwargs["iou"], dc.VIAL_NMS_IOU)
        self.assertAlmostEqual(kwargs["conf"], 0.12)

    def test_saved_two_vial_scene_produces_two_tracks(self):
        """Regression for cam0 run where only the right of two vials was boxed."""
        scene = (
            Path(__file__).resolve().parent
            / "vial_dataset"
            / "review"
            / "pending"
            / "cam0_20261012_185945_0010.jpg"
        )
        if not scene.exists() or not dc.LIVE_WEIGHTS.exists():
            self.skipTest("saved two-vial scene or live weights unavailable")
        image = dc.cv2.imread(str(scene))
        self.assertIsNotNone(image)
        depth = np.full(image.shape[:2], 500.0, dtype=np.float32)
        detector = dc.VialDetector(dc.LIVE_WEIGHTS, 0.05)

        raw = detector.detect(image, conf=0.05)
        self.assertGreaterEqual(len(dc.nms_vial_boxes(raw)), 2)

        cam = dc.OrbbecDepthCamera(
            detector=detector,
            camera_index=0,
            window="two-vial-regression",
            gates=dc.weak_gate_profile(),
        )
        primary = dc.locate_standalone(cam, image, depth)
        self.assertIsNotNone(primary)
        self.assertGreaterEqual(len(cam.fixes), 2)
        centers = sorted(fix.center[0] for fix in cam.fixes)
        self.assertLess(centers[0], 300)
        self.assertGreater(centers[1], 300)

    def test_fit_never_escapes_yolo_vertical_span(self):
        image = np.zeros((180, 180, 3), dtype=np.uint8)
        image[70:82, 65:115] = 255
        dc.cv2.rectangle(image, (68, 82), (112, 145), (180, 180, 180), 2)
        # Distractor edges above the proposal used to pull the fitted box upward.
        dc.cv2.rectangle(image, (65, 10), (115, 60), (255, 255, 255), 2)
        depth = np.full(image.shape[:2], 500.0, dtype=np.float32)
        proposal = (55, 65, 70, 90)

        fitted = dc.fit_vial_shape(image, depth, proposal)

        self.assertGreaterEqual(fitted[1], proposal[1])
        self.assertLessEqual(fitted[1] + fitted[3], proposal[1] + proposal[3])


class SoftClearIndependentTests(unittest.TestCase):
    """False-negative relief: white-cap vial with weak depth holes still accepted."""

    def _weak_hole_vial(self):
        # Reuse desk scene helpers from test_dual_review via local synthetic.
        from test_dual_review import _clear_vial_depth_scene

        image, depth, wall_box, slab_box, vial_box, jar_box = _clear_vial_depth_scene()
        vx, vy, vw, vh = vial_box
        # Fill cup with near depth so strict clear-cup fails (Orbbec fill-in).
        depth[vy + 10 : vy + vh, vx : vx + vw] = 520.0
        return image, depth, wall_box, slab_box, vial_box, jar_box

    def test_soft_clear_recovers_weak_depth_holes(self):
        image, depth, _wall, _slab, vial_box, _jar = self._weak_hole_vial()
        self.assertFalse(dc.looks_like_capped_clear_vial(depth, image, vial_box))
        self.assertFalse(
            dc.accepts_vial_box(depth, image, vial_box, peer_confirmed=False, soft_clear=False)
        )
        self.assertTrue(
            dc.accepts_vial_box(depth, image, vial_box, peer_confirmed=False, soft_clear=True)
        )

    def test_soft_clear_still_rejects_wall_and_empty_slab(self):
        image, depth, wall_box, slab_box, _vial, _jar = self._weak_hole_vial()
        self.assertFalse(
            dc.accepts_vial_box(depth, image, wall_box, soft_clear=True)
        )
        self.assertFalse(
            dc.accepts_vial_box(depth, image, slab_box, soft_clear=True)
        )

    def test_default_conf_lowered_for_recall(self):
        self.assertLess(dc.DEFAULT_YOLO_CONF, 0.25)
        self.assertAlmostEqual(dc.DEFAULT_YOLO_CONF, 0.15)
        strong = dc.strong_gate_profile()
        # camera_gates.json cam0_strong overrides hardcoded default when present.
        self.assertLessEqual(strong.yolo_conf, 0.18)
        self.assertGreaterEqual(strong.yolo_conf, 0.05)


class PeerIpcSmokeTests(unittest.TestCase):
    def test_udp_publish_receive(self):
        # Ephemeral base port to avoid clashing with a live tracker.
        base = 47950
        a = dc.PeerIpc(0, 1, base_port=base)
        b = dc.PeerIpc(1, 0, base_port=base)
        try:
            report = dc.PeerReport(
                timestamp=time.time(),
                camera_id=0,
                has_vial=True,
                box=(5, 5, 10, 20),
                distance_cm=33.0,
                conf=0.2,
            )
            a.publish(report)
            # Give the datagram a moment on localhost.
            got = None
            for _ in range(20):
                got = b.poll()
                if got is not None:
                    break
                time.sleep(0.01)
            self.assertIsNotNone(got)
            assert got is not None
            self.assertEqual(got.camera_id, 0)
            self.assertAlmostEqual(got.distance_cm, 33.0)
        finally:
            a.close()
            b.close()


class CloseIsolationTests(unittest.TestCase):
    def test_close_destroys_only_own_window(self):
        detector = dc.VialDetector(None, 0.25)
        cam = dc.OrbbecDepthCamera(detector=detector, camera_index=0, window="Vial-test-0")
        with mock.patch.object(dc.cv2, "destroyWindow") as destroy_one, mock.patch.object(
            dc.cv2, "destroyAllWindows"
        ) as destroy_all:
            cam.close()
        destroy_one.assert_called_once_with("Vial-test-0")
        destroy_all.assert_not_called()


class ReprojectAssistIpcTests(unittest.TestCase):
    def test_placeholder_extrinsics_refuse_peer_report(self):
        """IPC reproject must not invent boxes while stereo_extrinsics is PLACEHOLDER."""
        detector = dc.VialDetector(None, 0.25)
        cam = dc.OrbbecDepthCamera(detector=detector, camera_index=1, window="T")
        color = np.zeros((240, 320, 3), dtype=np.uint8)
        depth = np.full((240, 320), 400, dtype=np.uint16)
        peer = dc.PeerReport(
            timestamp=time.time(),
            camera_id=0,
            has_vial=True,
            box=(40, 40, 50, 70),
            distance_cm=40.0,
            conf=0.3,
        )
        helped = dc.try_3d_reproject_from_peer_report(
            cam, (color, depth), peer, soft_clear=True
        )
        self.assertIsNone(helped)

    def test_worker_argv_forwards_reproject_flag(self):
        args = dc.build_parser().parse_args(
            ["--dual-independent", "--3d-reproject-assist", "--no-edge-assist"]
        )
        cmd = dc._worker_argv_from_args(args, 0, 1)
        self.assertIn("--3d-reproject-assist", cmd)
        self.assertIn("--no-edge-assist", cmd)


if __name__ == "__main__":
    unittest.main()
