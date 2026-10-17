"""Synthetic tests for snapshot cap and review R/W/I routing (no camera needed)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import depthcamera as dc


def _blank(w: int = 80, h: int = 60):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _fix(box=(10, 10, 30, 40), distance_mm: float | None = 400.0) -> dc.VialFix:
    return dc.VialFix(box=box, distance_mm=distance_mm, x_mm=0.0, y_mm=0.0, z_mm=distance_mm)


class SnapshotCapTests(unittest.TestCase):
    def test_caps_at_max_and_skips_near_duplicate_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            pending = Path(tmp) / "pending"
            recorder = dc.SnapshotRecorder(pending_dir=pending, max_saves=3)
            color = _blank()
            t0 = 1000.0
            # First save
            self.assertIsNotNone(recorder.maybe_save(0, color, _fix(), now=t0))
            # Same box too soon — skip
            self.assertIsNone(recorder.maybe_save(0, color, _fix(), now=t0 + 0.5))
            # Same box after interval but IoU high — skip
            self.assertIsNone(recorder.maybe_save(0, color, _fix(), now=t0 + 4.0))
            # Moved box — save
            self.assertIsNotNone(recorder.maybe_save(0, color, _fix(box=(40, 10, 30, 40)), now=t0 + 8.0))
            # Other camera can save independently
            self.assertIsNotNone(recorder.maybe_save(1, color, _fix(), now=t0 + 8.0))
            self.assertTrue(recorder.full)
            # Cap: further saves refused
            self.assertIsNone(recorder.maybe_save(1, color, _fix(box=(5, 5, 20, 30)), now=t0 + 20.0))
            self.assertEqual(recorder.saved, 3)
            jpgs = list(pending.glob("*.jpg"))
            jsons = list(pending.glob("*.json"))
            self.assertEqual(len(jpgs), 3)
            self.assertEqual(len(jsons), 3)
            meta = json.loads(jsons[0].read_text(encoding="utf-8"))
            self.assertIn("camera", meta)
            self.assertIn("box", meta)
            self.assertIn("distance_mm", meta)
            # Clean still: solid black (no green overlay burned in)
            img = cv2.imread(str(jpgs[0]))
            self.assertIsNotNone(img)
            self.assertEqual(img.shape, color.shape)


class ReviewRoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.pending = root / "pending"
        self.wrong = root / "wrong"
        self.incorrect = root / "incorrect"
        self.raw = root / "raw"
        self.pending.mkdir()
        self.patches = [
            mock.patch.object(dc, "REVIEW_PENDING", self.pending),
            mock.patch.object(dc, "REVIEW_WRONG", self.wrong),
            mock.patch.object(dc, "REVIEW_INCORRECT", self.incorrect),
            mock.patch.object(dc, "RAW_DATASET", self.raw),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _write_still(self, name: str = "cam0_test_0001.jpg", box=(8, 8, 24, 36)):
        path = self.pending / name
        color = _blank(100, 80)
        cv2.imwrite(str(path), color)
        meta = {
            "camera": 0,
            "box": list(box),
            "distance_mm": 350.0,
            "image_width": 100,
            "image_height": 80,
        }
        path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
        return path

    def test_r_copies_to_raw_with_yolo_label(self):
        path = self._write_still()
        meta = dc.load_sidecar(path.with_suffix(".json"))
        dest = dc.accept_correct(path, meta)
        self.assertIsNotNone(dest)
        self.assertTrue(dest.exists())
        self.assertTrue(dest.with_suffix(".txt").exists())
        label = dest.with_suffix(".txt").read_text(encoding="utf-8").strip()
        parts = label.split()
        self.assertEqual(len(parts), 5)
        self.assertEqual(parts[0], "0")
        self.assertFalse(path.exists())
        self.assertFalse(list(self.pending.glob("*.jpg")))

    def test_r_rejects_lid_only_short_box(self):
        # h/w = 12/40 = 0.30 — flatter than sideways min; must not enter raw
        path = self._write_still("cam0_short_0001.jpg", box=(10, 10, 40, 12))
        meta = dc.load_sidecar(path.with_suffix(".json"))
        dest = dc.accept_correct(path, meta)
        self.assertIsNone(dest)
        self.assertTrue(path.exists())
        self.assertFalse(list(self.raw.glob("*.jpg")))

    def test_r_accepts_near_square_topdown_lid(self):
        # h/w = 0.98 — near-square top-down lid face; live detect allows these
        path = self._write_still("cam0_topdown_0001.jpg", box=(20, 20, 50, 49))
        meta = dc.load_sidecar(path.with_suffix(".json"))
        self.assertTrue(dc.is_review_acceptable_box(tuple(meta["box"])))
        dest = dc.accept_correct(path, meta)
        self.assertIsNotNone(dest)
        self.assertTrue(dest.exists())
        self.assertTrue(dest.with_suffix(".txt").exists())
        self.assertFalse(path.exists())

    def test_r_accepts_sideways_fallen_box(self):
        # h/w = 36/64 ≈ 0.56 — elongated sideways lid+cup
        path = self._write_still("cam0_sideways_0001.jpg", box=(20, 40, 64, 36))
        meta = dc.load_sidecar(path.with_suffix(".json"))
        self.assertTrue(dc._is_sideways_box(tuple(meta["box"])))
        self.assertTrue(dc.is_review_acceptable_box(tuple(meta["box"])))
        dest = dc.accept_correct(path, meta)
        self.assertIsNotNone(dest)
        self.assertTrue(dest.exists())
        self.assertFalse(path.exists())

    def test_w_empty_label_and_wrong_archive(self):
        path = self._write_still("cam0_w_0001.jpg")
        dest = dc.mark_wrong(path)
        self.assertTrue(dest.exists())
        self.assertEqual(dest.with_suffix(".txt").read_text(encoding="utf-8"), "")
        archived = list(self.wrong.glob("*.jpg"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].with_suffix(".txt").read_text(encoding="utf-8"), "")
        self.assertFalse(path.exists())

    def test_i_moves_to_incorrect_and_relabel_saves_raw(self):
        path = self._write_still("cam0_i_0001.jpg")

        def fake_relabel(moved: Path, window=None):
            # Simulate user drawing a tight box and saving
            return dc._copy_to_raw_with_label(moved, "0 0.500000 0.500000 0.200000 0.300000\n")

        with mock.patch.object(dc, "relabel_incorrect", side_effect=fake_relabel):
            result = dc.mark_incorrect(path)
        self.assertIsNotNone(result)
        self.assertTrue(result.exists())
        self.assertTrue((self.incorrect / "cam0_i_0001.jpg").exists() or any(self.incorrect.glob("*.jpg")))
        self.assertFalse(path.exists())
        label = result.with_suffix(".txt").read_text(encoding="utf-8").strip()
        self.assertTrue(label.startswith("0 "))

    def test_box_draw_state_accepts_drag_and_writes_raw(self):
        """Simulate I-path mouse drag without a GUI; box must land in raw/."""
        path = self._write_still("cam0_i_draw_0001.jpg", box=(5, 5, 40, 15))  # wrong short meta box
        moved = dc._move_pair(path, self.incorrect)
        state = dc.BoxDrawState(scale=1.0)
        # Drag a full-vial box: w=40, h=50 → aspect 1.25
        state.on_mouse(cv2.EVENT_LBUTTONDOWN, 10, 10, 0, None)
        state.on_mouse(cv2.EVENT_MOUSEMOVE, 50, 60, 0, None)
        state.on_mouse(cv2.EVENT_LBUTTONUP, 50, 60, 0, None)
        self.assertEqual(len(state.boxes), 1)
        label, msg = state.try_save_label(100, 80)
        self.assertIsNotNone(label)
        self.assertEqual(msg, "")
        self.assertTrue(label.startswith("0 "))
        dest = dc._copy_to_raw_with_label(moved, label)
        self.assertTrue(dest.exists())
        self.assertTrue(dest.with_suffix(".txt").exists())
        parts = dest.with_suffix(".txt").read_text(encoding="utf-8").strip().split()
        self.assertEqual(parts[0], "0")
        # Saved box is the drawn one, not the short meta box
        # cx=(10+50)/2/100=0.3, cy=(10+60)/2/80=0.4375, bw=0.4, bh=50/80=0.625
        self.assertAlmostEqual(float(parts[1]), 0.3, places=4)
        self.assertAlmostEqual(float(parts[3]), 0.4, places=4)
        self.assertAlmostEqual(float(parts[4]), 0.625, places=4)

    def test_box_draw_state_keeps_short_box_but_blocks_save(self):
        state = dc.BoxDrawState(scale=1.0)
        # Flat lid-strip drag: w=40, h=12 → aspect 0.30 (below sideways min)
        state.on_mouse(cv2.EVENT_LBUTTONDOWN, 10, 10, 0, None)
        state.on_mouse(cv2.EVENT_LBUTTONUP, 50, 22, 0, None)
        self.assertEqual(len(state.boxes), 1)  # drawing allowed
        label, msg = state.try_save_label(100, 80)
        self.assertIsNone(label)
        self.assertIn("h/w", msg.lower())
        self.assertEqual(len(state.boxes), 1)  # box kept for user to extend


class SharedDetectorTests(unittest.TestCase):
    def test_two_cameras_share_one_detector_instance(self):
        detector = dc.VialDetector(None, 0.25)
        a = dc.OrbbecDepthCamera(detector=detector, camera_index=0, window="A")
        b = dc.OrbbecDepthCamera(detector=detector, camera_index=1, window="B")
        self.assertIs(a.detector, b.detector)
        self.assertEqual(a.window, "A")
        self.assertEqual(b.window, "B")


class FullVialAspectTests(unittest.TestCase):
    def test_full_vial_aspect_gate(self):
        self.assertTrue(dc.is_full_vial_aspect((0, 0, 40, 50)))  # 1.25
        self.assertFalse(dc.is_full_vial_aspect((0, 0, 40, 20)))  # 0.5 lid-only
        self.assertFalse(dc.is_full_vial_aspect((0, 0, 40, 80)))  # 2.0 too tall

    def test_review_accepts_topdown_near_square_not_flat_lid(self):
        # Upright full vial still ok
        self.assertTrue(dc.is_review_acceptable_box((0, 0, 40, 50)))
        # Near-square top-down (was blocked by old R min floor 1.0)
        self.assertFalse(dc.is_full_vial_aspect((0, 0, 50, 49)))  # 0.98
        self.assertTrue(dc._is_topdown_box((0, 0, 50, 49)))
        self.assertTrue(dc.is_review_acceptable_box((0, 0, 50, 49)))
        # Sideways / fallen elongated lid+cup (clearly horizontal ≤0.71)
        self.assertTrue(dc._is_sideways_box((0, 0, 70, 40)))  # 0.57
        self.assertTrue(dc.is_review_acceptable_box((0, 0, 70, 40)))
        # Borderline squat (was old sideways max 0.85) is top-down, not sideways
        self.assertFalse(dc._is_sideways_box((0, 0, 70, 56)))  # 0.80
        self.assertTrue(dc._is_topdown_box((0, 0, 70, 56)))
        # Flat lid-strip still rejected (below sideways min 0.40)
        self.assertFalse(dc.is_review_acceptable_box((0, 0, 40, 14)))
        msg = dc.review_box_reject_message((0, 0, 40, 14))
        self.assertIn("almost flat", msg.lower())
        self.assertIn("press i", msg.lower())

    def test_fit_max_height_allows_taller_than_old_1_2(self):
        self.assertGreater(dc.FIT_MAX_HEIGHT_CAP_RATIO, 1.2)
        self.assertAlmostEqual(dc.FIT_MAX_HEIGHT_CAP_RATIO, 1.55)


def _desk_scene(w: int = 320, h: int = 240):
    """Gray desk + mottled wall + clear-ish vial with solid white cap."""
    image = np.full((h, w, 3), 90, dtype=np.uint8)
    # Mottled wall / background (no localized lid)
    wall = image[40:140, 20:100]
    wall[:] = (160, 160, 160)
    wall[::4, :] = (150, 150, 150)
    wall[:, ::5] = (145, 145, 145)
    # Uniform bright slab (false "white" region without cup contrast)
    image[40:120, 220:300] = (230, 230, 230)
    # Vial body (slightly darker) with white screw cap on top
    vx, vy, vw, vh = 130, 80, 36, 48
    image[vy + 10 : vy + vh, vx : vx + vw] = (70, 70, 65)
    image[vy : vy + 10, vx : vx + vw] = (245, 245, 245)
    # A few vertical edges so structure check passes
    image[vy + 10 : vy + vh, vx] = (40, 40, 40)
    image[vy + 10 : vy + vh, vx + vw - 1] = (40, 40, 40)
    wall_box = (20, 40, 80, 100)
    slab_box = (220, 40, 80, 80)
    vial_box = (vx, vy, vw, vh)
    return image, wall_box, slab_box, vial_box


class WhiteCapGatingTests(unittest.TestCase):
    def test_has_white_cap_accepts_vial_rejects_background(self):
        image, wall_box, slab_box, vial_box = _desk_scene()
        self.assertTrue(dc.has_white_cap(image, vial_box))
        self.assertFalse(dc.has_white_cap(image, wall_box))
        self.assertFalse(dc.has_white_cap(image, slab_box))

    def test_uniform_wall_fails_interior_structure(self):
        image, wall_box, slab_box, vial_box = _desk_scene()
        self.assertTrue(dc.has_interior_structure(image, vial_box))
        self.assertFalse(dc.has_interior_structure(image, slab_box))

    def test_choose_box_prefers_capped_over_larger_wall(self):
        image, wall_box, _slab, vial_box = _desk_scene()
        # Larger background blob listed first (old behavior would stick here)
        chosen = dc.choose_box([wall_box, vial_box], previous=None, image=image)
        self.assertEqual(chosen, vial_box)

    def test_choose_box_drops_bad_previous_hold(self):
        image, wall_box, _slab, vial_box = _desk_scene()
        # Locked onto wall previously; must jump to the real vial
        chosen = dc.choose_box([wall_box, vial_box], previous=wall_box, image=image)
        self.assertEqual(chosen, vial_box)

    def test_cap_run_strong_on_vial_weak_on_mottled_wall(self):
        image, wall_box, slab_box, vial_box = _desk_scene()
        self.assertGreaterEqual(dc.cap_run(image, vial_box), 0.75)
        # Mottled wall may still have a long bright run, but has_white_cap rejects it
        # via mid-band contrast; slab is bright throughout.
        self.assertFalse(dc.has_white_cap(image, wall_box))
        self.assertFalse(dc.has_white_cap(image, slab_box))


def _clear_vial_depth_scene(w: int = 320, h: int = 240, vial_z: float = 500.0):
    """Color + depth: white-cap clear vial (holes), uniform wall, opaque jar blob."""
    image, wall_box, slab_box, vial_box = _desk_scene(w, h)
    # Opaque mason-jar style: white lid + solid painted body (same color cues as vial).
    jx, jy, jw, jh = 250, 150, 40, 52
    image[jy + 10 : jy + jh, jx : jx + jw] = (55, 60, 70)
    image[jy : jy + 10, jx : jx + jw] = (245, 245, 245)
    image[jy + 10 : jy + jh, jx] = (30, 30, 30)
    image[jy + 10 : jy + jh, jx + jw - 1] = (30, 30, 30)
    jar_box = (jx, jy, jw, jh)

    depth = np.full((h, w), 1800.0, dtype=np.float32)  # far background
    # Uniform wall / slab: dense near-ish depth (opaque surface)
    wx, wy, ww, wh = wall_box
    depth[wy : wy + wh, wx : wx + ww] = 900.0
    sx, sy, sw, sh = slab_box
    depth[sy : sy + sh, sx : sx + sw] = 850.0
    # Clear vial: solid lid depth, cup mostly invalid / far (see-through)
    vx, vy, vw, vh = vial_box
    depth[vy : vy + 10, vx : vx + vw] = vial_z
    cup = depth[vy + 10 : vy + vh, vx : vx + vw]
    cup[:] = 0.0  # invalid holes through clear plastic
    cup[::3, ::4] = 1600.0  # sparse far wall peeking through
    # Opaque jar: lid + body both solid near depth
    depth[jy : jy + jh, jx : jx + jw] = vial_z + 40.0
    return image, depth, wall_box, slab_box, vial_box, jar_box


class ClearCupDepthTests(unittest.TestCase):
    def test_white_cap_depth_hole_box_passes(self):
        image, depth, _wall, _slab, vial_box, _jar = _clear_vial_depth_scene()
        self.assertTrue(dc.has_white_cap(image, vial_box))
        self.assertTrue(dc.looks_like_clear_cup(depth, image, vial_box))
        self.assertTrue(dc.looks_like_capped_clear_vial(depth, image, vial_box))

    def test_uniform_wall_fails_clear_cup(self):
        image, depth, wall_box, slab_box, _vial, _jar = _clear_vial_depth_scene()
        self.assertFalse(dc.looks_like_clear_cup(depth, image, wall_box))
        self.assertFalse(dc.looks_like_clear_cup(depth, image, slab_box))
        self.assertFalse(dc.looks_like_capped_clear_vial(depth, image, wall_box))

    def test_solid_opaque_blob_fails(self):
        image, depth, _wall, _slab, _vial, jar_box = _clear_vial_depth_scene()
        # Color may look capped, but dense near-depth in the body rejects mason-jar style.
        self.assertTrue(dc.has_white_cap(image, jar_box))
        self.assertFalse(dc.looks_like_clear_cup(depth, image, jar_box))
        self.assertFalse(dc.looks_like_capped_clear_vial(depth, image, jar_box))

    def test_choose_box_with_depth_prefers_clear_vial(self):
        image, depth, wall_box, _slab, vial_box, jar_box = _clear_vial_depth_scene()
        chosen = dc.choose_box(
            [wall_box, jar_box, vial_box],
            previous=None,
            image=image,
            depth_mm=depth,
        )
        self.assertEqual(chosen, vial_box)


class AnnotateLabelTests(unittest.TestCase):
    def test_annotate_puts_vial_first_in_label_lines(self):
        """Overlay must show 'vial' on the box, not only the distance readout."""
        image = _blank(200, 160)
        captured: list[list[str]] = []

        def capture_label(_image, _origin, lines, _color):
            captured.append(list(lines))

        with mock.patch.object(dc, "draw_label", side_effect=capture_label):
            dc.annotate(
                image,
                _fix(box=(40, 50, 40, 50), distance_mm=420.0),
                show_vial_labels=True,
            )
            # Force non-zero offsets so L/R and U/D lines appear.
            fix_offset = dc.VialFix(
                box=(40, 50, 40, 50),
                distance_mm=420.0,
                x_mm=-25.0,
                y_mm=18.0,
                z_mm=420.0,
            )
            dc.annotate(image, fix_offset, show_vial_labels=True)
            dc.annotate(
                image,
                _fix(box=(40, 50, 40, 50), distance_mm=None),
                show_vial_labels=True,
            )

        self.assertEqual(len(captured), 3)
        self.assertEqual(captured[0][0], "vial")
        self.assertIn("close", captured[0])
        self.assertTrue(any("cm away" in line for line in captured[0]))
        self.assertIn("close", captured[1])
        self.assertEqual(captured[2][0], "vial")
        self.assertIn("No depth on the cap", captured[2])

    def test_choose_box_prefers_nearer_depth(self):
        """Foreground/close vial must beat a farther rack lid of similar aspect."""
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        near_box = (40, 40, 50, 70)
        far_box = (200, 40, 50, 70)
        for box in (near_box, far_box):
            x, y, w, h = box
            image[y : y + 12, x : x + w] = (245, 245, 245)
            image[y + 12 : y + h, x : x + w] = (40, 50, 60)
            image[y + 12 : y + h, x] = (20, 20, 20)
            image[y + 12 : y + h, x + w - 1] = (20, 20, 20)
        depth = np.full((240, 320), 1800.0, dtype=np.float32)
        nx, ny, nw, nh = near_box
        fx, fy, fw, fh = far_box
        depth[ny : ny + 12, nx : nx + nw] = 350.0
        depth[ny + 12 : ny + nh, nx : nx + nw] = 0.0
        depth[fy : fy + 12, fx : fx + fw] = 900.0
        depth[fy + 12 : fy + fh, fx : fx + fw] = 0.0
        chosen = dc.choose_box(
            [far_box, near_box],
            previous=None,
            image=image,
            depth_mm=depth,
        )
        self.assertEqual(chosen, near_box)
        self.assertEqual(dc.range_band_label(350.0), "close")
        self.assertEqual(dc.range_band_label(900.0), "mid")

    def test_usable_detection_allows_near_square_lid_face(self):
        """Top-down lid faces must not die on upright aspect_min before lid@Z."""
        image = _blank(200, 160)
        # h/w = 36/40 = 0.9 — below strong aspect_min (~0.95) but top-down band.
        box = (80, 60, 40, 36)
        self.assertTrue(dc._is_topdown_box(box))
        self.assertTrue(dc.usable_detection(image, box))


class StickyPeerMemoryTests(unittest.TestCase):
    def test_activates_on_strict_hit_and_survives_peer_misses(self):
        mem = dc.PeerMemory(window=15)
        self.assertFalse(mem.active)
        self.assertTrue(mem.observe(True))
        self.assertEqual(mem.frames_since_strict, 0)
        # Peer (and even both cams) miss this frame — soft assist must stay on.
        for i in range(1, 16):
            self.assertTrue(mem.observe(False), msg=f"expected active at miss {i}")
            self.assertEqual(mem.frames_since_strict, i)
        self.assertTrue(mem.active)

    def test_clears_after_both_miss_longer_than_window(self):
        mem = dc.PeerMemory(window=15)
        mem.observe(True)
        for _ in range(15):
            mem.observe(False)
        self.assertTrue(mem.active)
        self.assertFalse(mem.observe(False))  # miss #16 → clear
        self.assertIsNone(mem.frames_since_strict)
        self.assertFalse(mem.active)

    def test_soft_hits_do_not_refresh_memory(self):
        """Only strict clears refresh; soft-gate assists must not extend the timer."""
        mem = dc.PeerMemory(window=3)
        mem.observe(True)
        mem.observe(False)  # frames_since = 1
        mem.observe(False)  # 2
        # Soft assist succeeding is not a strict hit — still aging.
        self.assertTrue(mem.observe(False))  # 3, still active
        self.assertFalse(mem.observe(False))  # 4 > window → clear

    def test_refresh_on_new_strict_hit(self):
        mem = dc.PeerMemory(window=5)
        mem.observe(True)
        for _ in range(4):
            mem.observe(False)
        self.assertEqual(mem.frames_since_strict, 4)
        mem.observe(True)
        self.assertEqual(mem.frames_since_strict, 0)
        self.assertTrue(mem.active)

    def test_soft_gate_accepts_white_cap_vial_without_clear_depth(self):
        """Angle/occlusion can kill clear-cup holes; peer memory unlocks soft gate."""
        image, depth, _wall, _slab, vial_box, _jar = _clear_vial_depth_scene()
        # Fill cup with near depth so clear-cup check fails (bad angle / occlusion).
        vx, vy, vw, vh = vial_box
        depth[vy + 10 : vy + vh, vx : vx + vw] = 520.0
        self.assertFalse(dc.looks_like_capped_clear_vial(depth, image, vial_box))
        self.assertFalse(
            dc.accepts_vial_box(
                depth, image, vial_box, peer_confirmed=False, soft_clear=False
            )
        )
        self.assertTrue(dc.accepts_vial_box(depth, image, vial_box, peer_confirmed=True))
        # Independent dual uses soft_clear instead of peer sticky memory.
        self.assertTrue(
            dc.accepts_vial_box(
                depth, image, vial_box, peer_confirmed=False, soft_clear=True
            )
        )

    def test_soft_gate_still_rejects_blank_wall(self):
        image, depth, wall_box, slab_box, _vial, _jar = _clear_vial_depth_scene()
        self.assertFalse(dc.accepts_vial_box(depth, image, wall_box, peer_confirmed=True))
        self.assertFalse(dc.accepts_vial_box(depth, image, slab_box, peer_confirmed=True))
        self.assertFalse(dc.accepts_vial_box(depth, image, wall_box, soft_clear=True))
        self.assertFalse(dc.accepts_vial_box(depth, image, slab_box, soft_clear=True))


if __name__ == "__main__":
    # Ensure imports resolve from this worktree
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    unittest.main()
