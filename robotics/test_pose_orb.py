"""Synthetic tests for sideways/top-down pose gates + tiny-weight ORB assist."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

import depthcamera as dc
from test_dual_review import _desk_scene


def _sideways_vial_scene(w: int = 320, h: int = 240):
    """Fallen vial: white cap on the left end, darker body to the right."""
    image = np.full((h, w, 3), 90, dtype=np.uint8)
    # Mottled wall
    wall = image[30:130, 10:90]
    wall[:] = (160, 160, 160)
    wall[::4, :] = (150, 150, 150)
    # Bright slab (false positive)
    image[40:100, 230:310] = (230, 230, 230)
    # Sideways vial: long axis ~70 px, short ~40 → h/w = 0.57
    vx, vy, vw, vh = 110, 90, 70, 40
    image[vy : vy + vh, vx + 14 : vx + vw] = (70, 70, 65)
    image[vy : vy + vh, vx : vx + 14] = (245, 245, 245)
    # Horizontal edges (cup top/bottom)
    image[vy, vx + 14 : vx + vw] = (40, 40, 40)
    image[vy + vh - 1, vx + 14 : vx + vw] = (40, 40, 40)
    vial_box = (vx, vy, vw, vh)
    wall_box = (10, 30, 80, 100)
    slab_box = (230, 40, 80, 60)
    return image, wall_box, slab_box, vial_box


def _topdown_lid_scene(w: int = 320, h: int = 240):
    """Near-square white lid face (top-down)."""
    image = np.full((h, w, 3), 85, dtype=np.uint8)
    lx, ly, lw, lh = 130, 90, 48, 46  # h/w ≈ 0.96
    image[ly : ly + lh, lx : lx + lw] = (240, 240, 240)
    # Soft rim so interior edges exist
    image[ly, lx : lx + lw] = (200, 200, 200)
    image[ly + lh - 1, lx : lx + lw] = (200, 200, 200)
    image[ly : ly + lh, lx] = (200, 200, 200)
    image[ly : ly + lh, lx + lw - 1] = (200, 200, 200)
    return image, (lx, ly, lw, lh)


def _textured_cap_pair():
    """Two similar textured white-cap crops (enough ORB keypoints)."""
    rng = np.random.default_rng(7)
    # Tall-ish crop so extract_orb keeps a useful white-cap band (not near-empty).
    a = np.full((96, 80, 3), 235, dtype=np.uint8)
    # Checker / corner texture — ORB needs high-contrast corners on the lid face.
    for i in range(0, 96, 8):
        for j in range(0, 80, 8):
            if ((i // 8) + (j // 8)) % 2 == 0:
                a[i : i + 4, j : j + 4] = 40
    a[10:18, 10:70] = 25
    a[70:78, 12:68] = 30
    a[20:70, 18:26] = 20
    a[20:70, 54:62] = 28
    noise = rng.integers(0, 25, size=a.shape, dtype=np.uint8)
    a = np.clip(a.astype(np.int16) - noise, 0, 255).astype(np.uint8)
    b = np.roll(a, 3, axis=1)
    # Full-frame ROI (skip white-band shrink) via None box path in one call;
    # also exercise boxed path with the full image box.
    box = (0, 0, 80, 96)
    return a, b, box


def _sideways_vial_right_cap(w: int = 320, h: int = 240):
    """Fallen vial with white cap on the right end (mirror of left-cap scene)."""
    image = np.full((h, w, 3), 90, dtype=np.uint8)
    vx, vy, vw, vh = 110, 90, 70, 40
    image[vy : vy + vh, vx : vx + vw - 14] = (70, 70, 65)
    image[vy : vy + vh, vx + vw - 14 : vx + vw] = (245, 245, 245)
    image[vy, vx : vx + vw - 14] = (40, 40, 40)
    image[vy + vh - 1, vx : vx + vw - 14] = (40, 40, 40)
    return image, (vx, vy, vw, vh)


class SidewaysPoseTests(unittest.TestCase):
    def setUp(self):
        dc.set_allow_sideways(True)
        dc.set_sideways_assist(True)

    def test_sideways_aspect_and_usable(self):
        box = (10, 10, 70, 40)  # 0.57
        self.assertTrue(dc._is_sideways_box(box))
        self.assertFalse(dc.is_full_vial_aspect(box))
        image = np.full((200, 300, 3), 80, dtype=np.uint8)
        self.assertTrue(dc.usable_detection(image, box))

    def test_flat_lid_strip_still_rejected(self):
        box = (10, 10, 50, 14)  # 0.28
        self.assertFalse(dc._is_sideways_box(box))
        self.assertFalse(dc.is_review_acceptable_box(box))
        image = np.full((200, 300, 3), 80, dtype=np.uint8)
        self.assertFalse(dc.usable_detection(image, box))

    def test_topdown_still_usable(self):
        image, box = _topdown_lid_scene()
        self.assertTrue(dc._is_topdown_box(box))
        self.assertTrue(dc.usable_detection(image, box))
        self.assertTrue(dc.is_review_acceptable_box(box))

    def test_sideways_has_white_cap_and_capped_look(self):
        image, wall_box, slab_box, vial_box = _sideways_vial_scene()
        self.assertTrue(dc._is_sideways_box(vial_box))
        self.assertTrue(dc.usable_detection(image, vial_box))
        self.assertTrue(dc.has_white_cap(image, vial_box))
        self.assertTrue(dc.looks_like_capped_vial(image, vial_box))
        self.assertFalse(dc.has_white_cap(image, wall_box))
        self.assertFalse(dc.has_white_cap(image, slab_box))

    def test_soft_clear_accepts_sideways_rejects_slab(self):
        image, _wall, slab_box, vial_box = _sideways_vial_scene()
        depth = np.full(image.shape[:2], 900.0, dtype=np.float32)
        vx, vy, vw, vh = vial_box
        depth[vy : vy + vh, vx : vx + 14] = 520.0
        # Weak clear holes in body
        depth[vy : vy + vh, vx + 14 : vx + vw] = 520.0
        self.assertTrue(
            dc.accepts_vial_box(
                depth, image, vial_box, peer_confirmed=False, soft_clear=True
            )
        )
        self.assertFalse(
            dc.accepts_vial_box(
                depth, image, slab_box, peer_confirmed=False, soft_clear=True
            )
        )

    def test_vial_scale_sideways_long_short(self):
        # At Z=500, fx=fy=600: 70×40 px → ~58×33 mm (long≈H, short≈Ø band).
        image, _w, _s, vial_box = _sideways_vial_scene()
        depth = np.full(image.shape[:2], 500.0, dtype=np.float32)
        vx, vy, vw, vh = vial_box
        depth[vy : vy + vh, vx : vx + 14] = 500.0
        K = (600.0, 600.0, 160.0, 120.0)
        self.assertTrue(dc.vial_scale(depth, image, vial_box, K))
        # Wall-sized sideways slab fails (long axis ≫ vial H).
        huge = (10, 40, 260, 120)
        image[40:160, 10:270] = (240, 240, 240)
        depth[40:160, 10:270] = 500.0
        self.assertFalse(dc.vial_scale(depth, image, huge, K))

    def test_grow_sideways_from_left_lid(self):
        """Cap-first horizontal growth: white end-cap → clear body to the right."""
        image, _wall, _slab, vial_box = _sideways_vial_scene()
        vx, vy, vw, vh = vial_box
        lid = (vx, vy, 14, vh)
        depth = np.full(image.shape[:2], 900.0, dtype=np.float32)
        depth[vy : vy + vh, vx : vx + 14] = 520.0
        # Holes through clear body
        body = depth[vy : vy + vh, vx + 14 : vx + vw]
        body[:] = 0.0
        grown = dc.grow_sideways_from_lid(image, lid, depth)
        self.assertIsNotNone(grown)
        assert grown is not None
        self.assertTrue(dc._is_sideways_box(grown))
        self.assertGreaterEqual(grown[2], vw - 8)
        self.assertGreater(dc._end_cap_lr_score(image, grown), 0.1)

    def test_grow_sideways_from_right_lid(self):
        image, vial_box = _sideways_vial_right_cap()
        vx, vy, vw, vh = vial_box
        lid = (vx + vw - 14, vy, 14, vh)
        grown = dc.grow_sideways_from_lid(image, lid, None)
        self.assertIsNotNone(grown)
        assert grown is not None
        self.assertTrue(dc._is_sideways_box(grown))
        self.assertEqual(dc._lid_end_side(image, grown), "right")

    def test_expand_wide_yolo_partial_box(self):
        """Wide partial YOLO box expands along long axis instead of being rejected."""
        image, _wall, _slab, vial_box = _sideways_vial_scene()
        vx, vy, vw, vh = vial_box
        # Truncated wide box: lid + short body stub (h/w≈0.80 — outside sideways band)
        partial = (vx, vy, 50, 40)  # h/w = 0.80
        depth = np.full(image.shape[:2], 900.0, dtype=np.float32)
        depth[vy : vy + vh, vx : vx + 14] = 520.0
        depth[vy : vy + vh, vx + 14 : vx + vw] = 0.0
        grown = dc.expand_wide_proposal_sideways(image, depth, partial)
        self.assertIsNotNone(grown)
        assert grown is not None
        self.assertTrue(dc._is_sideways_box(grown))
        self.assertGreaterEqual(grown[2], partial[2])

    def test_upright_lid_not_grown_horizontally(self):
        """Wide upright lid strips must not take the sideways horizontal path."""
        image = np.full((240, 320, 3), 80, dtype=np.uint8)
        # Wide top lid (upright view): bw/bh ≫ 1.35
        image[40:52, 120:170] = (245, 245, 245)
        image[52:100, 125:165] = (70, 70, 65)
        lid = (120, 40, 50, 12)
        self.assertIsNone(dc.grow_sideways_from_lid(image, lid, None))

    def test_cap_first_finds_synthetic_sideways(self):
        image, _wall, _slab, vial_box = _sideways_vial_scene()
        depth = np.full(image.shape[:2], 900.0, dtype=np.float32)
        vx, vy, vw, vh = vial_box
        depth[vy : vy + vh, vx : vx + 14] = 520.0
        depth[vy : vy + vh, vx + 14 : vx + vw] = 0.0
        found = dc.detect_vial_by_cap(image, depth, None)
        self.assertTrue(found)
        hit = any(
            dc._is_sideways_box(b) and dc.box_iou(b, vial_box) >= 0.35 for b in found
        )
        self.assertTrue(hit, f"no sideways overlap among {found}")

    def test_sideways_assist_cli_default_on(self):
        self.assertTrue(dc.DEFAULT_SIDEWAYS_ASSIST)
        parser = dc.build_parser()
        args = parser.parse_args([])
        self.assertTrue(args.sideways_assist)
        args_off = parser.parse_args(["--no-sideways-assist"])
        self.assertFalse(args_off.sideways_assist)

    def test_choose_box_sideways_only_when_no_upright(self):
        image, _wall, _slab, side_box = _sideways_vial_scene()
        depth = np.full(image.shape[:2], 900.0, dtype=np.float32)
        vx, vy, vw, vh = side_box
        depth[vy : vy + vh, vx : vx + 14] = 520.0
        depth[vy : vy + vh, vx + 14 : vx + vw] = 0.0
        chosen = dc.choose_box(
            [side_box],
            previous=None,
            image=image,
            depth_mm=depth,
            soft_clear=True,
        )
        self.assertEqual(chosen, side_box)


class OrbAssistTests(unittest.TestCase):
    def test_empty_match_soft_fails(self):
        blank = np.full((80, 80, 3), 40, dtype=np.uint8)
        kp, desc = dc.extract_orb_descriptors(blank, (10, 10, 40, 40))
        self.assertIsNone(kp)
        self.assertIsNone(desc)
        n, score = dc.match_orb_descriptors(None, None)
        self.assertEqual(n, 0)
        self.assertEqual(score, 0.0)
        self.assertIsNone(dc.orb_agree_label(None, None))
        self.assertEqual(dc.orb_cap_match_score(blank, (0, 0, 40, 40), blank, (0, 0, 40, 40)), 0.0)

    def test_textured_caps_match(self):
        a, b, box = _textured_cap_pair()
        # Full-frame extract (no box) — guarantees texture reaches ORB.
        _, da = dc.extract_orb_descriptors(a, None)
        _, db = dc.extract_orb_descriptors(b, None)
        self.assertIsNotNone(da)
        self.assertIsNotNone(db)
        n, score = dc.match_orb_descriptors(da, db)
        self.assertGreater(n, 0)
        self.assertGreater(score, 0.0)
        # Boxed white-cap path must soft-fail or return descriptors (never crash).
        _kp, _d = dc.extract_orb_descriptors(a, box)

    def test_pack_unpack_roundtrip(self):
        a, _b, _box = _textured_cap_pair()
        _, da = dc.extract_orb_descriptors(a, None)
        packed = dc.pack_orb_descriptors(da)
        self.assertIsNotNone(packed)
        assert packed is not None
        self.assertLessEqual(len(packed), dc.ORB_IPC_MAX_FEATURES)
        back = dc.unpack_orb_descriptors(packed)
        self.assertIsNotNone(back)
        assert back is not None
        self.assertEqual(back.shape[1], 32)

    def test_peer_report_orb_optional(self):
        report = dc.peer_report_from_fix(0, None, conf=0.2)
        self.assertIsNone(report.orb_desc)
        payload = report.to_payload()
        self.assertIn("orb_desc", payload)
        parsed = dc.parse_peer_report(payload)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertIsNone(parsed.orb_desc)

    def test_choose_box_orb_boost_does_not_invent(self):
        image, wall_box, slab_box, vial_box = _desk_scene()
        # No peer ORB → same as today (still picks capped vial via white-cap gates).
        chosen = dc.choose_box(
            [wall_box, slab_box, vial_box],
            previous=None,
            image=image,
            peer_orb_desc=None,
            orb_assist_weight=0.08,
        )
        self.assertEqual(chosen, vial_box)
        # Empty peer descriptors → soft no-op.
        chosen2 = dc.choose_box(
            [wall_box, vial_box],
            previous=None,
            image=image,
            peer_orb_desc=np.zeros((0, 32), dtype=np.uint8),
            orb_assist_weight=0.08,
        )
        self.assertEqual(chosen2, vial_box)

    def test_defaults_and_cli_flag_names(self):
        self.assertFalse(dc.DEFAULT_USE_ORB_ASSIST)
        self.assertGreaterEqual(dc.DEFAULT_ORB_ASSIST_WEIGHT, 0.03)
        self.assertLessEqual(dc.DEFAULT_ORB_ASSIST_WEIGHT, 0.1)
        self.assertTrue(dc.DEFAULT_ALLOW_SIDEWAYS)
        parser = dc.build_parser()
        args = parser.parse_args([])
        self.assertFalse(args.use_orb_assist)
        self.assertTrue(args.allow_sideways)
        args_on = parser.parse_args(["--orb-assist"])
        self.assertTrue(args_on.use_orb_assist)
        args_off = parser.parse_args(["--no-orb-assist"])
        self.assertFalse(args_off.use_orb_assist)
        args_ns = parser.parse_args(["--no-sideways"])
        self.assertFalse(args_ns.allow_sideways)

    def test_choose_box_upright_beats_sideways(self):
        """Upright-first: a full-vial box wins even if a sideways candidate is present."""
        image, _wall, _slab, side_box = _sideways_vial_scene()
        # Paint an upright vial elsewhere on the same frame.
        ux, uy, uw, uh = 200, 80, 40, 52  # h/w = 1.3
        image[uy : uy + uh, ux : ux + uw] = (70, 70, 65)
        image[uy : uy + 12, ux : ux + uw] = (245, 245, 245)
        upright = (ux, uy, uw, uh)
        self.assertTrue(dc.is_full_vial_aspect(upright))
        self.assertTrue(dc._is_sideways_box(side_box))
        with mock.patch.object(dc, "has_white_cap", side_effect=lambda _i, b, **_k: b in (upright, side_box)):
            chosen = dc.choose_box(
                [side_box, upright],
                previous=None,
                image=image,
                peer_orb_desc=np.ones((8, 32), dtype=np.uint8),
                orb_assist_weight=0.5,
            )
        self.assertEqual(chosen, upright)


if __name__ == "__main__":
    unittest.main()
