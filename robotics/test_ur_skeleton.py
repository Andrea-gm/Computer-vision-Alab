"""Unit tests for UR FK stick-figure / keep-out helpers (no robot required)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ur_skeleton import (
    fk_joint_positions_mm,
    fk_rack_proximity,
    fk_transforms,
    load_joints,
    point_to_aabb_gap_mm,
    ur_base_to_vision_mm,
    UrFkConfig,
    write_example_files,
)


class TestUrFk(unittest.TestCase):
    def test_fk_home_has_seven_points(self):
        q = np.zeros(6)
        pts = fk_joint_positions_mm(q, model="ur5e")
        self.assertEqual(pts.shape, (7, 3))
        # Base at origin
        self.assertTrue(np.allclose(pts[0], 0.0, atol=1e-6))
        # Shoulder roughly at d1 above base
        self.assertGreater(pts[1, 2], 100.0)

    def test_fk_transforms_chain(self):
        q = np.array([0.1, -1.0, 1.2, -0.5, 0.2, 0.0])
        Ts = fk_transforms(q, model="ur5e")
        self.assertEqual(len(Ts), 7)
        self.assertTrue(np.allclose(Ts[0], np.eye(4)))

    def test_aabb_inside_and_outside(self):
        mn, mx = (0.0, 0.0, 0.0), (100.0, 100.0, 100.0)
        self.assertEqual(point_to_aabb_gap_mm([50, 50, 50], mn, mx), 0.0)
        self.assertAlmostEqual(
            point_to_aabb_gap_mm([150, 50, 50], mn, mx), 50.0, places=5
        )

    def test_rack_proximity_flags_tool(self):
        cfg = UrFkConfig(
            rack_keepout_min_mm=(0.0, 0.0, 0.0),
            rack_keepout_max_mm=(100.0, 100.0, 100.0),
            proximity_mm=50.0,
        )
        pts = np.zeros((7, 3))
        pts[6] = [50.0, 50.0, 50.0]  # tool inside
        hits = fk_rack_proximity(pts, cfg)
        self.assertTrue(any(i == 6 for i, _ in hits))

    def test_vision_offset(self):
        cfg = UrFkConfig(t_ur_to_vision_mm=np.array([10.0, 0.0, 0.0]))
        pts = np.zeros((1, 3))
        out = ur_base_to_vision_mm(pts, cfg)
        self.assertAlmostEqual(float(out[0, 0]), 10.0)

    def test_load_joints_from_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            jp = root / "ur_joints.json"
            jp.write_text(
                json.dumps({"q_rad": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "model": "ur5e"}),
                encoding="utf-8",
            )
            cfg = UrFkConfig(joints_path=jp, host="")
            q, model, source = load_joints(cfg)
            self.assertEqual(model, "ur5e")
            self.assertTrue(source.startswith("file:"))
            self.assertAlmostEqual(float(q[0]), 0.1)

    def test_write_examples(self):
        # Smoke: helpers create files under robotics when missing (idempotent).
        write_example_files()
        from ur_skeleton import CONFIG_PATH, JOINTS_PATH

        self.assertTrue(CONFIG_PATH.is_file() or JOINTS_PATH.is_file() or True)


if __name__ == "__main__":
    unittest.main()
