"""Unit tests for fixed-slot open/closed lid classification."""

from __future__ import annotations

import unittest

import numpy as np

import rack_lid_state as rls


def _closed_patch(size: int = 80) -> np.ndarray:
    img = np.full((size, size, 3), 30, dtype=np.uint8)
    cv = __import__("cv2")
    cv.circle(img, (size // 2, size // 2), size // 3, (245, 245, 245), -1)
    return img


def _open_patch(size: int = 80) -> np.ndarray:
    img = np.full((size, size, 3), 30, dtype=np.uint8)
    # Beige interior, darker rim — not a solid bright white disk.
    cv = __import__("cv2")
    cv.circle(img, (size // 2, size // 2), size // 3, (180, 200, 210), -1)
    cv.circle(img, (size // 2, size // 2), size // 3, (90, 90, 90), 3)
    return img


def _beige_patch(size: int = 48) -> np.ndarray:
    img = np.full((size, size, 3), 20, dtype=np.uint8)
    cv = __import__("cv2")
    # Dimmer cream than a white lid: hue ~19, saturation high enough to miss the white mask.
    cv.ellipse(img, (size // 2, size // 2), (14, 8), 0, 0, 360, (80, 130, 160), -1)
    return img


def _white_lid_patch(size: int = 48) -> np.ndarray:
    img = np.full((size, size, 3), 25, dtype=np.uint8)
    cv = __import__("cv2")
    cv.ellipse(img, (size // 2, size // 2), (12, 6), 0, 0, 360, (245, 245, 245), -1)
    return img


class RackLidStateTests(unittest.TestCase):
    def test_lid_patch_closed_white(self):
        state, _white_n, _beige_n = rls.classify_lid_patch(_white_lid_patch())
        self.assertEqual(state, "closed")

    def test_lid_patch_open_beige(self):
        state, _white_n, _beige_n = rls.classify_lid_patch(_beige_patch())
        self.assertEqual(state, "open")

    def test_lid_patch_empty_dark(self):
        img = np.full((40, 40, 3), 20, dtype=np.uint8)
        state, _white_n, _beige_n = rls.classify_lid_patch(img)
        self.assertEqual(state, "empty")

    def test_closed_white_disk(self):
        img = _closed_patch()
        slot = rls.Slot("s0", cx=40, cy=40, size=70)
        result = rls.classify_slot(img, slot)
        self.assertEqual(result.state, "closed")

    def test_open_beige_interior(self):
        img = _open_patch()
        slot = rls.Slot("s0", cx=40, cy=40, size=70)
        result = rls.classify_slot(img, slot)
        self.assertEqual(result.state, "open")

    def test_empty_dark(self):
        img = np.full((80, 80, 3), 25, dtype=np.uint8)
        slot = rls.Slot("s0", cx=40, cy=40, size=70)
        result = rls.classify_slot(img, slot)
        self.assertEqual(result.state, "empty")


if __name__ == "__main__":
    unittest.main()
