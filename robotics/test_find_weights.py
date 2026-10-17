"""find_weights must prefer vial-live / bon-winner over newer scratch runs."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import depthcamera as dc


class FindWeightsTests(unittest.TestCase):
    def test_explicit_missing_returns_none(self):
        self.assertIsNone(dc.find_weights(Path("no_such_weights_zzz.pt")))

    def test_prefers_vial_live_over_newer_scratch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = root / "runs" / "vial-live" / "weights" / "best.pt"
            winner = root / "runs" / "bon-winner" / "weights" / "best.pt"
            scratch = root / "runs" / "detect" / "runs" / "bon-vid-08-bgfp" / "weights" / "best.pt"
            for path in (live, winner, scratch):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            # Make scratch look newer.
            import os
            import time

            newer = time.time() + 1000
            os.utime(scratch, (newer, newer))
            with mock.patch.object(dc, "ROOT", root), mock.patch.object(
                dc, "LIVE_WEIGHTS", live
            ), mock.patch.object(dc, "BON_WINNER_WEIGHTS", winner):
                self.assertEqual(dc.find_weights(None), live)

    def test_falls_back_to_bon_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            winner = root / "runs" / "bon-winner" / "weights" / "best.pt"
            scratch = root / "runs" / "vial-99" / "weights" / "best.pt"
            for path in (winner, scratch):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            live = root / "runs" / "vial-live" / "weights" / "best.pt"
            r1 = root / "runs" / "train-race" / "r1-winner" / "weights" / "best.pt"
            with mock.patch.object(dc, "ROOT", root), mock.patch.object(
                dc, "LIVE_WEIGHTS", live
            ), mock.patch.object(dc, "R1_WINNER_WEIGHTS", r1), mock.patch.object(
                dc, "BON_WINNER_WEIGHTS", winner
            ):
                self.assertEqual(dc.find_weights(None), winner)

    def test_prefers_r1_over_bon_when_no_live(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r1 = root / "runs" / "train-race" / "r1-winner" / "weights" / "best.pt"
            bon = root / "runs" / "bon-winner" / "weights" / "best.pt"
            for path in (r1, bon):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            live = root / "runs" / "vial-live" / "weights" / "best.pt"
            with mock.patch.object(dc, "ROOT", root), mock.patch.object(
                dc, "LIVE_WEIGHTS", live
            ), mock.patch.object(dc, "R1_WINNER_WEIGHTS", r1), mock.patch.object(
                dc, "BON_WINNER_WEIGHTS", bon
            ):
                self.assertEqual(dc.find_weights(None), r1)


if __name__ == "__main__":
    unittest.main()
