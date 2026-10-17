"""Unit tests for pure-Python ApproximateTime (no ROS required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orbbec_dual_sync.approx_time import ApproximateTimeSynchronizer2, pair_iter, Stamped


def test_pair_within_slop():
    sync = ApproximateTimeSynchronizer2(slop_s=0.05, queue_size=10)
    assert sync.add_a(1.000, "a0") is None
    pair = sync.add_b(1.030, "b0")
    assert pair is not None
    assert pair.a.data == "a0"
    assert pair.b.data == "b0"
    assert pair.dt_s == pytest.approx(0.03)


def test_drop_stale_when_beyond_slop():
    sync = ApproximateTimeSynchronizer2(slop_s=0.05, queue_size=10)
    sync.add_a(1.0, "old")
    # Peer arrives 200 ms later — old head must be dropped, no emit yet.
    assert sync.add_b(1.2, "new") is None
    assert sync.dropped_stale >= 1
    # Fresh a near b should match.
    pair = sync.add_a(1.21, "fresh")
    assert pair is not None
    assert pair.a.data == "fresh"
    assert pair.b.data == "new"


def test_pair_iter():
    a = [Stamped(0.0, 0), Stamped(0.1, 1), Stamped(0.2, 2)]
    b = [Stamped(0.02, 10), Stamped(0.19, 20)]
    pairs = list(pair_iter(a, b, slop_s=0.05, queue_size=10))
    assert len(pairs) >= 1
    assert pairs[0].dt_s <= 0.05
