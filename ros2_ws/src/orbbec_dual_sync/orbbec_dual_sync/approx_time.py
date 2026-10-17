"""Approximate-time pairing for stamped frames (ROS message_filters mimic).

Pure Python, no ROS required. Matches the spirit of
``message_filters.ApproximateTimeSynchronizer``: keep per-stream queues and
emit a pair when |t_a - t_b| <= slop, dropping stale unmatched samples.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Generic, Iterable, Iterator, Optional, Tuple, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Stamped(Generic[T]):
    """One sample with a monotonic or wall-clock stamp in seconds."""

    stamp_s: float
    data: T


@dataclass(frozen=True)
class SyncedPair(Generic[T]):
    a: Stamped[T]
    b: Stamped[T]

    @property
    def dt_s(self) -> float:
        return abs(self.a.stamp_s - self.b.stamp_s)

    @property
    def stamp_s(self) -> float:
        return 0.5 * (self.a.stamp_s + self.b.stamp_s)


class ApproximateTimeSynchronizer2(Generic[T]):
    """Two-stream ApproximateTime sync (USB cameras; no hardware genlock)."""

    def __init__(self, *, slop_s: float = 0.05, queue_size: int = 10) -> None:
        if slop_s < 0:
            raise ValueError("slop_s must be >= 0")
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        self.slop_s = float(slop_s)
        self.queue_size = int(queue_size)
        self._qa: Deque[Stamped[T]] = deque(maxlen=self.queue_size)
        self._qb: Deque[Stamped[T]] = deque(maxlen=self.queue_size)
        self.dropped_stale = 0
        self.emitted = 0
        self.rejected_dt = 0

    def clear(self) -> None:
        self._qa.clear()
        self._qb.clear()

    def add_a(self, stamp_s: float, data: T) -> Optional[SyncedPair[T]]:
        self._qa.append(Stamped(float(stamp_s), data))
        return self._try_match()

    def add_b(self, stamp_s: float, data: T) -> Optional[SyncedPair[T]]:
        self._qb.append(Stamped(float(stamp_s), data))
        return self._try_match()

    def add(self, stream: str, stamp_s: float, data: T) -> Optional[SyncedPair[T]]:
        key = stream.lower()
        if key in ("a", "0", "cam0", "left"):
            return self.add_a(stamp_s, data)
        if key in ("b", "1", "cam1", "right"):
            return self.add_b(stamp_s, data)
        raise ValueError(f"Unknown stream {stream!r}; use a/b or cam0/cam1")

    def _try_match(self) -> Optional[SyncedPair[T]]:
        while self._qa and self._qb:
            a = self._qa[0]
            b = self._qb[0]
            dt = abs(a.stamp_s - b.stamp_s)
            if dt <= self.slop_s:
                self._qa.popleft()
                self._qb.popleft()
                pair = SyncedPair(a=a, b=b)
                self.emitted += 1
                return pair
            # Drop the older head; it can never match a newer peer within slop
            # once the other stream has already moved past it.
            if a.stamp_s < b.stamp_s:
                self._qa.popleft()
                self.dropped_stale += 1
            else:
                self._qb.popleft()
                self.dropped_stale += 1
            self.rejected_dt += 1
        return None


def pair_iter(
    stream_a: Iterable[Stamped[T]],
    stream_b: Iterable[Stamped[T]],
    *,
    slop_s: float = 0.05,
    queue_size: int = 10,
) -> Iterator[SyncedPair[T]]:
    """Offline helper: feed two sorted/unsorted stamped iters through the syncer."""
    sync = ApproximateTimeSynchronizer2(slop_s=slop_s, queue_size=queue_size)
    ia = iter(stream_a)
    ib = iter(stream_b)
    next_a: Optional[Stamped[T]] = None
    next_b: Optional[Stamped[T]] = None
    done_a = done_b = False

    def _pull_a() -> None:
        nonlocal next_a, done_a
        if done_a:
            return
        try:
            next_a = next(ia)
        except StopIteration:
            next_a = None
            done_a = True

    def _pull_b() -> None:
        nonlocal next_b, done_b
        if done_b:
            return
        try:
            next_b = next(ib)
        except StopIteration:
            next_b = None
            done_b = True

    _pull_a()
    _pull_b()
    while next_a is not None or next_b is not None or sync._qa or sync._qb:
        if next_a is not None and (next_b is None or next_a.stamp_s <= next_b.stamp_s):
            pair = sync.add_a(next_a.stamp_s, next_a.data)
            _pull_a()
        elif next_b is not None:
            pair = sync.add_b(next_b.stamp_s, next_b.data)
            _pull_b()
        else:
            break
        if pair is not None:
            yield pair


def stamp_msg_to_seconds(stamp) -> float:
    """Convert builtin_interfaces/Time or object with sec/nanosec to float seconds."""
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    return float(sec) + 1e-9 * float(nanosec)


def seconds_to_stamp_parts(stamp_s: float) -> Tuple[int, int]:
    sec = int(stamp_s)
    nanosec = int(round((stamp_s - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    if nanosec < 0:
        sec -= 1
        nanosec += 1_000_000_000
    return sec, nanosec
