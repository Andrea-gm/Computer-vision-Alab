"""Shared detection types for the two-stage pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple


ORIENTATIONS = ("upright", "sideways", "other")


@dataclass
class Proposal:
    """Stage-A axis-aligned box proposal (xywh pixels)."""

    x: int
    y: int
    w: int
    h: int
    score: float
    source: str = "unknown"

    def as_xywh(self) -> Tuple[int, int, int, int]:
        return int(self.x), int(self.y), int(self.w), int(self.h)

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        return int(self.x), int(self.y), int(self.x + self.w), int(self.y + self.h)


@dataclass
class OrientedDet:
    """Stage-B detection with orientation class and refined box."""

    x: int
    y: int
    w: int
    h: int
    conf: float
    orientation: str  # upright | sideways | other
    source: str = "two_stage"
    stage_a_score: float = 0.0
    cls_id: int = 0
    label: str = "vial"

    def as_xywh(self) -> Tuple[int, int, int, int]:
        return int(self.x), int(self.y), int(self.w), int(self.h)

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return float(self.x), float(self.y), float(self.x + self.w), float(self.y + self.h)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def box_iou_xywh(a, b) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0 else inter / union


def clamp_box(
    x: int, y: int, w: int, h: int, frame_w: int, frame_h: int
) -> Tuple[int, int, int, int]:
    x = max(0, min(int(x), frame_w - 1))
    y = max(0, min(int(y), frame_h - 1))
    w = max(1, min(int(w), frame_w - x))
    h = max(1, min(int(h), frame_h - y))
    return x, y, w, h


def nms_xywh(
    boxes: list,
    scores: list,
    iou_thr: float = 0.45,
    max_dets: int = 8,
) -> list:
    """Greedy NMS; returns kept indices."""
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list = []
    while order and len(keep) < max_dets:
        i = order.pop(0)
        keep.append(i)
        order = [
            j
            for j in order
            if box_iou_xywh(boxes[i], boxes[j]) < iou_thr
        ]
    return keep
