"""Shared detection dataclass (no classical upright-only fallback)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Sequence

import cv2
import numpy as np


@dataclass
class Det:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int = 0
    label: str = "vial"
    source: str = "yolo"

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def aspect_hw(self) -> float:
        w = self.width
        return self.height / w if w > 1e-6 else 0.0


def det_to_dict(d: Det) -> dict:
    return asdict(d)


def annotate(bgr: np.ndarray, dets: Sequence[Det], *, color=(0, 220, 80)) -> np.ndarray:
    out = bgr.copy()
    for d in dets:
        p1 = (int(d.x1), int(d.y1))
        p2 = (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, color, 2)
        orient = "side" if d.aspect_hw < 0.95 else "up"
        tag = f"{d.label} {d.conf:.2f} [{orient}]"
        cv2.putText(
            out,
            tag,
            (p1[0], max(16, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def box_iou(a: Det, b: Det) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = a.width * a.height
    area_b = b.width * b.height
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0
