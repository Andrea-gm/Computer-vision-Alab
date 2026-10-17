"""Shared detection types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class Det:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int = 0
    label: str = "vial"
    source: str = ""

    def as_xyxy(self):
        return (self.x1, self.y1, self.x2, self.y2)


def det_to_dict(d: Det) -> Dict:
    return {
        "x1": float(d.x1),
        "y1": float(d.y1),
        "x2": float(d.x2),
        "y2": float(d.y2),
        "confidence": float(d.conf),
        "class_id": int(d.cls_id),
        "label": d.label,
        "source": d.source,
    }


def box_iou(a: Det, b: Det) -> float:
    ax1, ay1, ax2, ay2 = a.as_xyxy()
    bx1, by1, bx2, by2 = b.as_xyxy()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (aa + bb - inter + 1e-9)
