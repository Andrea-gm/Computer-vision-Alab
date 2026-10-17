"""Shared detection types."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List


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
    def xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, max(0.0, self.x2 - self.x1), max(0.0, self.y2 - self.y1))


@dataclass
class InferStats:
    backend: str
    model_path: str
    mode: str
    inference_ms: float
    num_dets: int
    num_slices: int = 0
    num_passes: int = 0


def det_to_dict(d: Det) -> dict:
    return asdict(d)


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
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def nms(dets: List[Det], iou_thr: float) -> List[Det]:
    if not dets:
        return []
    ordered = sorted(dets, key=lambda d: d.conf, reverse=True)
    keep: List[Det] = []
    while ordered:
        best = ordered.pop(0)
        keep.append(best)
        ordered = [d for d in ordered if box_iou(best, d) < iou_thr]
    return keep


def annotate(bgr, dets: List[Det], color=(0, 200, 120)):
    import cv2
    import numpy as np

    out = bgr.copy()
    for d in dets:
        p1 = (int(round(d.x1)), int(round(d.y1)))
        p2 = (int(round(d.x2)), int(round(d.y2)))
        cv2.rectangle(out, p1, p2, color, 2)
        tag = f"{d.label} {d.conf:.2f}"
        cv2.putText(
            out,
            tag,
            (p1[0], max(14, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out
