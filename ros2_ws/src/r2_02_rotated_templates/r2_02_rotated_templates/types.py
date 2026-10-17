"""Shared detection dataclass + drawing helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Sequence

import cv2


@dataclass
class Det:
    """Axis-aligned detection: xyxy pixels + confidence + class name."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int = 0
    label: str = "vial"
    source: str = "template"
    angle_deg: float = 0.0
    orientation: str = "unknown"  # upright | sideways | diagonal


def annotate(bgr, dets: Sequence[Det], *, color=(40, 200, 80)) -> object:
    out = bgr.copy()
    for d in dets:
        p1 = (int(d.x1), int(d.y1))
        p2 = (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, color, 2)
        tag = f"{d.label} {d.conf:.2f} {d.orientation}@{d.angle_deg:.0f}"
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


def nms_dets(dets: List[Det], iou_thr: float, max_keep: int) -> List[Det]:
    ordered = sorted(dets, key=lambda d: d.conf, reverse=True)
    kept: List[Det] = []
    for d in ordered:
        if all(box_iou(d, k) < iou_thr for k in kept):
            kept.append(d)
        if len(kept) >= max_keep:
            break
    return kept


def orientation_from_angle(angle_deg: float) -> str:
    """Map template rotation angle to coarse pose label."""
    a = abs(float(angle_deg)) % 180.0
    # Near 0 or 180 → upright template orientation; near 90 → sideways.
    dist_up = min(a, 180.0 - a)
    dist_side = abs(a - 90.0)
    if dist_side <= 25.0:
        return "sideways"
    if dist_up <= 25.0:
        return "upright"
    return "diagonal"
