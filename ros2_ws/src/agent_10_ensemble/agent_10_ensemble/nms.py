"""Non-maximum suppression for fused vial proposals."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .types import Detection


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def nms(detections: list[Detection], iou_thresh: float = 0.45) -> list[Detection]:
    """Greedy NMS ranked by score (descending)."""
    if not detections:
        return []
    order = sorted(range(len(detections)), key=lambda i: detections[i].score, reverse=True)
    keep: list[Detection] = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        best = detections[i]
        keep.append(best)
        for j in order:
            if j == i or j in suppressed:
                continue
            if box_iou(best.xyxy, detections[j].xyxy) >= iou_thresh:
                suppressed.add(j)
    return keep


def soft_nms(
    detections: list[Detection],
    iou_thresh: float = 0.45,
    sigma: float = 0.5,
    score_thresh: float = 0.05,
) -> list[Detection]:
    """Gaussian soft-NMS; useful when cap + YOLO boxes overlap imperfectly."""
    if not detections:
        return []
    boxes = np.array([d.xyxy for d in detections], dtype=np.float64)
    scores = np.array([d.score for d in detections], dtype=np.float64)
    sources = [d.source for d in detections]
    remaining = list(range(len(detections)))
    keep: list[Detection] = []
    while remaining:
        max_idx = max(remaining, key=lambda i: scores[i])
        remaining.remove(max_idx)
        if scores[max_idx] < score_thresh:
            continue
        keep.append(
            Detection(
                x1=float(boxes[max_idx, 0]),
                y1=float(boxes[max_idx, 1]),
                x2=float(boxes[max_idx, 2]),
                y2=float(boxes[max_idx, 3]),
                score=float(scores[max_idx]),
                source=sources[max_idx],
            )
        )
        survivors: list[int] = []
        for j in remaining:
            iou = box_iou(boxes[max_idx], boxes[j])
            if iou >= iou_thresh:
                scores[j] *= float(np.exp(-(iou * iou) / sigma))
            if scores[j] >= score_thresh:
                survivors.append(j)
        remaining = survivors
    return keep
