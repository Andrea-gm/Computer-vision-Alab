"""Classical white-cap heuristic for unlabeled / no-weights demo.

Finds a bright low-saturation blob (opaque white screw cap) and expands a
rough full-vial box downward. Precision is weaker than fine-tuned YOLO —
intended only as a demo path when weights are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Det:
    """Normalized detection: xyxy pixels + confidence + class name."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int = 0
    label: str = "vial"
    source: str = "classical"


def _white_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((v >= 185) & (s <= 55)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def detect_vial_classical(
    bgr: np.ndarray,
    *,
    max_dets: int = 3,
) -> List[Det]:
    """Return up to ``max_dets`` vial-ish boxes from white-cap blobs."""
    h, w = bgr.shape[:2]
    mask = _white_mask(bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = float(h * w)
    scored: List[Tuple[float, Det]] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        frac = area / max(area_img, 1.0)
        if frac < 0.0008 or frac > 0.18:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 8 or bh < 6:
            continue
        # Expand downward to cover clear cup (cap is typically top ~25–40%).
        expand_h = int(max(bh * 2.4, bw * 1.15))
        x1 = max(0, x - int(0.15 * bw))
        y1 = max(0, y - int(0.15 * bh))
        x2 = min(w - 1, x + bw + int(0.15 * bw))
        y2 = min(h - 1, y + expand_h)
        box_h = max(1, y2 - y1)
        box_w = max(1, x2 - x1)
        aspect = box_h / float(box_w)
        if aspect < 0.7 or aspect > 2.4:
            continue
        # Score: prefer mid-sized bright caps with upright-ish aspect.
        score = float(np.clip(0.35 + 0.4 * (1.0 - abs(aspect - 1.35) / 1.0), 0.2, 0.75))
        scored.append(
            (
                score,
                Det(float(x1), float(y1), float(x2), float(y2), score, source="classical"),
            )
        )
    scored.sort(key=lambda t: t[0], reverse=True)
    return [d for _, d in scored[:max_dets]]


def annotate(bgr: np.ndarray, dets: List[Det], *, color=(40, 180, 255)) -> np.ndarray:
    out = bgr.copy()
    for d in dets:
        p1 = (int(d.x1), int(d.y1))
        p2 = (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, color, 2)
        tag = f"{d.label} {d.conf:.2f} [{d.source}]"
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
