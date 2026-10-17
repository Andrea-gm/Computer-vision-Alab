"""Classical white-cap localization for vial lids (upright strips + end-caps)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class CapBlob:
    x: int
    y: int
    w: int
    h: int
    cx: float
    cy: float
    area: float
    score: float
    kind: str  # side_on | end_cap | top_downish

    @property
    def xywh(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def diameter(self) -> float:
        return float(max(self.w, self.h))


def white_cap_mask(
    bgr: np.ndarray,
    *,
    s_max: int = 55,
    v_min: int = 200,
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    best = ((val >= v_min) & (sat <= s_max)).astype(np.uint8) * 255
    best_frac = float(best.mean() / 255.0)
    if not (0.0005 <= best_frac <= 0.25):
        for vmin in (185, 170, 210, 220, 230):
            trial = ((val >= vmin) & (sat <= s_max)).astype(np.uint8) * 255
            frac = float(trial.mean() / 255.0)
            if 0.0005 <= frac <= 0.25 and abs(frac - 0.04) < abs(best_frac - 0.04):
                best, best_frac = trial, frac
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    best = cv2.morphologyEx(best, cv2.MORPH_OPEN, k)
    best = cv2.morphologyEx(best, cv2.MORPH_CLOSE, k)
    # Keep upright wide lids and sideways end-caps as compact blobs.
    best = cv2.bitwise_or(
        best,
        cv2.morphologyEx(
            best, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
        ),
    )
    best = cv2.bitwise_or(
        best,
        cv2.morphologyEx(
            best, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
        ),
    )
    return best


def find_cap_blobs(
    bgr: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    s_max: int = 55,
    v_min: int = 200,
    min_area: int = 35,
    max_area_frac: float = 0.22,
    roi_xyxy: Optional[Tuple[int, int, int, int]] = None,
) -> List[CapBlob]:
    h, w = bgr.shape[:2]
    if mask is None:
        mask = white_cap_mask(bgr, s_max=s_max, v_min=v_min)
    work = mask
    ox = oy = 0
    if roi_xyxy is not None:
        x1, y1, x2, y2 = [int(v) for v in roi_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return []
        work = mask[y1:y2, x1:x2]
        ox, oy = x1, y1
    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max_area_frac * w * h
    min_end = max(8, int(min(w, h) * 0.018))
    min_width = max(12, int(w * 0.025))
    out: List[CapBlob] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        x += ox
        y += oy
        if bw > 0.75 * w or bh > 0.75 * h:
            continue
        # Huge bright slabs are walls / paper, not vial lids.
        if area > 0.08 * w * h:
            continue
        aspect = bw / max(bh, 1)
        side_on = bw >= min_width and 1.55 <= aspect <= 12.0 and bh >= 3
        top_downish = (
            bw >= min_end
            and bh >= min_end
            and 0.75 <= aspect < 1.55
            and bh >= max(8, int(bw * 0.5))
        )
        end_cap = (
            bw >= min_end
            and bh >= min_end
            and 0.22 <= aspect <= 1.55
            and max(bw, bh) <= min(w, h) * 0.42
        )
        if not (side_on or top_downish or end_cap):
            continue
        m = cv2.moments(cnt)
        if m["m00"] > 1e-3:
            cx = float(m["m10"] / m["m00"]) + ox
            cy = float(m["m01"] / m["m00"]) + oy
        else:
            cx = x + bw * 0.5
            cy = y + bh * 0.5
        kind = "side_on" if side_on else ("end_cap" if end_cap else "top_downish")
        # Prefer compact bright blobs; penalize huge wall-like regions.
        compactness = area / float(max(bw * bh, 1))
        score = compactness * min(1.0, area / 400.0)
        out.append(
            CapBlob(
                x=x,
                y=y,
                w=bw,
                h=bh,
                cx=cx,
                cy=cy,
                area=area,
                score=float(score),
                kind=kind,
            )
        )
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def localize_cap_in_box(
    bgr: np.ndarray,
    xyxy: Sequence[float],
    *,
    s_max: int = 55,
    v_min: int = 200,
) -> Optional[CapBlob]:
    """Pick the best white-cap blob inside a YOLO / GT AABB (weak keypoint label)."""
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    blobs = find_cap_blobs(
        bgr, s_max=s_max, v_min=v_min, roi_xyxy=(x1, y1, x2, y2), min_area=20
    )
    if blobs:
        return blobs[0]
    # Fallback: brightest low-sat region centroid in the box.
    h, w = bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    score = hsv[:, :, 2].astype(np.float32) - hsv[:, :, 1].astype(np.float32)
    # Prefer ends of the box for upright/sideways caps.
    yy, xx = np.mgrid[0 : roi.shape[0], 0 : roi.shape[1]]
    bh, bw = float(roi.shape[0]), float(roi.shape[1])
    if bh >= bw:
        end_bias = np.maximum(1.0 - yy / max(bh * 0.35, 1.0), yy / max(bh, 1.0))
    else:
        end_bias = np.maximum(1.0 - xx / max(bw * 0.35, 1.0), xx / max(bw, 1.0))
    score = score * (0.55 + 0.45 * end_bias)
    flat = int(np.argmax(score))
    ly, lx = divmod(flat, score.shape[1])
    cx, cy = float(x1 + lx), float(y1 + ly)
    diam = float(min(bw, bh) * 0.45)
    return CapBlob(
        x=int(cx - diam * 0.5),
        y=int(cy - diam * 0.5),
        w=max(3, int(diam)),
        h=max(3, int(diam)),
        cx=cx,
        cy=cy,
        area=diam * diam,
        score=0.2,
        kind="weak",
    )
