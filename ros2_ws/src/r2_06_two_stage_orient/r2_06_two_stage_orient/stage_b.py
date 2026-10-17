"""Stage B — orientation class {upright, sideways, other} + box refine.

Uses white-cap location inside/near the proposal plus aspect ratio (and
optional depth-hole elongation) to pick a pose class, then expands/swaps the
AABB so lying vials cover lid→cup along the major axis.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .stage_a import white_cap_mask
from .types import OrientedDet, Proposal, clamp_box, nms_xywh


def _cap_centroid_in_roi(
    mask: np.ndarray, x: int, y: int, w: int, h: int
) -> Optional[Tuple[float, float, float, float]]:
    """Return (cx, cy, cw, ch) of dominant white blob intersecting the ROI."""
    H, W = mask.shape[:2]
    x0 = max(0, x - int(0.35 * w))
    y0 = max(0, y - int(0.35 * h))
    x1 = min(W, x + w + int(0.35 * w))
    y1 = min(H, y + h + int(0.35 * h))
    roi = mask[y0:y1, x0:x1]
    if roi.size == 0 or cv2.countNonZero(roi) < 20:
        return None
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 20:
        return None
    bx, by, bw, bh = cv2.boundingRect(cnt)
    cx = x0 + bx + bw * 0.5
    cy = y0 + by + bh * 0.5
    return cx, cy, float(bw), float(bh)


def _depth_axis_hint(
    depth_mm: Optional[np.ndarray], x: int, y: int, w: int, h: int
) -> Optional[str]:
    """Return 'horizontal' / 'vertical' from clear-cup depth holes, or None."""
    if depth_mm is None:
        return None
    H, W = depth_mm.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = depth_mm[y0:y1, x0:x1]
    invalid = ((patch <= 1.0) | (patch > 5000.0)).astype(np.uint8) * 255
    if cv2.countNonZero(invalid) < 40:
        return None
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    invalid = cv2.morphologyEx(invalid, cv2.MORPH_CLOSE, k)
    ys, xs = np.where(invalid > 0)
    if len(xs) < 30:
        return None
    cov = np.cov(np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=0))
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    if abs(axis[0]) >= abs(axis[1]):
        return "horizontal"
    return "vertical"


def classify_orientation(
    bgr: np.ndarray,
    prop: Proposal,
    *,
    depth_mm: Optional[np.ndarray] = None,
    cap_mask: Optional[np.ndarray] = None,
) -> Tuple[str, dict]:
    """Return (orientation, debug_info)."""
    x, y, w, h = prop.as_xywh()
    aspect = h / float(max(1, w))  # >1 tall, <1 wide
    mask = cap_mask if cap_mask is not None else white_cap_mask(bgr)
    cap = _cap_centroid_in_roi(mask, x, y, w, h)
    depth_hint = _depth_axis_hint(depth_mm, x, y, w, h)

    scores = {"upright": 0.0, "sideways": 0.0, "other": 0.0}
    info: dict = {"aspect": aspect, "depth_hint": depth_hint, "cap": None}

    # Aspect prior
    if aspect >= 1.15:
        scores["upright"] += 0.45
    elif aspect <= 0.75:
        scores["sideways"] += 0.45
    else:
        scores["upright"] += 0.15
        scores["sideways"] += 0.15
        scores["other"] += 0.10

    if cap is not None:
        cx, cy, cw, ch = cap
        info["cap"] = {"cx": cx, "cy": cy, "cw": cw, "ch": ch}
        # Relative position of cap inside proposal.
        rx = (cx - x) / float(max(1, w))
        ry = (cy - y) / float(max(1, h))
        info["cap_rel"] = (rx, ry)
        # Near top or bottom → upright; near left/right → sideways.
        edge_y = min(ry, 1.0 - ry)
        edge_x = min(rx, 1.0 - rx)
        if ry < 0.38 or ry > 0.62:
            scores["upright"] += 0.40
        if rx < 0.38 or rx > 0.62:
            scores["sideways"] += 0.40
        if edge_x > 0.28 and edge_y > 0.28:
            # Cap near center — often top-down / other.
            scores["other"] += 0.35
        # Cap itself elongated sideways strengthens sideways class.
        if cw >= ch * 1.25:
            scores["sideways"] += 0.15
        if ch >= cw * 1.25:
            scores["upright"] += 0.10
    else:
        scores["other"] += 0.20

    if depth_hint == "horizontal":
        scores["sideways"] += 0.25
    elif depth_hint == "vertical":
        scores["upright"] += 0.20

    # Source bias from Stage A.
    src = prop.source or ""
    if "sideways" in src or src.startswith("cap_left") or src.startswith("cap_right") or "rot90" in src or "rot270" in src:
        scores["sideways"] += 0.15
    if src.startswith("cap_down") or src.startswith("cap_up") or src == "yolo":
        scores["upright"] += 0.08

    orient = max(scores, key=scores.get)
    # Tie-break with aspect when close.
    ranked = sorted(scores.values(), reverse=True)
    if len(ranked) >= 2 and ranked[0] - ranked[1] < 0.08:
        if aspect >= 1.05:
            orient = "upright"
        elif aspect <= 0.85:
            orient = "sideways"
        else:
            orient = "other"
    info["scores"] = scores
    return orient, info


def refine_box(
    bgr: np.ndarray,
    prop: Proposal,
    orientation: str,
    *,
    cap_mask: Optional[np.ndarray] = None,
    info: Optional[dict] = None,
) -> Tuple[int, int, int, int]:
    """Expand / swap proposal so the AABB covers lid→cup for the pose class."""
    H, W = bgr.shape[:2]
    x, y, w, h = prop.as_xywh()
    # Trusted Stage-A sources already cover lid→cup — light pad only.
    if prop.source in ("sideways_I", "yolo") or (
        prop.source or ""
    ).startswith("yolo_rot"):
        pad = 2
        return clamp_box(x - pad, y - pad, w + 2 * pad, h + 2 * pad, W, H)
    mask = cap_mask if cap_mask is not None else white_cap_mask(bgr)
    cap = None
    if info and info.get("cap"):
        c = info["cap"]
        cap = (c["cx"], c["cy"], c["cw"], c["ch"])
    else:
        cap = _cap_centroid_in_roi(mask, x, y, w, h)

    if orientation == "sideways":
        # Already wide enough — light pad (avoid blowing past GT).
        if w >= h * 1.05:
            pad = 1
            return clamp_box(x - pad, y - pad, w + 2 * pad, h + 2 * pad, W, H)
        # Tall proposal → swap/expand horizontally from cap.
        diam = float(min(w, h))
        new_w = int(max(w, diam * 1.55))
        new_h = int(max(h * 0.80, diam * 0.95))
        if cap is not None:
            cx, cy, cw, ch = cap
            if cx <= x + 0.45 * w:
                return clamp_box(int(cx - 0.25 * cw), int(cy - 0.5 * new_h), new_w, new_h, W, H)
            return clamp_box(
                int(cx + 0.25 * cw - new_w), int(cy - 0.5 * new_h), new_w, new_h, W, H
            )
        return clamp_box(int(x + w / 2 - new_w / 2), int(y + h / 2 - new_h / 2), new_w, new_h, W, H)

    if orientation == "upright":
        # Already tall enough — do not overshoot tight GT (synth upright boxes).
        if h >= w * 1.05:
            pad = 1
            return clamp_box(x - pad, y - pad, w + 2 * pad, h + 2 * pad, W, H)
        # Wide proposal → grow height from cap (swap/expand).
        diam = float(min(w, h))
        new_h = int(max(h, diam * 1.55))
        new_w = int(max(w * 0.80, diam * 0.95))
        if cap is not None:
            cx, cy, cw, ch = cap
            if cy <= y + 0.45 * h:
                return clamp_box(int(cx - 0.5 * new_w), int(cy - 0.25 * ch), new_w, new_h, W, H)
            return clamp_box(
                int(cx - 0.5 * new_w), int(cy + 0.25 * ch - new_h), new_w, new_h, W, H
            )
        return clamp_box(x + (w - new_w) // 2, y + (h - new_h) // 2, new_w, new_h, W, H)

    # other — mild pad, keep aspect
    pad_x = int(0.06 * w)
    pad_y = int(0.06 * h)
    return clamp_box(x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y, W, H)


class StageBOrient:
    """Classify + refine a list of Stage-A proposals."""

    def __init__(
        self,
        *,
        min_conf: float = 0.18,
        nms_iou: float = 0.45,
        max_dets: int = 5,
        drop_other: bool = False,
    ) -> None:
        self.min_conf = float(min_conf)
        self.nms_iou = float(nms_iou)
        self.max_dets = int(max_dets)
        self.drop_other = bool(drop_other)

    def refine(
        self,
        bgr: np.ndarray,
        proposals: Sequence[Proposal],
        *,
        depth_mm: Optional[np.ndarray] = None,
    ) -> List[OrientedDet]:
        if not proposals:
            return []
        cap_mask = white_cap_mask(bgr)
        dets: List[OrientedDet] = []
        for prop in proposals:
            orient, info = classify_orientation(
                bgr, prop, depth_mm=depth_mm, cap_mask=cap_mask
            )
            # Trust specialized Stage-A sources for class labels.
            if prop.source == "sideways_I":
                orient = "sideways"
            elif prop.source == "yolo":
                # Native (unrotated) YOLO is upright-biased on this dataset.
                if orient == "other":
                    orient = "upright"
            if self.drop_other and orient == "other":
                continue
            rx, ry, rw, rh = refine_box(
                bgr, prop, orient, cap_mask=cap_mask, info=info
            )
            # Confidence: Stage-A score + orientation margin.
            scores = info.get("scores") or {}
            ranked = sorted(scores.values(), reverse=True)
            margin = (ranked[0] - ranked[1]) if len(ranked) >= 2 else ranked[0]
            conf = float(np.clip(0.55 * prop.score + 0.45 * (0.4 + margin), 0.05, 0.99))
            # Penalize ambiguous "other" from weak classical caps (FP on empties).
            if orient == "other" and not (prop.source or "").startswith("yolo"):
                conf *= 0.65
            if conf < self.min_conf:
                continue
            dets.append(
                OrientedDet(
                    x=rx,
                    y=ry,
                    w=rw,
                    h=rh,
                    conf=conf,
                    orientation=orient,
                    source=prop.source,
                    stage_a_score=float(prop.score),
                )
            )

        if not dets:
            return []
        boxes = [d.as_xywh() for d in dets]
        scores = [d.conf for d in dets]
        keep = nms_xywh(boxes, scores, iou_thr=self.nms_iou, max_dets=self.max_dets)
        return [dets[i] for i in keep]
