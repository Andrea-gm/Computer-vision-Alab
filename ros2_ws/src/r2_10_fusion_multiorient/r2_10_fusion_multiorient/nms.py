"""Orientation-aware NMS for multi-pose vial fusion."""

from __future__ import annotations

from typing import List, Sequence

from .types import Det, box_iou, classify_orientation


def orientation_aware_nms(
    dets: Sequence[Det],
    *,
    iou_thr: float = 0.45,
    max_det: int = 5,
    prefer_sources: Sequence[str] | None = None,
) -> List[Det]:
    """NMS that lightly prefers orientation-consistent, higher-conf boxes.

    Same-orientation overlaps suppress at ``iou_thr``. Cross-orientation
    overlaps require higher IoU (0.60) so a good sideways box is not killed
    by a weak upright fragment covering the same lid.
    """
    if not dets:
        return []
    prefer = tuple(prefer_sources or ())

    def sort_key(d: Det) -> tuple:
        src_bonus = 0.0
        for i, pref in enumerate(prefer):
            if pref in d.source:
                src_bonus = 1.0 - 0.05 * i
                break
        orient = d.orientation if d.orientation != "unknown" else classify_orientation(d)
        # Mild prior: sideways classical and YOLO TTA 90/270 are valuable.
        orient_bonus = 0.02 if orient == "sideways" and "classical" in d.source else 0.0
        return (d.conf + 0.03 * src_bonus + orient_bonus, d.conf)

    ordered = sorted(dets, key=sort_key, reverse=True)
    kept: List[Det] = []
    for d in ordered:
        d.orientation = d.orientation if d.orientation != "unknown" else classify_orientation(d)
        suppress = False
        for k in kept:
            iou = box_iou(d, k)
            same = d.orientation == k.orientation
            thr = iou_thr if same else max(iou_thr, 0.60)
            if iou >= thr:
                suppress = True
                break
        if not suppress:
            kept.append(d)
        if len(kept) >= max_det:
            break
    return kept
