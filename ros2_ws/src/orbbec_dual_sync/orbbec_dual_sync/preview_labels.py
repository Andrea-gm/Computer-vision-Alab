"""Contrast-friendly vial labels on detection boxes in sync preview."""

from __future__ import annotations

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX
# Keep in sync with robotics/depthcamera.py range bands.
CLOSE_RANGE_MM = 500.0
MID_RANGE_MM = 1000.0


def range_band_label(distance_cm: float | None) -> str:
    if distance_cm is None:
        return ""
    distance_mm = float(distance_cm) * 10.0
    if distance_mm <= CLOSE_RANGE_MM:
        return "close"
    if distance_mm <= MID_RANGE_MM:
        return "mid"
    return "far"


def vial_box_caption(
    _detection_source: str = "",
    *,
    distance_cm: float | None = None,
) -> str:
    band = range_band_label(distance_cm)
    return f"vial {band}".strip() if band else "vial"


def draw_vial_box(
    img: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    detection_source: str = "",
    distance_cm: float | None = None,
    box_color: tuple[int, int, int] = (0, 220, 0),
    show_label: bool = False,
) -> None:
    """Box outline only. Captions such as 'vial' / 'vial close' stay off unless show_label."""
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)
    if not show_label:
        return
    text = vial_box_caption(detection_source, distance_cm=distance_cm)
    scale, thickness = 0.55, 2
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    pad_x, pad_y = 4, 3
    top = max(0, y - th - baseline - pad_y * 2 - 2)
    left = max(0, x)
    right = min(img.shape[1] - 1, left + tw + pad_x * 2)
    bottom = min(img.shape[0] - 1, top + th + baseline + pad_y * 2)
    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 0), -1)
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 1)
    cv2.putText(
        img,
        text,
        (left + pad_x, top + th + pad_y),
        _FONT,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
