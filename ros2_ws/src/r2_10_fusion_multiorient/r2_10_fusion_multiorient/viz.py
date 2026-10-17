"""OpenCV helpers for annotated detection previews."""

from __future__ import annotations

import cv2
import numpy as np

from .types import Det

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def detection_caption(d: Det) -> str:
    """Human-readable box caption: vial and confidence."""
    return f"vial {d.conf:.2f}"


def draw_box_label(
    out: np.ndarray,
    x1: int,
    y1: int,
    text: str,
    *,
    box_color: tuple[int, int, int],
    scale: float = 0.55,
    thickness: int = 2,
) -> None:
    """Draw text above (x1, y1) with a filled background for contrast."""
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    pad_x, pad_y = 4, 3
    top = max(0, y1 - th - baseline - pad_y * 2 - 2)
    left = max(0, x1)
    right = min(out.shape[1] - 1, left + tw + pad_x * 2)
    bottom = min(out.shape[0] - 1, top + th + baseline + pad_y * 2)
    cv2.rectangle(out, (left, top), (right, bottom), (0, 0, 0), -1)
    cv2.rectangle(out, (left, top), (right, bottom), box_color, 1)
    text_y = top + th + pad_y
    cv2.putText(
        out,
        text,
        (left + pad_x, text_y),
        _FONT,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def annotate_detections(bgr: np.ndarray, dets) -> np.ndarray:
    out = bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(round(v)) for v in d.xyxy]
        color = {
            "upright": (40, 200, 40),
            "sideways": (40, 160, 255),
            "topdown": (220, 160, 40),
        }.get(d.orientation, (200, 200, 40))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        draw_box_label(
            out,
            x1,
            y1,
            detection_caption(d),
            box_color=color,
        )
    return out
