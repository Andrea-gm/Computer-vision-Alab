"""SAHI-style overlapping tile generation and box remapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .types import Det


@dataclass(frozen=True)
class Slice:
    """Axis-aligned crop in full-frame coordinates."""

    x: int
    y: int
    w: int
    h: int

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


def generate_slices(
    width: int,
    height: int,
    slice_w: int,
    slice_h: int,
    overlap_ratio: float = 0.25,
) -> List[Slice]:
    """Generate overlapping SAHI slices covering the full frame.

    Always includes edge-aligned slices so border vials are not clipped.
    If the frame is smaller than the slice, returns a single full-frame slice.
    """
    if width <= 0 or height <= 0:
        return []
    sw = min(int(slice_w), width)
    sh = min(int(slice_h), height)
    if sw >= width and sh >= height:
        return [Slice(0, 0, width, height)]

    overlap = float(np.clip(overlap_ratio, 0.0, 0.9))
    step_x = max(1, int(sw * (1.0 - overlap)))
    step_y = max(1, int(sh * (1.0 - overlap)))

    xs: List[int] = []
    x = 0
    while True:
        xs.append(x)
        if x + sw >= width:
            break
        x += step_x
        if x + sw > width:
            xs.append(max(0, width - sw))
            break
    ys: List[int] = []
    y = 0
    while True:
        ys.append(y)
        if y + sh >= height:
            break
        y += step_y
        if y + sh > height:
            ys.append(max(0, height - sh))
            break

    # Deduplicate while preserving order.
    seen = set()
    slices: List[Slice] = []
    for yy in ys:
        for xx in xs:
            key = (xx, yy, sw, sh)
            if key in seen:
                continue
            seen.add(key)
            slices.append(Slice(xx, yy, sw, sh))
    return slices


def crop_slice(bgr: np.ndarray, sl: Slice) -> np.ndarray:
    return bgr[sl.y : sl.y + sl.h, sl.x : sl.x + sl.w].copy()


def remap_dets_from_slice(dets: Sequence[Det], sl: Slice) -> List[Det]:
    out: List[Det] = []
    for d in dets:
        out.append(
            Det(
                x1=d.x1 + sl.x,
                y1=d.y1 + sl.y,
                x2=d.x2 + sl.x,
                y2=d.y2 + sl.y,
                conf=d.conf,
                cls_id=d.cls_id,
                label=d.label,
                source=d.source,
            )
        )
    return out


def rotate_image(bgr: np.ndarray, k90: int) -> np.ndarray:
    """Rotate by k*90° CCW (k in {0,1,2,3})."""
    k = int(k90) % 4
    if k == 0:
        return bgr
    if k == 1:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if k == 2:
        return cv2.rotate(bgr, cv2.ROTATE_180)
    return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)


def unrotate_det(d: Det, k90: int, src_w: int, src_h: int) -> Det:
    """Map a box from rotated-image coords back to the original frame.

    ``k90`` matches ``rotate_image``: 0 identity, 1=CCW90, 2=180, 3=CW90.
    ``src_w`` / ``src_h`` are the *original* (unrotated) width / height.
    """
    k = int(k90) % 4
    corners = [
        (d.x1, d.y1),
        (d.x2, d.y1),
        (d.x2, d.y2),
        (d.x1, d.y2),
    ]
    mapped = []
    for x, y in corners:
        if k == 0:
            mapped.append((x, y))
        elif k == 1:  # inverse of CCW90: (x,y)=(W-1-y_r, x_r)
            mapped.append((src_w - 1 - y, x))
        elif k == 2:
            mapped.append((src_w - 1 - x, src_h - 1 - y))
        else:  # inverse of CW90: (x,y)=(y_r, H-1-x_r)
            mapped.append((y, src_h - 1 - x))
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    return Det(
        x1=float(min(xs)),
        y1=float(min(ys)),
        x2=float(max(xs)),
        y2=float(max(ys)),
        conf=d.conf,
        cls_id=d.cls_id,
        label=d.label,
        source=d.source,
    )


def scale_image(bgr: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return bgr
    h, w = bgr.shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    interp = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(bgr, (nw, nh), interpolation=interp)


def unscale_det(d: Det, scale: float) -> Det:
    if abs(scale - 1.0) < 1e-6:
        return d
    inv = 1.0 / scale
    return Det(
        x1=d.x1 * inv,
        y1=d.y1 * inv,
        x2=d.x2 * inv,
        y2=d.y2 * inv,
        conf=d.conf,
        cls_id=d.cls_id,
        label=d.label,
        source=d.source,
    )
