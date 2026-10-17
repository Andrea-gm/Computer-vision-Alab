"""Oriented-box geometry: xywh↔OBB conversion, polygon IoU, orientation helpers.

Ultralytics YOLO-OBB labels use four normalized corners:
    class x1 y1 x2 y2 x3 y3 x4 y4
OpenCV rotated rects use (center, (w, h), angle_deg).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class OBBDet:
    """Oriented detection in pixel coordinates."""

    cx: float
    cy: float
    width: float  # along the long axis of the vial when possible
    height: float  # short axis (diameter-like)
    angle_deg: float  # OpenCV convention: degrees, long-side orientation
    conf: float = 1.0
    cls_id: int = 0
    label: str = "vial"
    source: str = "obb"
    orientation: str = "unknown"  # upright | sideways | unknown

    def corners(self) -> np.ndarray:
        """Return 4x2 float32 corner points (OpenCV order)."""
        return cv2.boxPoints(
            ((float(self.cx), float(self.cy)), (float(self.width), float(self.height)), float(self.angle_deg))
        ).astype(np.float32)

    def aabb(self) -> Tuple[float, float, float, float]:
        """Axis-aligned xyxy envelope."""
        pts = self.corners()
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        return x1, y1, x2, y2

    def to_dict(self) -> dict:
        d = asdict(self)
        d["corners"] = self.corners().tolist()
        d["aabb_xyxy"] = list(self.aabb())
        return d


def normalize_angle_deg(angle: float) -> float:
    """Map angle into (-90, 90]."""
    a = float(angle) % 180.0
    if a > 90.0:
        a -= 180.0
    if a <= -90.0:
        a += 180.0
    return a


def classify_orientation(width: float, height: float, angle_deg: float) -> str:
    """Classify upright vs sideways from OBB geometry.

    Sideways ≈ long axis near horizontal (|angle| close to 0 with w>h, or
    |angle| near 90 with h>w depending on size convention). We canonicalize
    so width is the longer side.
    """
    w, h, a = float(width), float(height), normalize_angle_deg(angle_deg)
    if w < h:
        w, h = h, w
        a = normalize_angle_deg(a + 90.0)
    # long-axis angle a: near 0 → horizontal (sideways), near ±90 → vertical
    long_axis_from_horiz = min(abs(a), 180.0 - abs(a))
    if long_axis_from_horiz <= 35.0:
        return "sideways"
    if long_axis_from_horiz >= 55.0:
        return "upright"
    return "unknown"


def canonical_obb(
    cx: float,
    cy: float,
    width: float,
    height: float,
    angle_deg: float,
    *,
    conf: float = 1.0,
    source: str = "obb",
) -> OBBDet:
    """Ensure width >= height (long side first) and set orientation tag."""
    w, h, a = float(width), float(height), float(angle_deg)
    if w < h:
        w, h = h, w
        a = a + 90.0
    a = normalize_angle_deg(a)
    ori = classify_orientation(w, h, a)
    return OBBDet(
        cx=float(cx),
        cy=float(cy),
        width=w,
        height=h,
        angle_deg=a,
        conf=float(conf),
        source=source,
        orientation=ori,
    )


def aabb_xywh_to_obb(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    angle_deg: Optional[float] = None,
    conf: float = 1.0,
    source: str = "aabb_convert",
) -> OBBDet:
    """Convert axis-aligned xywh (pixels) to an oriented box.

    If ``angle_deg`` is None, infer from aspect:
      - tall (h > w) → upright, angle ≈ -90° (long axis vertical in OpenCV)
      - wide (w > h) → sideways, angle ≈ 0°
    When a non-zero angle is provided (angled hard samples), recover approximate
    true length/diameter from the AABB envelope + known rotation.
    """
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    if angle_deg is None:
        if h >= w * 1.05:
            # vertical vial: OpenCV size=(w,h) angle=0 is upright AABB
            return canonical_obb(cx, cy, max(w, h), min(w, h), -90.0 if h >= w else 0.0, conf=conf, source=source)
        if w >= h * 1.05:
            return canonical_obb(cx, cy, w, h, 0.0, conf=conf, source=source)
        return canonical_obb(cx, cy, max(w, h), min(w, h), -90.0, conf=conf, source=source)

    a = float(angle_deg)
    length, diameter = _recover_size_from_aabb(w, h, a)
    return canonical_obb(cx, cy, length, diameter, a, conf=conf, source=source)


def _recover_size_from_aabb(bw: float, bh: float, angle_deg: float) -> Tuple[float, float]:
    """Recover (length, diameter) of a rotated rectangle from its AABB."""
    c = abs(math.cos(math.radians(angle_deg)))
    s = abs(math.sin(math.radians(angle_deg)))
    det = c * c - s * s
    if abs(det) < 1e-3:
        return max(bw, bh), min(bw, bh)
    # [c s; s c] [L; D] = [bw; bh]
    length = (bw * c - bh * s) / det
    diameter = (bh * c - bw * s) / det
    if length <= 1.0 or diameter <= 1.0 or length < diameter:
        return max(bw, bh), min(bw, bh)
    return float(length), float(diameter)


def yolo_xywhn_to_obb(
    cx_n: float,
    cy_n: float,
    w_n: float,
    h_n: float,
    img_w: int,
    img_h: int,
    *,
    angle_deg: Optional[float] = None,
    conf: float = 1.0,
) -> OBBDet:
    """YOLO normalized cx,cy,w,h → OBBDet pixels."""
    bw = w_n * img_w
    bh = h_n * img_h
    x = cx_n * img_w - 0.5 * bw
    y = cy_n * img_h - 0.5 * bh
    return aabb_xywh_to_obb(x, y, bw, bh, angle_deg=angle_deg, conf=conf)


def obb_to_yolo_corners_line(det: OBBDet, img_w: int, img_h: int) -> str:
    """Serialize OBBDet as Ultralytics OBB label line (normalized corners)."""
    pts = det.corners()
    coords: List[str] = []
    for x, y in pts:
        coords.append(f"{max(0.0, min(1.0, x / img_w)):.6f}")
        coords.append(f"{max(0.0, min(1.0, y / img_h)):.6f}")
    return f"{det.cls_id} " + " ".join(coords)


def corners_to_obb(corners: np.ndarray, *, conf: float = 1.0, source: str = "corners") -> OBBDet:
    rect = cv2.minAreaRect(corners.astype(np.float32))
    (cx, cy), (w, h), angle = rect
    return canonical_obb(cx, cy, w, h, angle, conf=conf, source=source)


def polygon_iou(a: OBBDet, b: OBBDet) -> float:
    """IoU of two oriented boxes via contour intersection (pixel grid)."""
    pts_a = a.corners().astype(np.float32)
    pts_b = b.corners().astype(np.float32)
    # Bound region
    all_pts = np.vstack([pts_a, pts_b])
    x0 = int(math.floor(all_pts[:, 0].min())) - 1
    y0 = int(math.floor(all_pts[:, 1].min())) - 1
    x1 = int(math.ceil(all_pts[:, 0].max())) + 1
    y1 = int(math.ceil(all_pts[:, 1].max())) + 1
    if x1 <= x0 or y1 <= y0:
        return 0.0
    w, h = x1 - x0, y1 - y0
    if w * h > 4_000_000:
        # Fallback: shapely-free approx via AABB IoU of envelopes
        return aabb_iou(a.aabb(), b.aabb())
    ma = np.zeros((h, w), np.uint8)
    mb = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(ma, np.rint(pts_a - np.array([x0, y0])).astype(np.int32), 1)
    cv2.fillConvexPoly(mb, np.rint(pts_b - np.array([x0, y0])).astype(np.int32), 1)
    inter = float(np.logical_and(ma, mb).sum())
    union = float(np.logical_or(ma, mb).sum())
    return 0.0 if union <= 0 else inter / union


def aabb_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def parse_angle_from_note(note: str) -> Optional[float]:
    if not note:
        return None
    m = re.search(r"angle\s*=\s*(-?\d+(?:\.\d+)?)", note, flags=re.I)
    if not m:
        return None
    return float(m.group(1))


def load_yolo_aabb_label(path, img_w: int, img_h: int) -> List[OBBDet]:
    """Load classic YOLO xywhn labels and convert each to OBB."""
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    out: List[OBBDet] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        # OBB corner format already?
        if len(parts) >= 9:
            coords = np.array([float(v) for v in parts[1:9]], dtype=np.float32).reshape(4, 2)
            coords[:, 0] *= img_w
            coords[:, 1] *= img_h
            out.append(corners_to_obb(coords, conf=1.0, source="gt_obb"))
            continue
        cls_id = int(float(parts[0]))
        cx_n, cy_n, w_n, h_n = map(float, parts[1:5])
        det = yolo_xywhn_to_obb(cx_n, cy_n, w_n, h_n, img_w, img_h)
        det.cls_id = cls_id
        det.source = "gt_aabb"
        out.append(det)
    return out


def annotate_obb(bgr: np.ndarray, dets: Iterable[OBBDet]) -> np.ndarray:
    vis = bgr.copy()
    for d in dets:
        pts = np.rint(d.corners()).astype(np.int32)
        color = (40, 200, 40) if d.orientation == "upright" else (40, 180, 220)
        if d.orientation == "unknown":
            color = (200, 180, 40)
        cv2.polylines(vis, [pts], True, color, 2, cv2.LINE_AA)
        x1, y1, _, _ = d.aabb()
        label = f"{d.label} {d.conf:.2f} {d.orientation} {d.angle_deg:.0f}deg"
        cv2.putText(
            vis,
            label,
            (int(x1), max(12, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def list_images(folder) -> List:
    from pathlib import Path

    root = Path(folder)
    if not root.is_dir():
        return []
    files = [p for p in sorted(root.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return files
