"""Shared detection / keypoint types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Keypoint:
    x: float
    y: float
    name: str = ""
    conf: float = 1.0


@dataclass
class Det:
    """Axis-aligned cover of an oriented vial box, plus skeleton keypoints."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int = 0
    label: str = "vial"
    source: str = "keypoint_axis"
    # Oriented box: center, size along axis / diameter, angle degrees (image, CCW from +x)
    cx: float = 0.0
    cy: float = 0.0
    length: float = 0.0
    diameter: float = 0.0
    angle_deg: float = 0.0
    cap: Optional[Keypoint] = None
    base: Optional[Keypoint] = None
    pose: str = ""  # upright | sideways | angled

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_xywh(self) -> Tuple[int, int, int, int]:
        return (
            int(round(self.x1)),
            int(round(self.y1)),
            int(round(max(0.0, self.x2 - self.x1))),
            int(round(max(0.0, self.y2 - self.y1))),
        )


def det_to_dict(d: Det) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "x1": float(d.x1),
        "y1": float(d.y1),
        "x2": float(d.x2),
        "y2": float(d.y2),
        "confidence": float(d.conf),
        "class_id": int(d.cls_id),
        "label": d.label,
        "source": d.source,
        "obb": {
            "cx": float(d.cx),
            "cy": float(d.cy),
            "length": float(d.length),
            "diameter": float(d.diameter),
            "angle_deg": float(d.angle_deg),
        },
        "pose": d.pose,
    }
    if d.cap is not None:
        out["cap"] = asdict(d.cap)
    if d.base is not None:
        out["base"] = asdict(d.base)
    return out


def box_iou(a: Det, b: Det) -> float:
    ax1, ay1, ax2, ay2 = a.as_xyxy()
    bx1, by1, bx2, by2 = b.as_xyxy()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (aa + bb - inter + 1e-9)


def xywh_to_det(x: float, y: float, w: float, h: float, conf: float = 1.0) -> Det:
    return Det(x1=x, y1=y, x2=x + w, y2=y + h, conf=conf)


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between undirected axes (mod 180)."""
    d = abs((a - b + 180.0) % 180.0)
    return float(min(d, 180.0 - d))


@dataclass
class InferStats:
    backend: str = "keypoint_axis"
    inference_ms: float = 0.0
    n_caps: int = 0
    n_kept: int = 0
    extras: Dict[str, Any] = field(default_factory=dict)


def list_images(folder) -> List:
    from pathlib import Path

    folder = Path(folder)
    if not folder.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
