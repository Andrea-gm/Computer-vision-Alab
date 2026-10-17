"""Shared detection types."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class Det:
    """Axis-aligned detection in pixel coordinates (xyxy)."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int = 0
    label: str = "vial"
    source: str = "fusion"
    orientation: str = "unknown"  # upright | sideways | topdown | unknown

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.xyxy
        return (
            int(round(x1)),
            int(round(y1)),
            max(1, int(round(x2 - x1))),
            max(1, int(round(y2 - y1))),
        )

    @property
    def aspect_hw(self) -> float:
        w = max(1e-6, self.x2 - self.x1)
        h = max(1e-6, self.y2 - self.y1)
        return float(h / w)

    def clamp(self, width: int, height: int) -> "Det":
        self.x1 = float(max(0, min(width - 1, self.x1)))
        self.y1 = float(max(0, min(height - 1, self.y1)))
        self.x2 = float(max(self.x1 + 1, min(width, self.x2)))
        self.y2 = float(max(self.y1 + 1, min(height, self.y2)))
        return self

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["xywh"] = list(self.xywh)
        d["aspect_hw"] = self.aspect_hw
        return d


def box_iou(a: Det, b: Det) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def xywh_to_det(
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    conf: float = 1.0,
    source: str = "gt",
    orientation: str = "unknown",
) -> Det:
    return Det(
        float(x),
        float(y),
        float(x + w),
        float(y + h),
        float(conf),
        source=source,
        orientation=orientation,
    )


def classify_orientation(det: Det) -> str:
    """Heuristic orientation from AABB aspect (h/w)."""
    hw = det.aspect_hw
    if hw < 0.85:
        return "sideways"
    if 0.85 <= hw <= 1.15:
        return "topdown"
    return "upright"
