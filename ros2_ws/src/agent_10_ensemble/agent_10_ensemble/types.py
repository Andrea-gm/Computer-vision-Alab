"""Shared detection types for the ensemble pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Detection:
    """Axis-aligned vial box in pixel coordinates (xyxy) with fused score."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    source: str = "ensemble"
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.as_int()
        return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

    def as_int(self) -> tuple[int, int, int, int]:
        return (
            int(round(self.x1)),
            int(round(self.y1)),
            int(round(self.x2)),
            int(round(self.y2)),
        )

    def clip(self, width: int, height: int) -> "Detection":
        x1 = float(max(0, min(width - 1, self.x1)))
        y1 = float(max(0, min(height - 1, self.y1)))
        x2 = float(max(0, min(width, self.x2)))
        y2 = float(max(0, min(height, self.y2)))
        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)
        return Detection(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            score=self.score,
            source=self.source,
            extras=dict(self.extras),
        )
