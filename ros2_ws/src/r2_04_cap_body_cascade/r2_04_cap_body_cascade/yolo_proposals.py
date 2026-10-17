"""Optional YOLO proposal stage for RGB-only upright stills.

Used when classical cap→body ranking fails on real benches (glare / no depth).
Keeps orientation-agnostic cascade as the sideways/depth path.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .types import Det

_CAMERA = Path(r"E:\Desktop\Jessica\robotics")


def resolve_weights(explicit: Optional[Union[str, Path]] = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
    candidates = [
        _CAMERA / "runs" / "train-race" / "r2-winner" / "weights" / "best.pt",
        _CAMERA / "runs" / "train-race" / "r3-winner" / "weights" / "best.pt",
        _CAMERA / "runs" / "train-race" / "r1-winner" / "weights" / "best.pt",
        _CAMERA / "runs" / "vial-live" / "weights" / "best.pt",
        _CAMERA / "yolov8s.pt",
        _CAMERA / "yolov8n.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


class YoloProposer:
    """Lazy Ultralytics YOLO → Det list."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ) -> None:
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.model_path = resolve_weights(model_path)
        self._model = None
        self.available = False
        self._error = ""
        self._try_load()

    def _try_load(self) -> None:
        if self.model_path is None:
            self._error = "no weights"
            return
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
            self.available = True
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            self._model = None
            self.available = False

    def propose(self, bgr: np.ndarray) -> List[Det]:
        if not self.available or self._model is None or bgr is None:
            return []
        results = self._model.predict(
            bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return []
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []
        out: List[Det] = []
        h, w = bgr.shape[:2]
        for b in r0.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            x1 = float(np.clip(x1, 0, w - 1))
            y1 = float(np.clip(y1, 0, h - 1))
            x2 = float(np.clip(x2, 0, w - 1))
            y2 = float(np.clip(y2, 0, h - 1))
            score = float(b.conf[0]) if b.conf is not None else 0.5
            bw, bh = x2 - x1, y2 - y1
            # Axis from box elongation (orientation cue for cascade fusion).
            if bw >= bh:
                ang = 0.0
            else:
                ang = 90.0
            out.append(
                Det(
                    x1,
                    y1,
                    x2,
                    y2,
                    conf=min(0.99, max(0.0, score)),
                    source="yolo",
                    axis_angle_deg=ang,
                    axis_method="yolo_proposal",
                )
            )
        return out
