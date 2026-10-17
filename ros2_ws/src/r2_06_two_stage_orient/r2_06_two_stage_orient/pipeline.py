"""Two-stage vial detector: Stage A proposals → Stage B orientation refine."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np

from .stage_a import CAMERA_ROOT_DEFAULT, StageAProposer
from .stage_b import StageBOrient
from .types import OrientedDet


@dataclass
class InferStats:
    backend: str
    model_path: str
    inference_ms: float
    n_proposals: int
    n_dets: int


class TwoStageOrientDetector:
    """Public API used by the ROS node and evaluate.py."""

    def __init__(
        self,
        *,
        camera_root: Union[str, Path] = CAMERA_ROOT_DEFAULT,
        model_path: Optional[Union[str, Path]] = None,
        conf: float = 0.12,
        iou: float = 0.45,
        imgsz: int = 640,
        use_yolo: bool = True,
        use_rotation_tta: bool = True,
        use_cap_proposals: bool = True,
        use_sideways_method: bool = True,
        min_conf: float = 0.22,
        max_dets: int = 3,
        drop_other: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.stage_a = StageAProposer(
            camera_root=camera_root,
            model_path=model_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            use_yolo=use_yolo,
            use_rotation_tta=use_rotation_tta,
            use_cap_proposals=use_cap_proposals,
            use_sideways_method=use_sideways_method,
            device=device,
        )
        self.stage_b = StageBOrient(
            min_conf=min_conf,
            max_dets=max_dets,
            drop_other=drop_other,
        )
        self.camera_root = Path(camera_root)

    def load(self) -> None:
        self.stage_a.load()

    @property
    def backend(self) -> str:
        return f"two_stage[{self.stage_a.backend}]"

    @property
    def model_path(self) -> str:
        return str(self.stage_a.weights or "")

    def detect(
        self,
        bgr: np.ndarray,
        depth_mm: Optional[np.ndarray] = None,
        intrinsics: Optional[dict] = None,
    ) -> tuple[List[OrientedDet], InferStats]:
        t0 = time.perf_counter()
        if self.stage_a.use_yolo and self.stage_a._model is None and self.stage_a._yolo_error is None:
            self.load()
        props = self.stage_a.propose(bgr, depth_mm=depth_mm, intrinsics=intrinsics)
        dets = self.stage_b.refine(bgr, props, depth_mm=depth_mm)
        ms = (time.perf_counter() - t0) * 1000.0
        stats = InferStats(
            backend=self.backend,
            model_path=self.model_path,
            inference_ms=ms,
            n_proposals=len(props),
            n_dets=len(dets),
        )
        return dets, stats

    def annotate(self, bgr: np.ndarray, dets: Sequence[OrientedDet]) -> np.ndarray:
        out = bgr.copy()
        colors = {
            "upright": (40, 200, 80),
            "sideways": (40, 160, 255),
            "other": (180, 180, 60),
        }
        for d in dets:
            color = colors.get(d.orientation, (200, 200, 200))
            p1 = (d.x, d.y)
            p2 = (d.x + d.w, d.y + d.h)
            cv2.rectangle(out, p1, p2, color, 2)
            tag = f"{d.orientation} {d.conf:.2f}"
            cv2.putText(
                out,
                tag,
                (p1[0], max(16, p1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return out
