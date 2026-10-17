"""Ultralytics YOLO vial detector + weight resolution + classical fallback."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Union
import time

import cv2
import numpy as np

from .classical_fallback import Det, annotate, detect_vial_classical

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class InferStats:
    backend: str
    model_path: str
    inference_ms: float
    num_dets: int


def default_camera_root() -> Path:
    """Locate Camera/ (contains vial_dataset / runs) above this package.

    Package lives at Camera/ros2_ws/src/agent_01_yolo/ — walk up until markers hit.
    """
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / "vial_dataset").is_dir() and (parent / "runs").is_dir():
            return parent
        if (parent / "vial_dataset").is_dir() and (parent / "yolov8s.pt").is_file():
            return parent
    # Fallback: .../Camera/ros2_ws/src/agent_01_yolo/agent_01_yolo → parents[4]=Camera
    for parent in here.parents:
        if parent.name == "ros2_ws":
            sibling = parent.parent / "robotics"
            if sibling.is_dir():
                return sibling
            break
    return here.parents[4]


def resolve_model_path(
    explicit: Optional[Union[str, Path]],
    camera_root: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Match Camera/depthcamera.find_weights preference order.

    vial-live → r1-winner → bon-winner → yolov8s.pt → yolov8n.pt
    """
    root = Path(camera_root) if camera_root else default_camera_root()
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    preferred = [
        root / "runs" / "vial-live" / "weights" / "best.pt",
        root / "runs" / "train-race" / "r1-winner" / "weights" / "best.pt",
        root / "runs" / "train-race" / "r3-winner" / "weights" / "best.pt",
        root / "runs" / "bon-winner" / "weights" / "best.pt",
        root / "yolov8s.pt",
        root / "yolov8n.pt",
    ]
    for path in preferred:
        if path.exists():
            return path
    return None


class VialYoloDetector:
    """One-stage detector. Prefers Ultralytics YOLO; can fall back to classical."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        *,
        camera_root: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        allow_classical_fallback: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.allow_classical_fallback = bool(allow_classical_fallback)
        self.camera_root = Path(camera_root) if camera_root else default_camera_root()
        self.device = device
        self.model = None
        self.backend = "none"
        self.model_path: Optional[Path] = resolve_model_path(model_path, self.camera_root)
        self._load()

    def _load(self) -> None:
        if self.model_path is None:
            if self.allow_classical_fallback:
                self.backend = "classical"
                return
            raise FileNotFoundError(
                "No YOLO weights found under Camera/runs or Camera/*.pt. "
                "Train with train_vial_model.py or set model_path."
            )
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
            self.backend = "yolo"
        except Exception as exc:  # noqa: BLE001 — surface load failure, keep demo path
            if not self.allow_classical_fallback:
                raise RuntimeError(f"Failed to load YOLO from {self.model_path}: {exc}") from exc
            self.model = None
            self.backend = "classical"
            self._load_error = str(exc)

    def predict(self, bgr: np.ndarray) -> tuple[List[Det], InferStats]:
        t0 = time.perf_counter()
        if self.backend == "yolo" and self.model is not None:
            dets = self._predict_yolo(bgr)
        else:
            dets = detect_vial_classical(bgr)
            dets = [d for d in dets if d.conf >= self.confidence_threshold]
        ms = (time.perf_counter() - t0) * 1000.0
        stats = InferStats(
            backend=self.backend,
            model_path=str(self.model_path or ""),
            inference_ms=ms,
            num_dets=len(dets),
        )
        return dets, stats

    def _predict_yolo(self, bgr: np.ndarray) -> List[Det]:
        kwargs = dict(
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(bgr, **kwargs)
        dets: List[Det] = []
        if not results:
            return dets
        r0 = results[0]
        names = getattr(r0, "names", None) or getattr(self.model, "names", {}) or {0: "vial"}
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return dets
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            cls_id = int(clss[i]) if len(clss) else 0
            label = str(names.get(cls_id, "vial") if isinstance(names, dict) else "vial")
            dets.append(
                Det(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    conf=float(confs[i]),
                    cls_id=cls_id,
                    label=label,
                    source="yolo",
                )
            )
        return dets

    def annotate(self, bgr: np.ndarray, dets: Sequence[Det]) -> np.ndarray:
        color = (0, 220, 80) if self.backend == "yolo" else (40, 180, 255)
        return annotate(bgr, list(dets), color=color)


def list_images(folder: Union[str, Path]) -> List[Path]:
    root = Path(folder)
    if not root.exists():
        return []
    files = [
        p
        for p in sorted(root.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return files


def load_yolo_label(label_path: Path, width: int, height: int) -> List[Det]:
    """Parse YOLO txt (class cx cy w h normalized) → pixel Det list."""
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    dets: List[Det] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, bw, bh = [float(v) for v in parts[1:]]
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        dets.append(
            Det(x1, y1, x2, y2, conf=1.0, cls_id=cls_id, label="vial", source="gt")
        )
    return dets


def box_iou(a: Det, b: Det) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def det_to_dict(d: Det) -> dict:
    return asdict(d)
