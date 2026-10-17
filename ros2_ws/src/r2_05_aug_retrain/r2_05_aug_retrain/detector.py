"""Ultralytics YOLO detector for r2_05 — prefers package-retrained weights."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PKG_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Det:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int = 0


@dataclass
class InferStats:
    backend: str
    model_path: str
    inference_ms: float
    num_dets: int


def default_camera_root() -> Path:
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / "vial_dataset").is_dir() and (parent / "runs").is_dir():
            return parent
        if (parent / "vial_dataset").is_dir() and (parent / "yolov8s.pt").is_file():
            return parent
    for parent in here.parents:
        if parent.name == "ros2_ws":
            sibling = parent.parent / "robotics"
            if sibling.is_dir():
                return sibling
            break
    return here.parents[4]


def package_weights() -> Path:
    return PKG_ROOT / "models" / "weights" / "best.pt"


def resolve_model_path(
    explicit: Optional[Union[str, Path]],
    camera_root: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Prefer r2_05 retrained weights; never silently fall back to vial-live."""
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    pkg = package_weights()
    if pkg.exists():
        return pkg
    root = Path(camera_root) if camera_root else default_camera_root()
    for cand in (
        PKG_ROOT / "runs" / "r2_05" / "weights" / "best.pt",
        root / "yolov8s.pt",
        root / "yolov8n.pt",
    ):
        if cand.exists():
            return cand
    return None


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_yolo_label(label_path: Path, width: int, height: int) -> List[Det]:
    if not label_path.exists():
        return []
    dets: List[Det] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = parts
        cx_f, cy_f, bw_f, bh_f = float(cx), float(cy), float(bw), float(bh)
        x1 = (cx_f - bw_f / 2.0) * width
        y1 = (cy_f - bh_f / 2.0) * height
        x2 = (cx_f + bw_f / 2.0) * width
        y2 = (cy_f + bh_f / 2.0) * height
        dets.append(Det(x1, y1, x2, y2, conf=1.0, cls=int(float(cls))))
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
    return {
        "x1": float(d.x1),
        "y1": float(d.y1),
        "x2": float(d.x2),
        "y2": float(d.y2),
        "conf": float(d.conf),
        "cls": int(d.cls),
    }


def annotate(bgr: np.ndarray, dets: Sequence[Det]) -> np.ndarray:
    out = bgr.copy()
    for d in dets:
        p1 = (int(d.x1), int(d.y1))
        p2 = (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, (0, 220, 80), 2)
        cv2.putText(
            out,
            f"vial {d.conf:.2f}",
            (p1[0], max(16, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 80),
            1,
            cv2.LINE_AA,
        )
    return out


class VialYoloDetector:
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        *,
        camera_root: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: Optional[str] = None,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.camera_root = Path(camera_root) if camera_root else default_camera_root()
        self.device = device
        self.model = None
        self.backend = "none"
        self.model_path: Optional[Path] = resolve_model_path(model_path, self.camera_root)
        self._load()

    def _load(self) -> None:
        if self.model_path is None:
            raise FileNotFoundError(
                "No r2_05 weights found. Train with scripts/train.py "
                f"(expected {package_weights()})."
            )
        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))
        self.backend = "yolo"

    def predict(self, bgr: np.ndarray) -> Tuple[List[Det], InferStats]:
        t0 = time.perf_counter()
        kwargs = dict(
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(bgr, **kwargs)
        ms = (time.perf_counter() - t0) * 1000.0
        dets: List[Det] = []
        if results:
            r0 = results[0]
            if r0.boxes is not None and len(r0.boxes):
                xyxy = r0.boxes.xyxy.cpu().numpy()
                confs = r0.boxes.conf.cpu().numpy()
                clss = r0.boxes.cls.cpu().numpy()
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = map(float, xyxy[i])
                    dets.append(
                        Det(x1, y1, x2, y2, conf=float(confs[i]), cls=int(clss[i]))
                    )
        stats = InferStats(
            backend=self.backend,
            model_path=str(self.model_path),
            inference_ms=ms,
            num_dets=len(dets),
        )
        return dets, stats
