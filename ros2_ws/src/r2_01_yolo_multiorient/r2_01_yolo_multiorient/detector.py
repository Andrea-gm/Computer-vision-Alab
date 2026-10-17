"""Multi-orientation Ultralytics YOLO vial detector + weight resolve."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from .types import Det, annotate

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class InferStats:
    backend: str
    model_path: str
    inference_ms: float
    num_dets: int


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def resolve_model_path(
    explicit: Optional[Union[str, Path]],
    camera_root: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Prefer package multi-orient weights, then Camera runs."""
    root = Path(camera_root) if camera_root else default_camera_root()
    pkg = package_root()
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    preferred = [
        pkg / "models" / "best.pt",
        pkg / "models" / "yolov8_multiorient_best.pt",
        root / "runs" / "r2-01-multiorient" / "weights" / "best.pt",
        root / "runs" / "vial-live" / "weights" / "best.pt",
        root / "runs" / "train-race" / "r1-winner" / "weights" / "best.pt",
        root / "runs" / "bon-winner" / "weights" / "best.pt",
        root / "yolov8s.pt",
        root / "yolov8n.pt",
    ]
    for path in preferred:
        if path.exists():
            return path
    return None


class VialYoloDetector:
    """YOLO-only detector — no upright-biased classical fallback."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        *,
        camera_root: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: Optional[str] = None,
        tta_rotate: bool = True,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.camera_root = Path(camera_root) if camera_root else default_camera_root()
        self.device = device
        # TTA: also run 90° CW / CCW and map boxes back (covers upright-biased weights).
        self.tta_rotate = bool(tta_rotate)
        self.model = None
        self.backend = "none"
        self.model_path: Optional[Path] = resolve_model_path(model_path, self.camera_root)
        self._load()

    def _load(self) -> None:
        if self.model_path is None:
            raise FileNotFoundError(
                "No YOLO weights found. Train with scripts/train_multiorient.py "
                "or set model_path to models/best.pt."
            )
        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))
        self.backend = "yolo"

    def predict(self, bgr: np.ndarray) -> tuple[List[Det], InferStats]:
        t0 = time.perf_counter()
        dets = self._predict_yolo(bgr)
        if self.tta_rotate:
            dets = self._nms(dets + self._predict_rotated_tta(bgr), self.iou_threshold)
        ms = (time.perf_counter() - t0) * 1000.0
        stats = InferStats(
            backend=self.backend + ("+rotTTA" if self.tta_rotate else ""),
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
        return self._boxes_to_dets(results)

    def _predict_rotated_tta(self, bgr: np.ndarray) -> List[Det]:
        """Infer on 90° and 270° rotations; remap boxes to original coords."""
        h, w = bgr.shape[:2]
        out: List[Det] = []
        # 90° CW: (x,y) -> (h-1-y, x) in new image of size (w,h) wait:
        # cv2.ROTATE_90_CLOCKWISE: new shape (W,H) from (H,W); (x,y)->(H-1-y, x)
        for code, remap in (
            (cv2_rotate_90_cw, _remap_from_90_cw),
            (cv2_rotate_90_ccw, _remap_from_90_ccw),
        ):
            rot = code(bgr)
            dets = self._predict_yolo(rot)
            for d in dets:
                x1, y1, x2, y2 = remap(d.x1, d.y1, d.x2, d.y2, h, w)
                out.append(
                    Det(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        conf=d.conf * 0.98,
                        cls_id=d.cls_id,
                        label=d.label,
                        source="yolo_rotTTA",
                    )
                )
        return out

    def _boxes_to_dets(self, results) -> List[Det]:
        dets: List[Det] = []
        if not results:
            return dets
        r0 = results[0]
        names = getattr(r0, "names", None) or getattr(self.model, "names", {}) or {0: "vial"}
        boxes = getattr(r0, "boxes", None)
        if boxes is None or len(boxes) == 0:
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

    @staticmethod
    def _nms(dets: List[Det], iou_thr: float) -> List[Det]:
        if not dets:
            return []
        ordered = sorted(dets, key=lambda d: d.conf, reverse=True)
        keep: List[Det] = []
        from .types import box_iou

        for d in ordered:
            if all(box_iou(d, k) < iou_thr for k in keep):
                keep.append(d)
        return keep

    def annotate(self, bgr: np.ndarray, dets: Sequence[Det]) -> np.ndarray:
        return annotate(bgr, list(dets), color=(0, 220, 80))


def cv2_rotate_90_cw(bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)


def cv2_rotate_90_ccw(bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _remap_from_90_cw(x1, y1, x2, y2, h_orig, w_orig):
    # Rotated image size: (w_orig, h_orig). Inverse of (x,y)->(h-1-y, x):
    # (xr, yr) in rot → (yr, h_orig-1-xr) in original.
    xs = [y1, y2]
    ys = [h_orig - 1 - x1, h_orig - 1 - x2]
    return min(xs), min(ys), max(xs), max(ys)


def _remap_from_90_ccw(x1, y1, x2, y2, h_orig, w_orig):
    # 90 CCW: (x,y)->(y, w-1-x). Inverse: (xr,yr)->(w-1-yr, xr)
    xs = [w_orig - 1 - y1, w_orig - 1 - y2]
    ys = [x1, x2]
    return min(xs), min(ys), max(xs), max(ys)


def list_images(folder: Union[str, Path]) -> List[Path]:
    root = Path(folder)
    if not root.exists():
        return []
    return [
        p
        for p in sorted(root.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def load_yolo_label(label_path: Path, width: int, height: int) -> List[Det]:
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
