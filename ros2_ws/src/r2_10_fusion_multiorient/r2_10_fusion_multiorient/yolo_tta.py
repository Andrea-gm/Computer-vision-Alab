"""YOLO inference with multi-orientation test-time augmentation (rotations)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np

from .types import Det, classify_orientation


def default_camera_root() -> Path:
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / "vial_dataset").is_dir() and (
            (parent / "runs").is_dir() or (parent / "yolov8s.pt").is_file()
        ):
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
    root = Path(camera_root) if camera_root else default_camera_root()
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    for path in (
        root / "runs" / "vial-live" / "weights" / "best.pt",
        root / "runs" / "train-race" / "r1-winner" / "weights" / "best.pt",
        root / "runs" / "train-race" / "r3-winner" / "weights" / "best.pt",
        root / "runs" / "bon-winner" / "weights" / "best.pt",
        root / "yolov8s.pt",
        root / "yolov8n.pt",
    ):
        if path.exists():
            return path
    return None


def _unrot90_box_once(
    x1: float, y1: float, x2: float, y2: float, rot_h: int, rot_w: int
) -> tuple[float, float, float, float]:
    """Map xyxy from image after one np.rot90 CCW back one step.

    After rot90: shape is (W_old, H_old) = (rot_h, rot_w)? 
    np.rot90(a) with a.shape=(H,W) → out.shape=(W,H).
    out[i,j] = a[j, W-1-i]  ⇒ old_row=j=x_new, old_col=W-1-i=W-1-y_new
    with W = width before rotation = rot_h (out rows).
    """
    w_old = rot_h
    corners = ((x1, y1), (x2, y1), (x1, y2), (x2, y2))
    xs: List[float] = []
    ys: List[float] = []
    for x, y in corners:
        ox = float(w_old - 1) - y
        oy = x
        xs.append(ox)
        ys.append(oy)
    return min(xs), min(ys), max(xs), max(ys)


def map_box_from_rotated(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    k: int,
    orig_h: int,
    orig_w: int,
) -> tuple[float, float, float, float]:
    """Map detection from np.rot90(img, k) coordinates back to original."""
    k = int(k) % 4
    if k == 0:
        return x1, y1, x2, y2
    # Current image size after k rotations from (orig_h, orig_w)
    h, w = orig_h, orig_w
    for _ in range(k):
        h, w = w, h  # shape after each rot90
    cx1, cy1, cx2, cy2 = x1, y1, x2, y2
    for _ in range(k):
        cx1, cy1, cx2, cy2 = _unrot90_box_once(cx1, cy1, cx2, cy2, h, w)
        # step back one rotation: previous shape was (w, h)
        h, w = w, h
    return cx1, cy1, cx2, cy2


class YoloTTADetector:
    """Ultralytics YOLO with optional 0/90/180/270° TTA."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        *,
        camera_root: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.15,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: Optional[str] = None,
        tta_angles: Sequence[int] = (0, 90, 180, 270),
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.device = device
        self.tta_k = sorted({(int(a) // 90) % 4 for a in tta_angles})
        self.camera_root = Path(camera_root) if camera_root else default_camera_root()
        self.model_path = resolve_model_path(model_path, self.camera_root)
        self.model = None
        self.backend = "none"
        self._load()

    def _load(self) -> None:
        if self.model_path is None:
            return
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
            self.backend = "yolo"
        except Exception as exc:  # noqa: BLE001
            self.model = None
            self.backend = "none"
            self._load_error = str(exc)

    @property
    def available(self) -> bool:
        return self.backend == "yolo" and self.model is not None

    def predict(self, bgr: np.ndarray) -> List[Det]:
        if not self.available or bgr is None or bgr.size == 0:
            return []
        h0, w0 = bgr.shape[:2]
        all_dets: List[Det] = []

        def _run_k(k: int) -> List[Det]:
            img = np.ascontiguousarray(np.rot90(bgr, k)) if k else bgr
            dets = self._predict_once(img)
            out: List[Det] = []
            for d in dets:
                if k:
                    x1, y1, x2, y2 = map_box_from_rotated(
                        d.x1, d.y1, d.x2, d.y2, k, h0, w0
                    )
                    d.x1, d.y1, d.x2, d.y2 = x1, y1, x2, y2
                    d.conf = min(0.99, d.conf * 1.05)
                    d.source = f"yolo_tta{k * 90}"
                else:
                    d.source = "yolo"
                d.clamp(w0, h0)
                d.orientation = classify_orientation(d)
                out.append(d)
            return out

        # Always run native orientation first (upright / trained pose).
        native = _run_k(0) if 0 in self.tta_k else []
        all_dets.extend(native)
        strong_upright = any(
            d.conf >= max(self.confidence_threshold, 0.25) and d.aspect_hw >= 0.95
            for d in native
        )
        # Rotation TTA only when native YOLO is weak — cuts empty FP + latency.
        if not strong_upright:
            for k in self.tta_k:
                if k == 0:
                    continue
                all_dets.extend(_run_k(k))
        return all_dets

    def _predict_once(self, bgr: np.ndarray) -> List[Det]:
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
            conf = float(confs[i])
            cls_id = int(clss[i]) if len(clss) else 0
            label = str(names.get(cls_id, "vial") if isinstance(names, dict) else "vial")
            dets.append(
                Det(x1, y1, x2, y2, conf, cls_id=cls_id, label=label, source="yolo")
            )
        return dets


def annotate(bgr: np.ndarray, dets: Sequence[Det]) -> np.ndarray:
    from .viz import annotate_detections

    return annotate_detections(bgr, dets)
