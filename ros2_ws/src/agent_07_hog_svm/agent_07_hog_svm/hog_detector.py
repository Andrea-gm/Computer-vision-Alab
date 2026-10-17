"""HOG + Linear SVM vial detector (OpenCV features, sklearn/OpenCV SVM)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Default HOG window tuned to median vial aspect (~0.84 w/h) on this dataset.
DEFAULT_WIN_SIZE = (64, 80)  # (width, height)
DEFAULT_BLOCK_SIZE = (16, 16)
DEFAULT_BLOCK_STRIDE = (8, 8)
DEFAULT_CELL_SIZE = (8, 8)
DEFAULT_NBINS = 9


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    score: float

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    def as_dict(self) -> dict:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "w": int(self.w),
            "h": int(self.h),
            "score": float(self.score),
            "label": "vial",
        }


def make_hog(
    win_size: Tuple[int, int] = DEFAULT_WIN_SIZE,
    block_size: Tuple[int, int] = DEFAULT_BLOCK_SIZE,
    block_stride: Tuple[int, int] = DEFAULT_BLOCK_STRIDE,
    cell_size: Tuple[int, int] = DEFAULT_CELL_SIZE,
    nbins: int = DEFAULT_NBINS,
) -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        _winSize=win_size,
        _blockSize=block_size,
        _blockStride=block_stride,
        _cellSize=cell_size,
        _nbins=nbins,
    )


def yolo_to_xyxy(
    xc: float, yc: float, bw: float, bh: float, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    w = bw * img_w
    h = bh * img_h
    x1 = int(round(xc * img_w - w / 2.0))
    y1 = int(round(yc * img_h - h / 2.0))
    x2 = int(round(xc * img_w + w / 2.0))
    y2 = int(round(yc * img_h + h / 2.0))
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w, x2))
    y2 = max(0, min(img_h, y2))
    return x1, y1, x2, y2


def parse_yolo_label(path: Path, img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    if not path.exists():
        return boxes
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, xc, yc, bw, bh = parts[:5]
        boxes.append(yolo_to_xyxy(float(xc), float(yc), float(bw), float(bh), img_w, img_h))
    return boxes


def iou_xyxy(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def nms(dets: List[Detection], iou_thresh: float = 0.3) -> List[Detection]:
    if not dets:
        return []
    order = sorted(dets, key=lambda d: d.score, reverse=True)
    keep: List[Detection] = []
    while order:
        best = order.pop(0)
        keep.append(best)
        order = [
            d
            for d in order
            if iou_xyxy(best.xyxy, d.xyxy) < iou_thresh
        ]
    return keep


def crop_resize_gray(
    bgr: np.ndarray, xyxy: Sequence[int], win_size: Tuple[int, int]
) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    if x2 <= x1 or y2 <= y1:
        return None
    h, w = bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return cv2.resize(gray, win_size, interpolation=cv2.INTER_AREA)


def compute_hog_features(hog: cv2.HOGDescriptor, gray_win: np.ndarray) -> np.ndarray:
    feat = hog.compute(gray_win)
    return np.asarray(feat, dtype=np.float32).reshape(-1)


class HogSvmDetector:
    """Sliding-window HOG + linear SVM detector."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        confidence_threshold: float = 0.0,
        scale_factor: float = 1.25,
        step_size: int = 8,
        max_image_side: int = 640,
        nms_iou: float = 0.3,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.scale_factor = float(scale_factor)
        self.step_size = int(step_size)
        self.max_image_side = int(max_image_side)
        self.nms_iou = float(nms_iou)

        self.win_size = DEFAULT_WIN_SIZE
        self.hog = make_hog(self.win_size)
        self.weights: Optional[np.ndarray] = None  # includes bias as last element style
        self.bias: float = 0.0
        self.meta: dict = {}

        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"HOG-SVM model not found: {path}")

        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            self.weights = np.asarray(data["weights"], dtype=np.float32)
            self.bias = float(data["bias"])
            ws = tuple(data.get("win_size", list(DEFAULT_WIN_SIZE)))
            self.win_size = (int(ws[0]), int(ws[1]))
            self.hog = make_hog(self.win_size)
            self.meta = data.get("meta", {})
            return

        # OpenCV FileStorage (yml/yaml/xml) with keys weights, bias, win_w, win_h
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise RuntimeError(f"Failed to open model: {path}")
        weights = fs.getNode("weights").mat()
        bias_node = fs.getNode("bias")
        self.bias = float(bias_node.real())
        win_w = int(fs.getNode("win_w").real())
        win_h = int(fs.getNode("win_h").real())
        fs.release()
        self.weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        self.win_size = (win_w, win_h)
        self.hog = make_hog(self.win_size)

    def save(self, model_path: str | Path, meta: Optional[dict] = None) -> None:
        if self.weights is None:
            raise RuntimeError("No trained weights to save")
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": self.weights.astype(float).tolist(),
            "bias": float(self.bias),
            "win_size": [int(self.win_size[0]), int(self.win_size[1])],
            "meta": meta or self.meta,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

        # Also write OpenCV-readable sidecar for tooling that prefers FileStorage.
        yml = path.with_suffix(".yml")
        fs = cv2.FileStorage(str(yml), cv2.FILE_STORAGE_WRITE)
        fs.write("weights", self.weights.reshape(-1, 1))
        fs.write("bias", float(self.bias))
        fs.write("win_w", int(self.win_size[0]))
        fs.write("win_h", int(self.win_size[1]))
        fs.release()

    def set_linear_svm(self, weights: np.ndarray, bias: float, win_size: Tuple[int, int]) -> None:
        self.weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        self.bias = float(bias)
        self.win_size = (int(win_size[0]), int(win_size[1]))
        self.hog = make_hog(self.win_size)

    def score_window(self, gray_win: np.ndarray) -> float:
        if self.weights is None:
            raise RuntimeError("Model not loaded")
        feat = compute_hog_features(self.hog, gray_win)
        return float(np.dot(self.weights, feat) + self.bias)

    def _letterbox_scale(self, bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = bgr.shape[:2]
        m = max(h, w)
        if m <= self.max_image_side:
            return bgr, 1.0
        scale = self.max_image_side / float(m)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        return resized, scale

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        if self.weights is None:
            raise RuntimeError("Model not loaded")
        if bgr is None or bgr.size == 0:
            return []

        work, scale = self._letterbox_scale(bgr)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
        win_w, win_h = self.win_size
        dets: List[Detection] = []

        pyramid_scale = 1.0
        img = gray
        while True:
            h, w = img.shape[:2]
            if h < win_h or w < win_w:
                break
            y = 0
            while y + win_h <= h:
                x = 0
                while x + win_w <= w:
                    patch = img[y : y + win_h, x : x + win_w]
                    score = self.score_window(patch)
                    if score >= self.confidence_threshold:
                        # map back to original image coordinates
                        ox = int(round(x / (pyramid_scale * scale)))
                        oy = int(round(y / (pyramid_scale * scale)))
                        ow = int(round(win_w / (pyramid_scale * scale)))
                        oh = int(round(win_h / (pyramid_scale * scale)))
                        dets.append(Detection(ox, oy, ow, oh, score))
                    x += self.step_size
                y += self.step_size

            pyramid_scale /= self.scale_factor
            nw = int(round(gray.shape[1] * pyramid_scale))
            nh = int(round(gray.shape[0] * pyramid_scale))
            if nw < win_w or nh < win_h:
                break
            img = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

        return nms(dets, self.nms_iou)

    def annotate(self, bgr: np.ndarray, dets: Iterable[Detection]) -> np.ndarray:
        out = bgr.copy()
        for d in dets:
            cv2.rectangle(out, (d.x, d.y), (d.x + d.w, d.y + d.h), (0, 220, 0), 2)
            label = f"vial {d.score:.2f}"
            cv2.putText(
                out,
                label,
                (d.x, max(0, d.y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 0),
                1,
                cv2.LINE_AA,
            )
        return out


def resolve_default_model() -> Path:
    here = Path(__file__).resolve().parent.parent
    candidate = here / "models" / "hog_svm_vial.json"
    return candidate
