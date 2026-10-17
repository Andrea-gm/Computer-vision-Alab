"""SimpleBlobDetector + MSER + geometric priors for vial localization.

Strategy (tuned on vial_dataset/split_70_10_20):
  1. CLAHE + Gaussian blur; HSV white-cap mask (low S, high V).
  2. OpenCV SimpleBlobDetector on bright blobs (round tops / cap ellipses).
  3. MSER + connected-component contours on the white mask, scored by
     circularity / convexity / inertia / area / aspect.
  4. Expand each accepted cap region to a full-vial box using side-view
     geometric priors (width/height scales from labeled caps).
  5. Score threshold + greedy IoU NMS.

Clear plastic bodies are weak blob targets; white caps are the primary cue.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .types import Det, box_iou

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class BlobParams:
    confidence_threshold: float = 0.70
    max_detections: int = 2
    clahe_clip: float = 2.0
    gaussian_ksize: int = 5
    white_s_max: int = 70
    white_v_min: int = 170
    blob_min_area: float = 120.0
    blob_max_area: float = 25000.0
    blob_min_circularity: float = 0.55
    blob_min_inertia: float = 0.35
    blob_min_convexity: float = 0.75
    blob_min_dist: float = 20.0
    blob_min_size: float = 14.0
    blob_max_size: float = 220.0
    mser_delta: int = 4
    mser_min_area: int = 150
    mser_max_area: int = 20000
    geom_min_circularity: float = 0.50
    geom_min_convexity: float = 0.80
    geom_min_inertia: float = 0.30
    geom_min_aspect: float = 0.50
    geom_max_aspect: float = 2.00
    expand_width_scale: float = 2.4
    expand_height_scale: float = 3.6
    nms_iou: float = 0.35


@dataclass
class InferStats:
    backend: str = "blob_mser_geom"
    inference_ms: float = 0.0
    n_blob: int = 0
    n_mser: int = 0
    n_cc: int = 0
    n_kept: int = 0
    extras: Dict = field(default_factory=dict)


def list_images(folder: Path) -> List[Path]:
    if not folder or not Path(folder).exists():
        return []
    folder = Path(folder)
    files: List[Path] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def load_yolo_label(path: Path, width: int, height: int) -> List[Det]:
    if not path.exists():
        return []
    dets: List[Det] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        dets.append(Det(x1, y1, x2, y2, conf=1.0, cls_id=cls_id, label="vial", source="gt"))
    return dets


class VialBlobDetector:
    """Classical blob + MSER vial detector with tunable geometric priors."""

    def __init__(self, params: Optional[BlobParams] = None) -> None:
        self.params = params or BlobParams()
        self.backend = "blob_mser_geom"
        self._rebuild()

    def _rebuild(self) -> None:
        p = self.params
        bp = cv2.SimpleBlobDetector_Params()
        bp.filterByColor = True
        bp.blobColor = 255
        bp.minThreshold = 20
        bp.maxThreshold = 220
        bp.thresholdStep = 12
        bp.filterByArea = True
        bp.minArea = float(p.blob_min_area)
        bp.maxArea = float(p.blob_max_area)
        bp.filterByCircularity = True
        bp.minCircularity = float(p.blob_min_circularity)
        bp.filterByInertia = True
        bp.minInertiaRatio = float(p.blob_min_inertia)
        bp.filterByConvexity = True
        bp.minConvexity = float(p.blob_min_convexity)
        bp.minDistBetweenBlobs = float(p.blob_min_dist)
        self._blob = cv2.SimpleBlobDetector_create(bp)

        mser = cv2.MSER_create()
        mser.setDelta(int(p.mser_delta))
        mser.setMinArea(int(p.mser_min_area))
        mser.setMaxArea(int(p.mser_max_area))
        self._mser = mser

    def update_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, v)
        self._rebuild()

    def preprocess(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p = self.params
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=float(p.clahe_clip), tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        k = int(p.gaussian_ksize)
        if k % 2 == 0:
            k += 1
        k = max(3, k)
        blur = cv2.GaussianBlur(enhanced, (k, k), 0)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        mask = ((s < int(p.white_s_max)) & (v > int(p.white_v_min))).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return blur, mask

    def _score_contour(self, cnt, img_area: float) -> Optional[Dict]:
        p = self.params
        area = float(cv2.contourArea(cnt))
        if area < p.mser_min_area or area > p.mser_max_area or area > 0.15 * img_area:
            return None
        peri = float(cv2.arcLength(cnt, True))
        circ = 4.0 * np.pi * area / (peri * peri + 1e-6)
        hull = cv2.convexHull(cnt)
        conv = area / (float(cv2.contourArea(hull)) + 1e-6)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h + 1e-6)
        inertia = 1.0
        if len(cnt) >= 5:
            try:
                (_, _), (ma, mi), _ = cv2.fitEllipse(cnt)
                inertia = float(min(ma, mi) / (max(ma, mi) + 1e-6))
            except cv2.error:
                pass
        if circ < p.geom_min_circularity:
            return None
        if conv < p.geom_min_convexity:
            return None
        if inertia < p.geom_min_inertia:
            return None
        if not (p.geom_min_aspect <= aspect <= p.geom_max_aspect):
            return None
        score = (
            0.25 * circ
            + 0.25 * conv
            + 0.25 * inertia
            + 0.25 * min(1.0, area / 5000.0)
        )
        return {
            "score": float(score),
            "circ": circ,
            "conv": conv,
            "inertia": inertia,
            "aspect": aspect,
            "rect": (int(x), int(y), int(w), int(h)),
            "area": area,
        }

    def _expand_cap_to_vial(
        self, cx: float, cy: float, diameter: float, height: int, width: int
    ) -> Tuple[float, float, float, float]:
        p = self.params
        r = max(diameter * 0.5, 10.0)
        vw = float(np.clip(r * p.expand_width_scale, 24.0, 0.45 * width))
        vh = float(np.clip(r * p.expand_height_scale, 36.0, 0.75 * height))
        vcx = cx
        vcy = min(height - 1.0, cy + 0.4 * (vh - 2.0 * r))
        x1 = max(0.0, vcx - vw / 2.0)
        y1 = max(0.0, vcy - vh / 2.0)
        x2 = min(width - 1.0, vcx + vw / 2.0)
        y2 = min(height - 1.0, vcy + vh / 2.0)
        return x1, y1, x2, y2

    def _nms(self, cands: List[Tuple[Tuple[float, float, float, float], float, str]]) -> List[Det]:
        p = self.params
        cands = [c for c in cands if c[1] >= p.confidence_threshold]
        cands.sort(key=lambda t: t[1], reverse=True)
        kept: List[Det] = []
        for box, sc, src in cands:
            det = Det(box[0], box[1], box[2], box[3], conf=float(sc), source=src)
            if any(box_iou(det, k) > p.nms_iou for k in kept):
                continue
            kept.append(det)
            if len(kept) >= int(p.max_detections):
                break
        return kept

    def predict(self, bgr: np.ndarray) -> Tuple[List[Det], InferStats]:
        if bgr is None or bgr.size == 0:
            return [], InferStats(inference_ms=0.0)
        t0 = time.perf_counter()
        p = self.params
        h, w = bgr.shape[:2]
        img_area = float(h * w)
        blur, white = self.preprocess(bgr)
        cands: List[Tuple[Tuple[float, float, float, float], float, str]] = []
        n_blob = n_mser = n_cc = 0

        for img, src, boost in ((blur, "blob", 0.05), (white, "blob_mask", 0.12)):
            for kp in self._blob.detect(img):
                cx, cy = float(kp.pt[0]), float(kp.pt[1])
                d = float(kp.size)
                if d < p.blob_min_size or d > p.blob_max_size:
                    continue
                sc = 0.5 + boost + 0.002 * min(d, 80.0)
                sc += 0.05 * (1.0 - abs(cx / float(w) - 0.5) * 1.2)
                box = self._expand_cap_to_vial(cx, cy, d, h, w)
                cands.append((box, min(0.99, sc), src))
                n_blob += 1

        regions, _ = self._mser.detectRegions(white)
        for pts in regions:
            cnt = pts.reshape(-1, 1, 2)
            g = self._score_contour(cnt, img_area)
            if g is None:
                continue
            x, y, bw, bh = g["rect"]
            cx, cy = x + bw / 2.0, y + bh / 2.0
            box = self._expand_cap_to_vial(cx, cy, float(max(bw, bh)), h, w)
            cands.append((box, min(0.99, 0.4 + 0.6 * g["score"] + 0.1), "mser"))
            n_mser += 1

        nlab, labels, stats, cents = cv2.connectedComponentsWithStats(white, 8)
        for i in range(1, nlab):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < p.mser_min_area or area > p.mser_max_area:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            aspect = bw / float(bh + 1e-6)
            if not (0.45 <= aspect <= 2.2):
                continue
            mask = (labels == i).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            g = self._score_contour(cnts[0], img_area)
            if g is None:
                continue
            cx, cy = float(cents[i, 0]), float(cents[i, 1])
            box = self._expand_cap_to_vial(cx, cy, float(max(bw, bh)), h, w)
            cands.append((box, min(0.99, 0.45 + 0.55 * g["score"]), "cc"))
            n_cc += 1

        kept = self._nms(cands)
        ms = (time.perf_counter() - t0) * 1000.0
        return kept, InferStats(
            backend=self.backend,
            inference_ms=ms,
            n_blob=n_blob,
            n_mser=n_mser,
            n_cc=n_cc,
            n_kept=len(kept),
        )

    def annotate(self, bgr: np.ndarray, dets: Sequence[Det]) -> np.ndarray:
        out = bgr.copy()
        for d in dets:
            x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 80), 2)
            label = f"{d.label} {d.conf:.2f} {d.source}"
            cv2.putText(
                out,
                label,
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 80),
                1,
                cv2.LINE_AA,
            )
        return out
