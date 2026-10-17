"""Classical feature extractor + lightweight verifier for vial proposals."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .types import Detection

FEATURE_DIM = 50


def extract_features(bgr: np.ndarray, det: Detection) -> np.ndarray:
    """Compact RGB features that encode white-cap + clear-body structure."""
    h, w = bgr.shape[:2]
    d = det.clip(w, h)
    x1, y1, x2, y2 = d.as_int()
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aspect = bw / float(bh)
    area_frac = (bw * bh) / float(max(1, w * h))

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    white = ((sat <= 55) & (val >= 150)).astype(np.float32)
    white_frac = float(white.mean())

    # Band fractions — upright caps sit near the top; sideways at left/right.
    bands = []
    for frac0, frac1 in ((0.0, 0.22), (0.22, 0.45), (0.45, 0.75), (0.75, 1.0)):
        a = int(frac0 * bh)
        b = max(a + 1, int(frac1 * bh))
        bands.append(float(white[a:b, :].mean()) if b > a else 0.0)
    side_bands = []
    for frac0, frac1 in ((0.0, 0.25), (0.75, 1.0)):
        a = int(frac0 * bw)
        b = max(a + 1, int(frac1 * bw))
        side_bands.append(float(white[:, a:b].mean()) if b > a else 0.0)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 110)
    edge_frac = float(edges.mean()) / 255.0

    # Side-wall edge energy (clear cup cue).
    col_w = max(1, bw // 8)
    left_e = float(edges[:, :col_w].mean()) / 255.0
    right_e = float(edges[:, -col_w:].mean()) / 255.0
    top_e = float(edges[: max(1, bh // 8), :].mean()) / 255.0
    bot_e = float(edges[-max(1, bh // 8) :, :].mean()) / 255.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = (cv2.phase(gx, gy, angleInDegrees=True) / 30.0).astype(np.int32) % 6
    # 2x2 spatial x 6 orientation soft HOG → 24 dims
    hog = []
    for yi in range(2):
        for xi in range(2):
            y0 = yi * bh // 2
            y1b = (yi + 1) * bh // 2
            x0 = xi * bw // 2
            x1b = (xi + 1) * bw // 2
            cell_ang = ang[y0:y1b, x0:x1b]
            cell_mag = mag[y0:y1b, x0:x1b]
            hist = np.zeros(6, dtype=np.float32)
            if cell_ang.size:
                for b in range(6):
                    hist[b] = float(cell_mag[cell_ang == b].sum())
                s = float(hist.sum()) + 1e-6
                hist /= s
            hog.extend(hist.tolist())

    sat_mean = float(sat.mean()) / 255.0
    sat_std = float(sat.std()) / 255.0
    val_mean = float(val.mean()) / 255.0
    val_std = float(val.std()) / 255.0

    # Cap location prior: max of top/left/right white bands.
    end_cap = max(bands[0], side_bands[0], side_bands[1])
    body_white = bands[2]  # mid-lower should be lower for clear cup
    cap_body_contrast = max(0.0, end_cap - body_white)

    yolo_conf = float(det.extras.get("yolo_conf", 0.0))
    source_cap = 1.0 if det.source == "cap" else 0.0
    source_edge = 1.0 if det.source == "edge" else 0.0
    source_yolo = 1.0 if det.source == "yolo" else 0.0

    feats = [
        aspect,
        area_frac,
        white_frac,
        *bands,
        *side_bands,
        edge_frac,
        left_e,
        right_e,
        top_e,
        bot_e,
        sat_mean,
        sat_std,
        val_mean,
        val_std,
        end_cap,
        body_white,
        cap_body_contrast,
        yolo_conf,
        source_cap,
        source_edge,
        source_yolo,
        float(det.score),
        *hog,
    ]
    arr = np.asarray(feats, dtype=np.float32)
    if arr.size < FEATURE_DIM:
        arr = np.pad(arr, (0, FEATURE_DIM - arr.size))
    elif arr.size > FEATURE_DIM:
        arr = arr[:FEATURE_DIM]
    return arr


def heuristic_score(feats: np.ndarray) -> float:
    """Fallback verifier when no sklearn model is loaded."""
    # Layout must match extract_features packing.
    aspect = float(feats[0])
    white_frac = float(feats[2])
    end_cap = float(feats[18])
    body_white = float(feats[19])
    cap_body = float(feats[20])
    edge_frac = float(feats[9])
    yolo_conf = float(feats[21])
    prior = float(feats[25])  # original proposal score

    aspect_term = 1.0 - min(1.0, abs(aspect - 0.65) / 0.55)
    score = (
        0.22 * aspect_term
        + 0.18 * min(white_frac / 0.25, 1.0)
        + 0.20 * min(end_cap / 0.35, 1.0)
        + 0.12 * min(cap_body / 0.25, 1.0)
        + 0.10 * min(edge_frac / 0.15, 1.0)
        + 0.10 * yolo_conf
        + 0.08 * prior
        - 0.15 * max(0.0, body_white - 0.45)  # reject blank white walls
    )
    return float(np.clip(score, 0.0, 1.0))


class Verifier:
    """Stage-3 scorer: trained logistic regression or heuristic fallback."""

    def __init__(self, model_path: str | Path | None = None):
        self.model: Any = None
        self.model_path = Path(model_path) if model_path else None
        if self.model_path and self.model_path.is_file():
            self.load(self.model_path)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            blob = pickle.load(f)
        self.model = blob.get("model", blob)
        self.model_path = Path(path)

    def save(self, path: str | Path, model: Any, meta: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": model, "meta": meta or {}, "feature_dim": FEATURE_DIM}, f)
        self.model = model
        self.model_path = path

    def score(self, bgr: np.ndarray, det: Detection) -> float:
        feats = extract_features(bgr, det)
        if self.model is not None:
            try:
                if hasattr(self.model, "predict_proba"):
                    proba = self.model.predict_proba(feats.reshape(1, -1))[0]
                    # Assume class 1 is vial.
                    classes = list(getattr(self.model, "classes_", [0, 1]))
                    if 1 in classes:
                        return float(proba[classes.index(1)])
                    return float(proba[-1])
                if hasattr(self.model, "decision_function"):
                    raw = float(self.model.decision_function(feats.reshape(1, -1))[0])
                    return float(1.0 / (1.0 + np.exp(-raw)))
            except Exception:
                pass
        return heuristic_score(feats)

    def accept(self, bgr: np.ndarray, det: Detection, thresh: float) -> tuple[bool, float]:
        s = self.score(bgr, det)
        return s >= thresh, s
