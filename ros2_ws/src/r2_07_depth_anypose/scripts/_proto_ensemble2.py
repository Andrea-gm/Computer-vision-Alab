"""Strict-first ensemble: never let loose suppress strict hits."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent_05_blob_geometry"))
from agent_05_blob_geometry.blob_detector import (  # noqa: E402
    BlobParams,
    VialBlobDetector,
    load_yolo_label,
)
from agent_05_blob_geometry.types import Det, box_iou  # noqa: E402

ROOT = Path(r"E:\Desktop\Jessica\robotics\vial_dataset\split_70_10_20")

strict = VialBlobDetector(
    BlobParams(
        confidence_threshold=0.35,
        white_v_min=170,
        blob_min_circularity=0.25,
        geom_min_circularity=0.25,
        geom_min_convexity=0.55,
        geom_min_inertia=0.12,
        blob_min_area=60,
        mser_min_area=80,
        expand_width_scale=2.4,
        expand_height_scale=3.6,
        max_detections=3,
    )
)
loose = VialBlobDetector(
    BlobParams(
        confidence_threshold=0.30,
        white_v_min=135,
        white_s_max=105,
        blob_min_circularity=0.20,
        geom_min_circularity=0.20,
        geom_min_convexity=0.50,
        geom_min_inertia=0.10,
        blob_min_area=35,
        blob_min_size=7.0,
        mser_min_area=35,
        expand_width_scale=2.3,
        expand_height_scale=3.4,
        max_detections=5,
    )
)


def struct_ok(bgr: np.ndarray, d: Det) -> bool:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = map(int, [d.x1, d.y1, d.x2, d.y2])
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    aspect = bh / float(bw)
    area = bw * bh / float(w * h)
    if aspect < 1.05 or aspect > 2.9:
        return False
    if area < 0.007 or area > 0.50:
        return False
    borders = int(x1 <= 2) + int(y1 <= 2) + int(x2 >= w - 3) + int(y2 >= h - 3)
    if borders >= 2:
        return False
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    roi = gray[y1:y2, x1:x2]
    if roi.size < 20:
        return False
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    lw = max(2, bw // 6)
    edge = float(np.mean(np.abs(gx[:, :lw])) + np.mean(np.abs(gx[:, -lw:]))) / 70.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    top = hsv[y1 : y1 + max(2, bh // 5), x1:x2]
    tw = float(((top[:, :, 2] >= 130) & (top[:, :, 1] <= 100)).mean()) if top.size else 0
    return edge >= 0.18 or (tw >= 0.12 and edge >= 0.10)


def predict(bgr: np.ndarray) -> list[Det]:
    kept, _ = strict.predict(bgr)
    extra, _ = loose.predict(bgr)
    for d in extra:
        if not struct_ok(bgr, d):
            continue
        if any(box_iou(d, k) > 0.30 for k in kept):
            continue
        kept.append(d)
        if len(kept) >= 3:
            break
    return kept[:2]


tp = fp = fn = 0
ious: list[float] = []
for split in ("val", "test"):
    for p in list((ROOT / "images" / split).glob("*.jpg")) + list(
        (ROOT / "images" / split).glob("*.png")
    ):
        im = cv2.imread(str(p))
        h, w = im.shape[:2]
        gts = [
            g
            for g in load_yolo_label(ROOT / "labels" / split / f"{p.stem}.txt", w, h)
            if (g.y2 - g.y1) / (g.x2 - g.x1 + 1e-6) >= 1.05
        ]
        preds = predict(im)
        if not gts:
            fp += len(preds)
            continue
        used: set[int] = set()
        for g in gts:
            bi, bj = -1, 0.0
            for i, pr in enumerate(preds):
                if i in used:
                    continue
                v = box_iou(pr, g)
                if v > bj:
                    bi, bj = i, v
            if bi >= 0 and bj >= 0.5:
                used.add(bi)
                tp += 1
                ious.append(bj)
            else:
                fn += 1
        fp += len(preds) - len(used)

print(
    f"strict-first rec={tp / (tp + fn):.3f} prec={tp / max(tp + fp, 1):.3f} "
    f"tp={tp} fp={fp} fn={fn} miou={float(np.mean(ious) if ious else 0):.3f}"
)
