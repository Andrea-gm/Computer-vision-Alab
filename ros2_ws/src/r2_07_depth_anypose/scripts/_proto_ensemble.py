"""Prototype: strict agent_05 + loose structure-gated tier."""
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
        confidence_threshold=0.45,
        white_v_min=170,
        white_s_max=75,
        blob_min_circularity=0.30,
        geom_min_circularity=0.28,
        geom_min_convexity=0.60,
        geom_min_inertia=0.15,
        blob_min_area=50,
        mser_min_area=60,
        expand_width_scale=2.4,
        expand_height_scale=3.6,
        max_detections=3,
    )
)
loose = VialBlobDetector(
    BlobParams(
        confidence_threshold=0.35,
        white_v_min=138,
        white_s_max=100,
        blob_min_circularity=0.22,
        geom_min_circularity=0.22,
        geom_min_convexity=0.55,
        geom_min_inertia=0.12,
        blob_min_area=40,
        blob_min_size=8.0,
        mser_min_area=40,
        expand_width_scale=2.4,
        expand_height_scale=3.6,
        max_detections=4,
    )
)


def struct_ok(bgr: np.ndarray, d: Det) -> bool:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = map(int, [d.x1, d.y1, d.x2, d.y2])
    bw, bh = x2 - x1, y2 - y1
    if bw < 12 or bh < 16:
        return False
    aspect = bh / float(bw)
    area = bw * bh / float(w * h)
    if aspect < 1.05 or aspect > 2.8:
        return False
    if area < 0.006 or area > 0.55:
        return False
    borders = int(x1 <= 2) + int(y1 <= 2) + int(x2 >= w - 3) + int(y2 >= h - 3)
    if borders >= 2 and area < 0.25:
        return False
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    roi = gray[y1:y2, x1:x2]
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    lw = max(2, bw // 6)
    edge = float(np.mean(np.abs(gx[:, :lw])) + np.mean(np.abs(gx[:, -lw:]))) / 80.0
    return edge >= 0.15 or area >= 0.20


def predict(bgr: np.ndarray) -> list[Det]:
    a, _ = strict.predict(bgr)
    b, _ = loose.predict(bgr)
    b = [d for d in b if struct_ok(bgr, d)]
    cands = [(d, d.conf + (0.08 if d in a else 0.0)) for d in a] + [
        (d, d.conf * 0.95) for d in b
    ]
    cands.sort(key=lambda t: t[1], reverse=True)
    kept: list[Det] = []
    for d, sc in cands:
        d.conf = float(min(0.99, sc))
        if any(box_iou(d, k) > 0.35 for k in kept):
            continue
        kept.append(d)
        if len(kept) >= 2:
            break
    return kept


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
    f"ensemble rec={tp / (tp + fn):.3f} prec={tp / max(tp + fp, 1):.3f} "
    f"tp={tp} fp={fp} fn={fn} miou={float(np.mean(ious) if ious else 0):.3f}"
)
