"""Keep more raw cands; kill only large top-border wall slabs."""
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


def predict_many(det: VialBlobDetector, bgr: np.ndarray) -> list[Det]:
    p = det.params
    old_max, old_thr = p.max_detections, p.confidence_threshold
    p.max_detections = 25
    p.confidence_threshold = 0.25
    dets, _ = det.predict(bgr)
    p.max_detections, p.confidence_threshold = old_max, old_thr
    h, w = bgr.shape[:2]
    scored: list[tuple[Det, float]] = []
    for d in dets:
        bw, bh = d.x2 - d.x1, d.y2 - d.y1
        area = bw * bh / float(w * h)
        aspect = bh / max(bw, 1e-6)
        if aspect < 1.05 or aspect > 3.0:
            continue
        sc = float(d.conf)
        # Large top-glued wall / paper
        if d.y1 <= 3 and area >= 0.08:
            sc -= 0.55
        if d.source == "cc" and area >= 0.10:
            sc -= 0.12
        # Prefer mid-field
        cy = 0.5 * (d.y1 + d.y2) / h
        sc += 0.06 * (1.0 - abs(cy - 0.55))
        if sc < 0.30:
            continue
        d.conf = float(min(0.99, sc))
        scored.append((d, sc))
    scored.sort(key=lambda t: t[1], reverse=True)
    kept: list[Det] = []
    for d, _ in scored:
        if any(box_iou(d, k) > 0.35 for k in kept):
            continue
        kept.append(d)
        if len(kept) >= 2:
            break
    return kept


det = VialBlobDetector(
    BlobParams(
        confidence_threshold=0.35,
        white_v_min=160,
        blob_min_circularity=0.25,
        geom_min_circularity=0.25,
        geom_min_convexity=0.55,
        geom_min_inertia=0.12,
        blob_min_area=50,
        mser_min_area=50,
        expand_width_scale=2.4,
        expand_height_scale=3.6,
        max_detections=25,
    )
)

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
        preds = predict_many(det, im)
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
    f"topwall-pen rec={tp / (tp + fn):.3f} prec={tp / max(tp + fp, 1):.3f} "
    f"tp={tp} fp={fp} fn={fn} miou={float(np.mean(ious) if ious else 0):.3f}"
)

# Check former FNs
for rel in ["val/vial_0056.jpg", "test/vial_0012.jpg", "test/vial_0052.jpg"]:
    split, name = rel.split("/")
    im = cv2.imread(str(ROOT / "images" / split / name))
    h, w = im.shape[:2]
    g = load_yolo_label(ROOT / "labels" / split / f"{Path(name).stem}.txt", w, h)[0]
    preds = predict_many(det, im)
    best = max((box_iou(p, g) for p in preds), default=0.0)
    print(name, "best_iou", round(best, 3), "n", len(preds))
