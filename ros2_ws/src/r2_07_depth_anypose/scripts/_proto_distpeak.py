"""Strict blob + distance-transform peaks inside large white CCs."""
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
        max_detections=3,
    )
)


def dist_peaks(bgr: np.ndarray, existing: list[Det]) -> list[Det]:
    h, w = bgr.shape[:2]
    _, white = det.preprocess(bgr)
    dist = cv2.distanceTransform(white, cv2.DIST_L2, 5)
    # Peaks
    kernel = np.ones((15, 15), np.uint8)
    dil = cv2.dilate(dist, kernel)
    peaks = (dist == dil) & (dist >= 4.0)
    ys, xs = np.where(peaks)
    # Keep strongest peaks
    vals = dist[ys, xs]
    order = np.argsort(-vals)
    added: list[Det] = []
    for idx in order[:40]:
        cx, cy = float(xs[idx]), float(ys[idx])
        rad = float(dist[ys[idx], xs[idx]])
        diam = max(12.0, min(80.0, rad * 2.2))
        # Cap-like: prefer peaks not in huge empty wall centers — check local circularity via white ring
        for ws, hs in ((2.2, 3.2), (2.4, 3.6), (2.6, 3.8)):
            old = det.params.expand_width_scale, det.params.expand_height_scale
            det.params.expand_width_scale, det.params.expand_height_scale = ws, hs
            box = det._expand_cap_to_vial(cx, cy, diam, h, w)
            det.params.expand_width_scale, det.params.expand_height_scale = old
            d = Det(*box, conf=0.55 + 0.01 * rad, source="distpeak")
            bw, bh = d.x2 - d.x1, d.y2 - d.y1
            if bh / max(bw, 1e-6) < 1.05:
                continue
            area = bw * bh / float(w * h)
            if area < 0.008 or area > 0.45:
                continue
            if d.y1 <= 2 and area > 0.12:
                continue
            if any(box_iou(d, e) > 0.30 for e in existing + added):
                continue
            # Require some vertical edge energy
            xi, yi = int(d.x1), int(d.y1)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            roi = gray[yi : yi + int(bh), xi : xi + int(bw)]
            if roi.size < 30:
                continue
            gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
            lw = max(2, int(bw) // 6)
            edge = float(np.mean(np.abs(gx[:, :lw])) + np.mean(np.abs(gx[:, -lw:]))) / 70.0
            if edge < 0.16:
                continue
            added.append(d)
            break
        if len(added) >= 3:
            break
    return added


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
        preds, _ = det.predict(im)
        # Always try dist peaks as additional proposals
        preds = preds + dist_peaks(im, preds)
        # NMS
        preds = sorted(preds, key=lambda d: d.conf, reverse=True)
        kept: list[Det] = []
        for d in preds:
            # demote top-wall
            area = (d.x2 - d.x1) * (d.y2 - d.y1) / float(w * h)
            conf = d.conf
            if d.y1 <= 2 and area >= 0.08:
                conf -= 0.4
            if conf < 0.32:
                continue
            d.conf = conf
            if any(box_iou(d, k) > 0.35 for k in kept):
                continue
            kept.append(d)
            if len(kept) >= 2:
                break
        preds = kept
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
    f"distpeak rec={tp / (tp + fn):.3f} prec={tp / max(tp + fp, 1):.3f} "
    f"tp={tp} fp={fp} fn={fn} miou={float(np.mean(ious) if ious else 0):.3f}"
)
for rel in ["val/vial_0056.jpg", "test/vial_0012.jpg", "test/vial_0052.jpg"]:
    split, name = rel.split("/")
    im = cv2.imread(str(ROOT / "images" / split / name))
    h, w = im.shape[:2]
    g = load_yolo_label(ROOT / "labels" / split / f"{Path(name).stem}.txt", w, h)[0]
    preds, _ = det.predict(im)
    preds = preds + dist_peaks(im, preds)
    best = max((box_iou(p, g) for p in preds), default=0.0)
    print(name, "best_iou_any", round(best, 3))
