"""Rerank agent_05 cands: penalize wall slabs, keep more raw then filter."""
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


def predict_rerank(det: VialBlobDetector, bgr: np.ndarray) -> list[Det]:
    """Copy of predict but keep more cands and rerank."""
    p = det.params
    h, w = bgr.shape[:2]
    img_area = float(h * w)
    blur, white = det.preprocess(bgr)
    cands: list[tuple[tuple[float, float, float, float], float, str]] = []

    for img, src, boost in ((blur, "blob", 0.08), (white, "blob_mask", 0.14)):
        for kp in det._blob.detect(img):
            cx, cy = float(kp.pt[0]), float(kp.pt[1])
            d = float(kp.size)
            if d < p.blob_min_size or d > p.blob_max_size:
                continue
            sc = 0.52 + boost + 0.002 * min(d, 80.0)
            for ws, hs in ((2.2, 3.2), (2.4, 3.6), (2.6, 3.8)):
                # temporary override scales
                old = p.expand_width_scale, p.expand_height_scale
                p.expand_width_scale, p.expand_height_scale = ws, hs
                box = det._expand_cap_to_vial(cx, cy, d, h, w)
                p.expand_width_scale, p.expand_height_scale = old
                cands.append((box, sc, src))

    regions, _ = det._mser.detectRegions(white)
    for pts in regions:
        cnt = pts.reshape(-1, 1, 2)
        g = det._score_contour(cnt, img_area)
        if g is None:
            continue
        x, y, bw, bh = g["rect"]
        cx, cy = x + bw / 2.0, y + bh / 2.0
        box = det._expand_cap_to_vial(cx, cy, float(max(bw, bh)), h, w)
        cands.append((box, min(0.99, 0.45 + 0.55 * g["score"] + 0.08), "mser"))

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
        g = det._score_contour(cnts[0], img_area)
        if g is None:
            continue
        cx, cy = float(cents[i, 0]), float(cents[i, 1])
        box = det._expand_cap_to_vial(cx, cy, float(max(bw, bh)), h, w)
        # CC wall slabs get lower base score
        cands.append((box, min(0.99, 0.35 + 0.45 * g["score"]), "cc"))

    # Rerank
    scored: list[tuple[Det, float]] = []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    for box, sc, src in cands:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 14:
            continue
        aspect = bh / max(bw, 1e-6)
        area = bw * bh / img_area
        if aspect < 1.05 or aspect > 3.0:
            continue
        if area < 0.005 or area > 0.70:
            continue
        borders = (
            int(x1 <= 2) + int(y1 <= 2) + int(x2 >= w - 3) + int(y2 >= h - 3)
        )
        pen = 0.20 * borders
        if src == "cc":
            pen += 0.08
        xi, yi = int(x1), int(y1)
        bwi, bhi = int(bw), int(bh)
        roi = gray[yi : yi + bhi, xi : xi + bwi]
        if roi.size < 20:
            continue
        gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        lw = max(2, bwi // 6)
        edge = float(np.mean(np.abs(gx[:, :lw])) + np.mean(np.abs(gx[:, -lw:]))) / 80.0
        final = sc + 0.15 * min(1.0, edge) - pen
        if final < 0.40:
            continue
        scored.append((Det(x1, y1, x2, y2, conf=float(min(0.99, final)), source=src), final))

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
        white_v_min=165,
        blob_min_circularity=0.25,
        geom_min_circularity=0.25,
        geom_min_convexity=0.55,
        geom_min_inertia=0.12,
        blob_min_area=50,
        mser_min_area=50,
        expand_width_scale=2.4,
        expand_height_scale=3.6,
        max_detections=20,
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
        preds = predict_rerank(det, im)
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
    f"rerank rec={tp / (tp + fn):.3f} prec={tp / max(tp + fp, 1):.3f} "
    f"tp={tp} fp={fp} fn={fn} miou={float(np.mean(ious) if ious else 0):.3f}"
)
