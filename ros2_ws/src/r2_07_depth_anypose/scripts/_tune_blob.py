"""Quick param sweep for upright still recall (dev helper)."""
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
from agent_05_blob_geometry.types import box_iou  # noqa: E402

ROOT = Path(r"E:\Desktop\Jessica\robotics\vial_dataset\split_70_10_20")

CONFIGS = [
    dict(
        confidence_threshold=0.50,
        white_v_min=170,
        blob_min_circularity=0.35,
        geom_min_circularity=0.35,
        geom_min_convexity=0.65,
        geom_min_inertia=0.18,
        expand_width_scale=2.4,
        expand_height_scale=3.6,
        max_detections=2,
    ),
    dict(
        confidence_threshold=0.40,
        white_v_min=165,
        blob_min_circularity=0.30,
        geom_min_circularity=0.30,
        geom_min_convexity=0.60,
        geom_min_inertia=0.15,
        expand_width_scale=2.4,
        expand_height_scale=3.6,
        max_detections=3,
    ),
    dict(
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
    ),
    dict(
        confidence_threshold=0.45,
        white_v_min=170,
        blob_min_circularity=0.40,
        geom_min_circularity=0.38,
        geom_min_convexity=0.70,
        geom_min_inertia=0.20,
        expand_width_scale=2.0,
        expand_height_scale=2.8,
        max_detections=2,
    ),
    dict(
        confidence_threshold=0.45,
        white_v_min=170,
        blob_min_circularity=0.40,
        geom_min_circularity=0.38,
        geom_min_convexity=0.70,
        geom_min_inertia=0.20,
        expand_width_scale=2.8,
        expand_height_scale=4.0,
        max_detections=2,
    ),
]


def run(kw: dict) -> None:
    det = VialBlobDetector(BlobParams(**kw))
    tp = fp = fn = 0
    ious: list[float] = []
    for split in ("val", "test"):
        imgs = list((ROOT / "images" / split).glob("*.jpg")) + list(
            (ROOT / "images" / split).glob("*.png")
        )
        for p in imgs:
            im = cv2.imread(str(p))
            h, w = im.shape[:2]
            gts = [
                g
                for g in load_yolo_label(ROOT / "labels" / split / f"{p.stem}.txt", w, h)
                if (g.y2 - g.y1) / (g.x2 - g.x1 + 1e-6) >= 1.05
            ]
            preds, _ = det.predict(im)
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
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    print(
        f"rec={rec:.3f} prec={prec:.3f} tp={tp} fp={fp} fn={fn} "
        f"miou={float(np.mean(ious) if ious else 0):.3f} "
        f"circ={kw.get('blob_min_circularity')} "
        f"expH={kw.get('expand_height_scale')} maxd={kw.get('max_detections')}"
    )


if __name__ == "__main__":
    for c in CONFIGS:
        run(c)
