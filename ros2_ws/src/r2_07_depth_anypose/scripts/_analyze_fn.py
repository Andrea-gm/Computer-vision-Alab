"""List upright FN/FP cases for agent_05 blob detector."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent_05_blob_geometry"))
from agent_05_blob_geometry.blob_detector import (  # noqa: E402
    BlobParams,
    VialBlobDetector,
    load_yolo_label,
)
from agent_05_blob_geometry.types import box_iou  # noqa: E402

ROOT = Path(r"E:\Desktop\Jessica\robotics\vial_dataset\split_70_10_20")
det = VialBlobDetector(
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

fns = []
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
        if not gts:
            continue
        preds, _ = det.predict(im)
        for g in gts:
            best = max((box_iou(pr, g) for pr in preds), default=0.0)
            if best < 0.5:
                ar = (g.y2 - g.y1) / (g.x2 - g.x1 + 1e-6)
                area = (g.x2 - g.x1) * (g.y2 - g.y1) / (w * h)
                fns.append((best, ar, area, split, p.name, w, h))

fns.sort()
print(f"n_fn={len(fns)}")
for row in fns[:25]:
    print(
        f"iou={row[0]:.3f} ar={row[1]:.2f} area={row[2]:.3f} {row[3]}/{row[4]} {row[5]}x{row[6]}"
    )
