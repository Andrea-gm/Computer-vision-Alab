#!/usr/bin/env python3
"""Evaluate HOG+SVM vial detector on YOLO-labeled split (precision/recall/IoU)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from agent_07_hog_svm.hog_detector import (  # noqa: E402
    HogSvmDetector,
    iou_xyxy,
    parse_yolo_label,
    resolve_default_model,
)


def list_pairs(split_root: Path, split: str) -> List[Tuple[Path, Path]]:
    images_dir = split_root / "images" / split
    labels_dir = split_root / "labels" / split
    pairs = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        pairs.append((img, labels_dir / f"{img.stem}.txt"))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"E:\Desktop\Jessica\robotics\vial_dataset\split_70_10_20"),
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "validate", "test"])
    parser.add_argument("--model", type=Path, default=resolve_default_model())
    parser.add_argument("--conf", type=float, default=0.0)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-image-side", type=int, default=480)
    parser.add_argument("--step-size", type=int, default=8)
    parser.add_argument("--out", type=Path, default=_PKG_ROOT / "models" / "eval_metrics.json")
    args = parser.parse_args()

    split = args.split
    if split == "val" and not (args.dataset / "images" / "val").exists():
        split = "validate"

    pairs = list_pairs(args.dataset, split)
    if not pairs:
        print(f"No images in {args.dataset}/images/{split}")
        return 1

    detector = HogSvmDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        max_image_side=args.max_image_side,
        step_size=args.step_size,
    )

    tp = fp = fn = 0
    empty_fp = 0
    ious: List[float] = []
    times: List[float] = []
    per_image = []

    for img_path, label_path in pairs:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gt = parse_yolo_label(label_path, w, h)
        t0 = time.perf_counter()
        dets = detector.detect(bgr)
        times.append(time.perf_counter() - t0)

        img_tp = img_fp = 0
        matched = set()
        if not gt:
            empty_fp += len(dets)
            fp += len(dets)
            img_fp = len(dets)
        else:
            for d in dets:
                best_i, best_iou = -1, 0.0
                for i, g in enumerate(gt):
                    if i in matched:
                        continue
                    v = iou_xyxy(d.xyxy, g)
                    if v > best_iou:
                        best_iou, best_i = v, i
                if best_iou >= args.iou and best_i >= 0:
                    tp += 1
                    img_tp += 1
                    matched.add(best_i)
                    ious.append(best_iou)
                else:
                    fp += 1
                    img_fp += 1
            fn += len(gt) - len(matched)

        per_image.append(
            {
                "image": img_path.name,
                "gt": len(gt),
                "pred": len(dets),
                "tp": img_tp,
                "fp": img_fp,
            }
        )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_ms = float(np.mean(times) * 1000.0) if times else 0.0

    report = {
        "algorithm": "HOG + Linear SVM",
        "model": str(args.model),
        "dataset": str(args.dataset),
        "split": split,
        "n_images": len(pairs),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "mean_matched_iou": mean_iou,
        "empty_fp": empty_fp,
        "mean_infer_ms": mean_ms,
        "conf": args.conf,
        "iou_thresh": args.iou,
        "per_image": per_image,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_image"}, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
