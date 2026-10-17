#!/usr/bin/env python3
"""Offline metrics for Agent #10 ensemble (mAP@0.5, accuracy, FPS, empties FP)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from agent_10_ensemble.ensemble_pipeline import EnsembleConfig, EnsembleVialDetector  # noqa: E402
from agent_10_ensemble.nms import box_iou  # noqa: E402


def yolo_to_xyxy(line: str, w: int, h: int):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    _, xc, yc, bw, bh = map(float, parts[:5])
    return (
        (xc - bw / 2.0) * w,
        (yc - bh / 2.0) * h,
        (xc + bw / 2.0) * w,
        (yc + bh / 2.0) * h,
    )


def load_split(split_root: Path, split: str):
    img_dir = split_root / "images" / split
    lbl_dir = split_root / "labels" / split
    items = []
    for img_path in sorted(img_dir.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        lbl = lbl_dir / (img_path.stem + ".txt")
        lines = []
        if lbl.is_file():
            lines = [ln for ln in lbl.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        items.append((img_path, lines))
    return items


def average_precision(recalls, precisions) -> float:
    recalls = np.asarray(recalls, dtype=np.float64)
    precisions = np.asarray(precisions, dtype=np.float64)
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def eval_map(all_preds, all_gts, iou_thresh: float = 0.5) -> float:
    """Single-class VOC-style AP@iou_thresh."""
    scores, matches = [], []
    npos = sum(len(g) for g in all_gts)
    if npos == 0:
        return 0.0
    for preds, gts in zip(all_preds, all_gts):
        matched = [False] * len(gts)
        for score, box in sorted(preds, key=lambda x: -x[0]):
            scores.append(score)
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gts):
                if matched[j]:
                    continue
                iou = box_iou(box, gt)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thresh and best_j >= 0:
                matched[best_j] = True
                matches.append(1)
            else:
                matches.append(0)
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    matches = np.asarray(matches)[order]
    tp = np.cumsum(matches)
    fp = np.cumsum(1 - matches)
    recalls = tp / float(npos)
    precisions = tp / np.maximum(tp + fp, 1e-9)
    return average_precision(recalls, precisions)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split",
        default=r"E:\Desktop\Jessica\robotics\vial_dataset\split_70_10_20_latest2",
    )
    ap.add_argument("--subset", default="test", choices=["train", "val", "test"])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--verifier-thresh", type=float, default=0.42)
    ap.add_argument("--verifier", default=str(_PKG_ROOT / "models" / "verifier.pkl"))
    ap.add_argument("--yolo-weights", default="")
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--camera-root", default=r"E:\Desktop\Jessica\robotics")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    cfg = EnsembleConfig(
        use_yolo=not args.no_yolo,
        yolo_weights=args.yolo_weights or None,
        verifier_path=args.verifier if Path(args.verifier).is_file() else None,
        verifier_thresh=args.verifier_thresh,
        camera_root=args.camera_root,
    )
    det = EnsembleVialDetector(cfg)
    items = load_split(Path(args.split), args.subset)
    if args.max_images > 0:
        items = items[: args.max_images]

    all_preds, all_gts = [], []
    tp = fp_empty = fn = correct = 0
    n_pos = n_empty = 0
    ious = []
    t0 = time.perf_counter()
    n_frames = 0

    for img_path, lines in items:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gts = [b for b in (yolo_to_xyxy(ln, w, h) for ln in lines) if b]
        t1 = time.perf_counter()
        preds = det.detect(bgr)
        t2 = time.perf_counter()
        n_frames += 1
        # accumulate time in loop via outer; per-frame below
        _ = t2 - t1

        pred_tuples = [(d.score, d.xyxy) for d in preds]
        all_preds.append(pred_tuples)
        all_gts.append(gts)

        if not gts:
            n_empty += 1
            if preds:
                fp_empty += 1
            else:
                correct += 1
            continue

        n_pos += 1
        if not preds:
            fn += 1
            continue
        best = max(box_iou(preds[0].xyxy, g) for g in gts)
        ious.append(best)
        if best >= args.iou:
            tp += 1
            correct += 1
        else:
            fn += 1

    elapsed = time.perf_counter() - t0
    fps = n_frames / elapsed if elapsed > 0 else 0.0
    ap50 = eval_map(all_preds, all_gts, args.iou)
    mean_iou = float(np.mean(ious)) if ious else 0.0
    acc = correct / max(1, n_pos + n_empty)
    recall = tp / max(1, n_pos)

    report = {
        "agent": "agent_10_ensemble",
        "split": str(args.split),
        "subset": args.subset,
        "n_images": n_frames,
        "n_positives": n_pos,
        "n_empties": n_empty,
        "iou_thresh": args.iou,
        "mAP_50": ap50,
        "accuracy_iou": acc,
        "recall_iou": recall,
        "tp": tp,
        "fn": fn,
        "empty_fp": fp_empty,
        "mean_best_iou": mean_iou,
        "fps": fps,
        "seconds": elapsed,
        "yolo_weights": det.yolo_weights,
        "verifier": str(det.verifier.model_path) if det.verifier.model_path else "heuristic",
        "verifier_thresh": args.verifier_thresh,
    }

    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
