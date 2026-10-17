#!/usr/bin/env python3
"""Evaluate watershed/GrabCut vial detector on the labeled dataset.

Reports precision / recall / mean IoU when YOLO GT labels exist.
No ROS required.

Usage:
  cd E:\\Desktop\\Jessica\\Camera\\ros2_ws\\src\\agent_09_watershed_grabcut
  python scripts/evaluate.py
  python scripts/evaluate.py --split val --conf 0.22
  python scripts/evaluate.py --dataset E:\\Desktop\\Jessica\\Camera\\vial_dataset\\raw
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_09_watershed_grabcut.segmenter import (  # noqa: E402
    Det,
    WatershedGrabCutDetector,
    box_iou,
    list_images,
    load_yolo_label,
)


def _pairs_from_raw(raw_dir: Path) -> List[Tuple[Path, Path]]:
    return [(img, img.with_suffix(".txt")) for img in list_images(raw_dir)]


def _pairs_from_split(split_dir: Path, split: str) -> List[Tuple[Path, Path]]:
    name = "val" if split in ("val", "validate") else split
    img_dir = split_dir / "images" / name
    lbl_dir = split_dir / "labels" / name
    if not img_dir.exists():
        return []
    return [(img, lbl_dir / f"{img.stem}.txt") for img in list_images(img_dir)]


def match_dets(
    preds: Sequence[Det],
    gts: Sequence[Det],
    *,
    iou_thr: float,
) -> Tuple[int, int, int, List[float]]:
    if not gts and not preds:
        return 0, 0, 0, []
    if not gts:
        return 0, len(preds), 0, []
    if not preds:
        return 0, 0, len(gts), []
    used_gt = set()
    tps = 0
    ious: List[float] = []
    for p in sorted(preds, key=lambda d: d.conf, reverse=True):
        best_i = -1
        best_iou = 0.0
        for gi, g in enumerate(gts):
            if gi in used_gt:
                continue
            iou = box_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_i = gi
        if best_i >= 0 and best_iou >= iou_thr:
            used_gt.add(best_i)
            tps += 1
            ious.append(best_iou)
    fps = len(preds) - tps
    fns = len(gts) - tps
    return tps, fps, fns, ious


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate agent_09 watershed/GrabCut")
    ap.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Image folder (raw) or omit to use --split-dir",
    )
    ap.add_argument(
        "--split-dir",
        type=str,
        default=r"E:\Desktop\Jessica\robotics\vial_dataset\split_70_10_20_latest2",
    )
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--iou", type=float, default=0.5, help="Match IoU threshold")
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--grabcut-iters", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="Max images (0=all)")
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    if args.dataset:
        pairs = _pairs_from_raw(Path(args.dataset))
        source = args.dataset
    else:
        pairs = _pairs_from_split(Path(args.split_dir), args.split)
        source = f"{args.split_dir} [{args.split}]"
    if not pairs:
        print(f"No image/label pairs found under {source}", file=sys.stderr)
        return 2
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    detector = WatershedGrabCutDetector(
        confidence_threshold=args.conf,
        max_side=args.max_side,
        grabcut_iters=args.grabcut_iters,
    )

    tp = fp = fn = 0
    all_ious: List[float] = []
    best_ious: List[float] = []
    times: List[float] = []
    frames_with_det = 0
    confs: List[float] = []
    n_gt_total = 0

    for img_path, lbl_path in pairs:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gts = load_yolo_label(lbl_path, w, h)
        n_gt_total += len(gts)
        t0 = time.perf_counter()
        preds, stats = detector.predict(bgr)
        times.append(stats.inference_ms)
        if preds:
            frames_with_det += 1
            confs.append(max(d.conf for d in preds))
        tps, fps_, fns, ious = match_dets(preds, gts, iou_thr=args.iou)
        tp += tps
        fp += fps_
        fn += fns
        all_ious.extend(ious)
        if gts:
            best = 0.0
            for g in gts:
                for p in preds:
                    best = max(best, box_iou(p, g))
            best_ious.append(best)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    mean_match_iou = float(sum(all_ious) / len(all_ious)) if all_ious else 0.0
    mean_best_iou = float(sum(best_ious) / len(best_ious)) if best_ious else 0.0
    mean_ms = float(sum(times) / len(times)) if times else 0.0

    has_gt = n_gt_total > 0
    report = {
        "agent": "agent_09_watershed_grabcut",
        "backend": "watershed_grabcut",
        "source": source,
        "n_images": len(pairs),
        "n_gt_boxes": n_gt_total,
        "has_gt": has_gt,
        "iou_match_threshold": args.iou,
        "confidence_threshold": args.conf,
        "precision": precision if has_gt else None,
        "recall": recall if has_gt else None,
        "mean_matched_iou": mean_match_iou if has_gt else None,
        "mean_best_iou_per_gt_image": mean_best_iou if has_gt else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "frames_with_detection": frames_with_det,
        "mean_top_confidence": float(sum(confs) / len(confs)) if confs else 0.0,
        "mean_inference_ms": mean_ms,
        "fps_approx": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
    }

    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
