#!/usr/bin/env python3
"""Evaluate blob+geometry vial detector on the labeled dataset.

Reports precision / recall / mean IoU at IoU≥0.5 when YOLO labels exist.
Otherwise qualitative score + timing.

Usage (no ROS required):
  cd E:\\Desktop\\Jessica\\Camera\\ros2_ws\\src\\agent_05_blob_geometry
  python scripts/evaluate.py
  python scripts/evaluate.py --split test --conf 0.70
  python scripts/evaluate.py --dataset E:\\Desktop\\Jessica\\Camera\\vial_photos
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_05_blob_geometry.blob_detector import (  # noqa: E402
    BlobParams,
    VialBlobDetector,
    list_images,
    load_yolo_label,
)
from agent_05_blob_geometry.types import Det, box_iou  # noqa: E402

DEFAULT_CAMERA = Path(r"E:\Desktop\Jessica\robotics")
DEFAULT_SPLIT = DEFAULT_CAMERA / "vial_dataset" / "split_70_10_20"


def _pairs_from_dir(folder: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for img in list_images(folder):
        pairs.append((img, img.with_suffix(".txt")))
    return pairs


def _pairs_from_split(split_dir: Path, split: str) -> List[Tuple[Path, Path]]:
    name = "val" if split in ("val", "validate") else split
    img_dir = split_dir / "images" / name
    lbl_dir = split_dir / "labels" / name
    if not img_dir.exists():
        return []
    pairs = []
    for img in list_images(img_dir):
        pairs.append((img, lbl_dir / f"{img.stem}.txt"))
    return pairs


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
    ordered = sorted(preds, key=lambda d: d.conf, reverse=True)
    for p in ordered:
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


def evaluate(
    pairs: List[Tuple[Path, Path]],
    detector: VialBlobDetector,
    *,
    iou_thr: float,
    max_images: Optional[int],
) -> dict:
    if max_images is not None:
        pairs = pairs[: max(0, int(max_images))]

    has_any_gt = any(p[1].exists() for p in pairs)
    tp = fp = fn = 0
    all_ious: List[float] = []
    times_ms: List[float] = []
    frames_with_det = 0
    conf_sum = 0.0
    conf_n = 0
    empty_fp = 0
    pos_images = 0
    empty_images = 0
    pos_correct = 0

    for img_path, lbl_path in pairs:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gts = load_yolo_label(lbl_path, w, h) if lbl_path.exists() else []
        is_empty = lbl_path.exists() and (not gts)
        if is_empty:
            empty_images += 1
        elif gts:
            pos_images += 1

        t0 = time.perf_counter()
        preds, stats = detector.predict(bgr)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        if stats.inference_ms > 0:
            times_ms[-1] = stats.inference_ms

        if preds:
            frames_with_det += 1
            conf_sum += max(d.conf for d in preds)
            conf_n += 1

        if has_any_gt:
            t, f, n, ious = match_dets(preds, gts, iou_thr=iou_thr)
            tp += t
            fp += f
            fn += n
            all_ious.extend(ious)
            if is_empty and preds:
                empty_fp += 1
            if gts and t >= 1:
                pos_correct += 1

    mean_ms = sum(times_ms) / len(times_ms) if times_ms else 0.0
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    img_acc = pos_correct / pos_images if pos_images else 0.0

    out = {
        "backend": detector.backend,
        "algorithm": "SimpleBlobDetector+MSER+geometric_priors",
        "num_images": len(pairs),
        "has_ground_truth": has_any_gt,
        "iou_threshold": iou_thr,
        "confidence_threshold": detector.params.confidence_threshold,
        "timing": {
            "mean_infer_ms": round(mean_ms, 3),
            "approx_fps": round(fps, 3),
        },
    }
    if has_any_gt:
        out["metrics"] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "mean_iou_matched": round(mean_iou, 4),
            "image_accuracy_pos": round(img_acc, 4),
            "pos_images": pos_images,
            "pos_correct_iou": pos_correct,
            "empty_images": empty_images,
            "empty_fp": empty_fp,
        }
    else:
        qual = (frames_with_det / len(pairs)) if pairs else 0.0
        mean_conf = conf_sum / conf_n if conf_n else 0.0
        out["qualitative"] = {
            "frames_with_detection_ratio": round(qual, 4),
            "mean_top_confidence": round(mean_conf, 4),
            "note": "No GT labels found — qualitative only.",
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate agent_05 blob+geometry detector")
    ap.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Image folder (labels as sibling .txt). Overrides --split-dir.",
    )
    ap.add_argument(
        "--split-dir",
        type=str,
        default=str(DEFAULT_SPLIT),
        help="YOLO split root with images/{split} and labels/{split}",
    )
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--conf", type=float, default=0.70)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    if args.dataset:
        pairs = _pairs_from_dir(Path(args.dataset))
        source = args.dataset
    else:
        pairs = _pairs_from_split(Path(args.split_dir), args.split)
        source = f"{args.split_dir} [{args.split}]"
        if not pairs:
            # fallback: vial_photos
            fallback = DEFAULT_CAMERA / "vial_photos"
            pairs = _pairs_from_dir(fallback)
            source = str(fallback)

    if not pairs:
        print("No images found.", file=sys.stderr)
        return 1

    params = BlobParams(confidence_threshold=float(args.conf))
    detector = VialBlobDetector(params)
    max_images = int(args.max_images) if args.max_images > 0 else None
    result = evaluate(pairs, detector, iou_thr=float(args.iou), max_images=max_images)
    result["source"] = source

    print(json.dumps(result, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
