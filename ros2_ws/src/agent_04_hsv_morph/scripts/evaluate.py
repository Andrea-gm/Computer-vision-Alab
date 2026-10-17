#!/usr/bin/env python3
"""Evaluate HSV/morph vial detector on the labeled dataset.

Reports precision / recall / mean IoU at IoU≥0.5 when YOLO-format GT exists.

Usage (no ROS required):
  cd E:\\Desktop\\Jessica\\Camera\\ros2_ws\\src\\agent_04_hsv_morph
  python scripts/evaluate.py
  python scripts/evaluate.py --split test
  python scripts/evaluate.py --dataset E:\\Desktop\\Jessica\\Camera\\vial_dataset\\raw
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

from agent_04_hsv_morph.detector import (  # noqa: E402
    Det,
    HsvMorphParams,
    VialHsvMorphDetector,
    box_iou,
    default_camera_root,
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
    return tps, len(preds) - tps, len(gts) - tps, ious


def evaluate(
    pairs: List[Tuple[Path, Path]],
    detector: VialHsvMorphDetector,
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
    pos_images = 0
    empty_images = 0
    pos_correct = 0

    for img_path, lbl_path in pairs:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gts = load_yolo_label(lbl_path, w, h) if lbl_path.exists() else []
        if lbl_path.exists() and not gts:
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
            if gts and t > 0:
                pos_correct += 1

    n_frames = max(1, len(times_ms))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    mean_iou = float(sum(all_ious) / len(all_ious)) if all_ious else 0.0
    result = {
        "backend": detector.backend,
        "frames": len(times_ms),
        "has_gt": has_any_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "mean_iou_matched": mean_iou,
        "iou_threshold": iou_thr,
        "image_level_pos_recall": pos_correct / max(1, pos_images),
        "pos_images": pos_images,
        "empty_images": empty_images,
        "frames_with_det": frames_with_det,
        "mean_top_conf": (conf_sum / conf_n) if conf_n else 0.0,
        "mean_infer_ms": float(sum(times_ms) / n_frames),
        "params": detector.params.__dict__,
    }
    return result


def main() -> int:
    camera = default_camera_root()
    default_split = camera / "vial_dataset" / "split_70_10_20"
    ap = argparse.ArgumentParser(description="Evaluate agent_04 HSV/morph vial detector")
    ap.add_argument("--dataset", type=str, default="", help="Raw image folder (YOLO txt beside)")
    ap.add_argument("--split-dir", type=str, default=str(default_split))
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    ap.add_argument("--s-max", type=int, default=None, help="Override hsv_s_max")
    ap.add_argument("--v-min", type=int, default=None, help="Override hsv_v_min")
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    params = HsvMorphParams()
    if args.s_max is not None:
        params.hsv_s_max = int(args.s_max)
    if args.v_min is not None:
        params.hsv_v_min = int(args.v_min)
    if args.min_score is not None:
        params.min_score = float(args.min_score)

    detector = VialHsvMorphDetector(params)

    if args.dataset:
        pairs = _pairs_from_raw(Path(args.dataset))
        source = args.dataset
    else:
        pairs = _pairs_from_split(Path(args.split_dir), args.split)
        source = f"{args.split_dir} [{args.split}]"
        if not pairs:
            raw = camera / "vial_dataset" / "raw"
            pairs = _pairs_from_raw(raw)
            source = str(raw)

    if not pairs:
        print("No images found.", file=sys.stderr)
        return 1

    max_images = args.max_images if args.max_images > 0 else None
    result = evaluate(pairs, detector, iou_thr=args.iou, max_images=max_images)
    result["source"] = source

    print(json.dumps(result, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
