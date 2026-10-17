#!/usr/bin/env python3
"""Evaluate r2_05 weights separately on upright and sideways holdouts.

  python scripts/evaluate.py
  python scripts/evaluate.py --model models/weights/best.pt --conf 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from r2_05_aug_retrain.detector import (  # noqa: E402
    Det,
    VialYoloDetector,
    box_iou,
    list_images,
    load_yolo_label,
    package_weights,
)


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


def pairs_for_holdout(root: Path, name: str) -> List[Tuple[Path, Path]]:
    img_dir = root / "images" / name
    lbl_dir = root / "labels" / name
    pairs = []
    for img in list_images(img_dir):
        pairs.append((img, lbl_dir / f"{img.stem}.txt"))
    return pairs


def evaluate_split(
    pairs: List[Tuple[Path, Path]],
    detector: VialYoloDetector,
    *,
    iou_thr: float,
    max_images: Optional[int],
) -> dict:
    if max_images is not None:
        pairs = pairs[: max(0, int(max_images))]
    tp = fp = fn = 0
    all_ious: List[float] = []
    times_ms: List[float] = []
    pos_images = empty_images = pos_correct = empty_fp = 0

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
        times_ms.append(
            stats.inference_ms if stats.inference_ms > 0 else (time.perf_counter() - t0) * 1000.0
        )

        t, f, n, ious = match_dets(preds, gts, iou_thr=iou_thr)
        tp += t
        fp += f
        fn += n
        all_ious.extend(ious)
        if gts and t > 0:
            pos_correct += 1
        if not gts and f > 0:
            empty_fp += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    return {
        "images": len(pairs),
        "pos_images": pos_images,
        "empty_images": empty_images,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mean_iou": round(mean_iou, 4),
        "image_recall": round(pos_correct / pos_images, 4) if pos_images else 0.0,
        "empty_fp": empty_fp,
        "mean_infer_ms": round(sum(times_ms) / len(times_ms), 2) if times_ms else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=PKG / "data" / "aug_split",
    )
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=PKG / "results" / "eval_upright_sideways.json",
    )
    args = ap.parse_args()

    model_path = args.model or package_weights()
    if not Path(model_path).exists():
        print(f"Missing weights: {model_path}. Run scripts/train.py first.", file=sys.stderr)
        return 1
    if not (args.data_root / "images" / "holdout_upright").exists():
        print(
            f"Missing holdouts under {args.data_root}. Run scripts/build_aug_dataset.py.",
            file=sys.stderr,
        )
        return 1

    detector = VialYoloDetector(
        model_path,
        confidence_threshold=args.conf,
        imgsz=args.imgsz,
    )

    upright = evaluate_split(
        pairs_for_holdout(args.data_root, "holdout_upright"),
        detector,
        iou_thr=args.iou,
        max_images=args.max_images,
    )
    sideways = evaluate_split(
        pairs_for_holdout(args.data_root, "holdout_sideways"),
        detector,
        iou_thr=args.iou,
        max_images=args.max_images,
    )

    gap = upright["recall"] - sideways["recall"]
    report = {
        "model": str(Path(model_path).resolve()),
        "conf": args.conf,
        "iou_thr": args.iou,
        "imgsz": args.imgsz,
        "holdout_upright": upright,
        "holdout_sideways": sideways,
        "recall_gap_upright_minus_sideways": round(gap, 4),
        "sideways_recall_ok": sideways["recall"] >= max(0.0, upright["recall"] - 0.25),
        "note": (
            "Target: sideways recall must not collapse vs upright "
            "(gap ideally < 0.25 absolute)."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== UPRIGHT HOLDOUT ===")
    print(json.dumps(upright, indent=2))
    print("=== SIDEWAYS HOLDOUT ===")
    print(json.dumps(sideways, indent=2))
    print(
        f"recall gap (upright - sideways) = {gap:.4f} "
        f"| sideways_ok={report['sideways_recall_ok']}"
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
