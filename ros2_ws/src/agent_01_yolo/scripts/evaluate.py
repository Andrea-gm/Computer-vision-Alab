#!/usr/bin/env python3
"""Evaluate YOLO vial detector on the labeled dataset.

If YOLO GT labels exist (*.txt next to images, or split_70_10_20), reports
precision / recall / mean IoU at IoU≥0.5 (same gate as Camera plan.md).

If no GT: reports qualitative score (mean conf of top det, frames with det)
plus timing.

Usage (no ROS required):
  cd E:\\Desktop\\Jessica\\Camera\\ros2_ws\\src\\agent_01_yolo
  python scripts/evaluate.py
  python scripts/evaluate.py --split test --conf 0.25
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

from agent_01_yolo.classical_fallback import Det  # noqa: E402
from agent_01_yolo.detector import (  # noqa: E402
    VialYoloDetector,
    box_iou,
    default_camera_root,
    list_images,
    load_yolo_label,
    resolve_model_path,
)


def _pairs_from_raw(raw_dir: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for img in list_images(raw_dir):
        pairs.append((img, img.with_suffix(".txt")))
    return pairs


def _pairs_from_split(split_dir: Path, split: str) -> List[Tuple[Path, Path]]:
    """split in {train, val, validate, test}."""
    name = "val" if split in ("val", "validate") else split
    img_dir = split_dir / "images" / name
    lbl_dir = split_dir / "labels" / name
    if not img_dir.exists():
        # Some rebuilds only symlink images; fall back to raw via manifest.
        return []
    pairs = []
    for img in list_images(img_dir):
        lbl = lbl_dir / f"{img.stem}.txt"
        pairs.append((img, lbl))
    return pairs


def match_dets(
    preds: Sequence[Det],
    gts: Sequence[Det],
    *,
    iou_thr: float,
) -> Tuple[int, int, int, List[float]]:
    """Greedy IoU match. Returns tp, fp, fn, matched_ious."""
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
    detector: VialYoloDetector,
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
    pos_correct = 0  # image-level: ≥1 TP at IoU thr (plan.md test accuracy)

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
        # prefer detector's reported ms when available
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
        "model_path": str(detector.model_path or ""),
        "num_images": len(pairs),
        "has_ground_truth": has_any_gt,
        "iou_threshold": iou_thr,
        "confidence_threshold": detector.confidence_threshold,
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
    camera = default_camera_root()
    ap = argparse.ArgumentParser(description="Evaluate agent_01_yolo on vial dataset")
    ap.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Folder of images (+ optional YOLO .txt). Default: split or raw.",
    )
    ap.add_argument(
        "--split-dir",
        type=Path,
        default=camera / "vial_dataset" / "split_70_10_20",
    )
    ap.add_argument(
        "--split",
        choices=("train", "val", "validate", "test", "raw"),
        default="test",
    )
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5, help="Match IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "eval_results.json",
    )
    ap.add_argument("--no-fallback", action="store_true")
    args = ap.parse_args()

    if args.dataset is not None:
        pairs = _pairs_from_raw(args.dataset)
        source = str(args.dataset)
    elif args.split == "raw":
        pairs = _pairs_from_raw(camera / "vial_dataset" / "raw")
        source = str(camera / "vial_dataset" / "raw")
    else:
        pairs = _pairs_from_split(args.split_dir, args.split)
        source = f"{args.split_dir} [{args.split}]"
        if not pairs:
            # Fallback: use raw listing filtered by manifest if present.
            manifest = args.split_dir / "manifest.json"
            raw = camera / "vial_dataset" / "raw"
            if manifest.exists():
                data = json.loads(manifest.read_text(encoding="utf-8"))
                key = "validate" if args.split in ("val", "validate") else args.split
                names = data.get(key, [])
                pairs = []
                for name in names:
                    img = raw / name
                    if img.exists():
                        pairs.append((img, img.with_suffix(".txt")))
                source = f"manifest:{key} via raw ({len(pairs)} images)"

    if not pairs:
        print(f"No images found for source={source}", file=sys.stderr)
        return 1

    model = resolve_model_path(args.model, camera)
    print(f"source={source}")
    print(f"images={len(pairs)} model={model}")

    detector = VialYoloDetector(
        model,
        camera_root=camera,
        confidence_threshold=args.conf,
        imgsz=args.imgsz,
        allow_classical_fallback=not args.no_fallback,
    )
    print(f"backend={detector.backend}")

    result = evaluate(pairs, detector, iou_thr=args.iou, max_images=args.max_images)
    result["source"] = source
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
