#!/usr/bin/env python3
"""Offline eval with SEPARATE upright vs sideways metrics.

Subset definition (documented, reproducible)
--------------------------------------------
1. If data/multi_orient/orientation_manifest.json exists, use its `orient` tags.
2. Else classify each image:
   - sideways: filename contains 'sideways' OR stem ends with '_r90'/'_r270'
               OR GT median aspect (h/w) < 0.95
   - upright:  filename contains 'upright' OR GT median aspect >= 0.95
   - negative: empty GT label
   - other:    everything else (reported but not in upright/sideways)

Default eval set = multi_orient val+test + sideways_race synth positives/negatives
so both orientations are scored even before prepare_multiorient_dataset runs.

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --model models/best.pt --conf 0.25
  python scripts/evaluate.py --no-tta
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2_01_yolo_multiorient.detector import (  # noqa: E402
    VialYoloDetector,
    default_camera_root,
    list_images,
    load_yolo_label,
    resolve_model_path,
)
from r2_01_yolo_multiorient.types import Det, box_iou  # noqa: E402


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
        best_i, best_iou = -1, 0.0
        for gi, g in enumerate(gts):
            if gi in used_gt:
                continue
            iou = box_iou(p, g)
            if iou > best_iou:
                best_iou, best_i = iou, gi
        if best_i >= 0 and best_iou >= iou_thr:
            used_gt.add(best_i)
            tps += 1
            ious.append(best_iou)
    return tps, len(preds) - tps, len(gts) - tps, ious


def classify_orient(stem: str, gts: Sequence[Det], manifest_orient: Optional[str]) -> str:
    if manifest_orient in ("upright", "sideways", "negative", "other", "empty"):
        if manifest_orient == "empty":
            return "negative"
        return manifest_orient
    s = stem.lower()
    if "sideways" in s or s.endswith("_r90") or s.endswith("_r270"):
        return "sideways"
    if "upright" in s:
        return "upright"
    if not gts:
        return "negative"
    aspects = [g.aspect_hw for g in gts if g.width > 1e-6]
    if not aspects:
        return "negative"
    med = float(np.median(aspects))
    if med < 0.95:
        return "sideways"
    return "upright"


def load_manifest(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {s["stem"]: s.get("orient", "other") for s in data.get("samples", [])}


def collect_pairs(
    camera: Path,
    multi_dir: Path,
    *,
    include_synth: bool,
    max_per_subset: Optional[int],
) -> List[Tuple[Path, Path, str]]:
    """Return list of (image, label, orient)."""
    man = load_manifest(multi_dir / "orientation_manifest.json")
    pairs: List[Tuple[Path, Path, str]] = []
    seen = set()

    def add(img: Path, lbl: Path, orient_hint: Optional[str] = None) -> None:
        # Dedupe by stem so multi_orient copies + sideways_race originals do not double-count.
        key = img.stem.lower()
        if key in seen or not img.exists():
            return
        seen.add(key)
        bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if bgr is None:
            return
        h, w = bgr.shape[:2]
        gts = load_yolo_label(lbl, w, h) if lbl.exists() else []
        orient = classify_orient(img.stem, gts, orient_hint or man.get(img.stem))
        pairs.append((img, lbl, orient))

    # multi_orient val + test if present
    if multi_dir.exists():
        for split in ("val", "test"):
            img_dir = multi_dir / "images" / split
            lbl_dir = multi_dir / "labels" / split
            if not img_dir.exists():
                continue
            for img in list_images(img_dir):
                add(img, lbl_dir / f"{img.stem}.txt", man.get(img.stem))

    # Always include sideways_race synth for honest sideways numbers
    if include_synth:
        synth = camera / "sideways_race" / "synth"
        man_path = synth / "manifest.json"
        if man_path.exists():
            data = json.loads(man_path.read_text(encoding="utf-8"))
            for entry in data.get("samples", []):
                kind = entry.get("kind", "")
                role = entry.get("role", "")
                color = synth / entry["color"]
                # write ephemeral YOLO labels next to a cache dir
                cache = ROOT / "data" / "eval_synth_labels"
                cache.mkdir(parents=True, exist_ok=True)
                lbl = cache / f"{entry['id']}.txt"
                iw = int(entry.get("image_width", 640))
                ih = int(entry.get("image_height", 480))
                lines = []
                for b in entry.get("boxes") or []:
                    x, y, bw, bh = map(float, b)
                    cx = (x + bw / 2.0) / iw
                    cy = (y + bh / 2.0) / ih
                    lines.append(f"0 {cx:.6f} {cy:.6f} {bw / iw:.6f} {bh / ih:.6f}")
                lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                if role == "positive" and "sideways" in kind:
                    orient = "sideways"
                elif role == "positive" and kind == "upright":
                    orient = "upright"
                elif role == "negative" or kind in ("empty", "hard_negative"):
                    orient = "negative"
                else:
                    orient = "other"
                add(color, lbl, orient)

    # Fallback: vial_dataset test split if nothing else
    if not pairs:
        split = camera / "vial_dataset" / "split_70_10_20"
        for img in list_images(split / "images" / "test"):
            add(img, split / "labels" / "test" / f"{img.stem}.txt")

    if max_per_subset is not None:
        capped: List[Tuple[Path, Path, str]] = []
        counts: Dict[str, int] = defaultdict(int)
        for p in pairs:
            if counts[p[2]] >= max_per_subset:
                continue
            counts[p[2]] += 1
            capped.append(p)
        pairs = capped
    return pairs


def summarize(tp, fp, fn, ious, pos_images, pos_correct, empty_images, empty_fp, n_images, times_ms):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    img_acc = pos_correct / pos_images if pos_images else 0.0
    mean_ms = sum(times_ms) / len(times_ms) if times_ms else 0.0
    return {
        "num_images": n_images,
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
        "mean_infer_ms": round(mean_ms, 3),
    }


def evaluate_by_orient(
    pairs: List[Tuple[Path, Path, str]],
    detector: VialYoloDetector,
    *,
    iou_thr: float,
) -> dict:
    buckets = {
        "upright": {"tp": 0, "fp": 0, "fn": 0, "ious": [], "pos": 0, "pos_ok": 0, "empty": 0, "empty_fp": 0, "n": 0, "ms": []},
        "sideways": {"tp": 0, "fp": 0, "fn": 0, "ious": [], "pos": 0, "pos_ok": 0, "empty": 0, "empty_fp": 0, "n": 0, "ms": []},
        "negative": {"tp": 0, "fp": 0, "fn": 0, "ious": [], "pos": 0, "pos_ok": 0, "empty": 0, "empty_fp": 0, "n": 0, "ms": []},
        "other": {"tp": 0, "fp": 0, "fn": 0, "ious": [], "pos": 0, "pos_ok": 0, "empty": 0, "empty_fp": 0, "n": 0, "ms": []},
        "all": {"tp": 0, "fp": 0, "fn": 0, "ious": [], "pos": 0, "pos_ok": 0, "empty": 0, "empty_fp": 0, "n": 0, "ms": []},
    }

    for img_path, lbl_path, orient in pairs:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gts = load_yolo_label(lbl_path, w, h) if lbl_path.exists() else []
        t0 = time.perf_counter()
        preds, stats = detector.predict(bgr)
        ms = stats.inference_ms if stats.inference_ms > 0 else (time.perf_counter() - t0) * 1000.0
        tp, fp, fn, ious = match_dets(preds, gts, iou_thr=iou_thr)
        is_empty = lbl_path.exists() and (not gts)

        for key in (orient if orient in buckets else "other", "all"):
            b = buckets[key]
            b["n"] += 1
            b["tp"] += tp
            b["fp"] += fp
            b["fn"] += fn
            b["ious"].extend(ious)
            b["ms"].append(ms)
            if is_empty:
                b["empty"] += 1
                if preds:
                    b["empty_fp"] += 1
            elif gts:
                b["pos"] += 1
                if tp >= 1:
                    b["pos_ok"] += 1

    out = {}
    for name, b in buckets.items():
        out[name] = summarize(
            b["tp"], b["fp"], b["fn"], b["ious"], b["pos"], b["pos_ok"],
            b["empty"], b["empty_fp"], b["n"], b["ms"],
        )
    return out


def main() -> int:
    camera = default_camera_root()
    multi_dir = ROOT / "data" / "multi_orient"
    ap = argparse.ArgumentParser(description="Eval multi-orient YOLO (upright vs sideways)")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-synth", action="store_true")
    ap.add_argument("--max-per-subset", type=int, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "eval_results.json")
    args = ap.parse_args()

    pairs = collect_pairs(
        camera,
        multi_dir,
        include_synth=not args.no_synth,
        max_per_subset=args.max_per_subset,
    )
    if not pairs:
        print("No eval images found", file=sys.stderr)
        return 1

    model = resolve_model_path(args.model, camera)
    print(f"model={model}")
    print(f"pairs={len(pairs)} tta_rotate={not args.no_tta}")
    counts = defaultdict(int)
    for _, _, o in pairs:
        counts[o] += 1
    print("subset counts:", dict(counts))

    detector = VialYoloDetector(
        model,
        camera_root=camera,
        confidence_threshold=args.conf,
        imgsz=args.imgsz,
        tta_rotate=not args.no_tta,
    )
    print(f"backend={detector.backend}")

    metrics = evaluate_by_orient(pairs, detector, iou_thr=args.iou)
    result = {
        "backend": detector.backend,
        "model_path": str(detector.model_path or ""),
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "tta_rotate": not args.no_tta,
        "subset_definition": {
            "upright": "manifest upright OR filename upright_* OR GT median h/w >= 0.95",
            "sideways": "manifest sideways OR filename *sideways* OR *_r90/*_r270 OR GT median h/w < 0.95",
            "negative": "empty GT / empty|hard_negative synth",
        },
        "subset_counts": dict(counts),
        "metrics_by_orient": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== UPRIGHT ===")
    print(json.dumps(metrics["upright"], indent=2))
    print("\n=== SIDEWAYS ===")
    print(json.dumps(metrics["sideways"], indent=2))
    print("\n=== ALL ===")
    print(json.dumps(metrics["all"], indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
