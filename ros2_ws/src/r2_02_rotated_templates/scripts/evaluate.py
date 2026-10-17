#!/usr/bin/env python3
"""Evaluate rotated-template detector with upright vs sideways metrics.

Usage (no ROS required):
  cd E:\\Desktop\\Jessica\\Camera\\ros2_ws\\src\\r2_02_rotated_templates
  python scripts/extract_templates.py
  python scripts/evaluate.py
  python scripts/evaluate.py --split test --conf 0.48
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2_02_rotated_templates.detector import (  # noqa: E402
    RotatedTemplateDetector,
    list_images,
    load_yolo_label,
)
from r2_02_rotated_templates.template_bank import default_camera_root  # noqa: E402
from r2_02_rotated_templates.types import Det, box_iou  # noqa: E402


def _pairs_from_raw(raw_dir: Path) -> List[Tuple[Path, Path]]:
    return [(img, img.with_suffix(".txt")) for img in list_images(raw_dir)]


def _pairs_from_split(split_dir: Path, split: str) -> List[Tuple[Path, Path]]:
    name = "val" if split in ("val", "validate") else split
    img_dir = split_dir / "images" / name
    lbl_dir = split_dir / "labels" / name
    if not img_dir.exists():
        return []
    return [(img, lbl_dir / f"{img.stem}.txt") for img in list_images(img_dir)]


def _template_source_names(templates_dir: Path) -> set:
    meta_path = templates_dir / "meta.json"
    if not meta_path.exists():
        return set()
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return set()
    return {str(e.get("source_image", "")) for e in data.get("templates", []) if e.get("source_image")}


def _sideways_pairs(
    synth_dir: Path,
    *,
    exclude_sources: Optional[set] = None,
) -> List[Tuple[Path, object, str]]:
    """Return (image_path, gt_dets_or_empty, kind)."""
    out = []
    exclude_sources = exclude_sources or set()
    if not synth_dir.exists():
        return out
    for meta_path in sorted(synth_dir.glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        kind = str(meta.get("kind", meta_path.stem))
        color_name = meta.get("color")
        if not color_name:
            continue
        if color_name in exclude_sources:
            continue  # hold out template-bank sources
        img_path = synth_dir / color_name
        if not img_path.exists():
            continue
        gts: List[Det] = []
        role = str(meta.get("role", ""))
        is_pos = role == "positive" or kind.startswith("sideways") or kind.startswith(
            "hard_sideways"
        )
        # Treat upright/topdown/empty/hard_negative/real_wide as negatives for
        # the sideways metric (must not silently fire on upright-only failure).
        if is_pos:
            for box in meta.get("boxes") or []:
                x, y, bw, bh = [float(v) for v in box[:4]]
                gts.append(
                    Det(
                        x,
                        y,
                        x + bw,
                        y + bh,
                        conf=1.0,
                        source="gt",
                        orientation="sideways",
                    )
                )
        out.append((img_path, gts, kind))
    # Interleave kinds so --max-sideways does not sample only empties/hard_*.
    by_kind: Dict[str, List[Tuple[Path, object, str]]] = {}
    for item in out:
        by_kind.setdefault(item[2], []).append(item)
    interleaved: List[Tuple[Path, object, str]] = []
    keys = sorted(by_kind.keys())
    idx = 0
    while len(interleaved) < len(out):
        progressed = False
        for k in keys:
            bucket = by_kind[k]
            if idx < len(bucket):
                interleaved.append(bucket[idx])
                progressed = True
        if not progressed:
            break
        idx += 1
    return interleaved


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
    return tps, len(preds) - tps, len(gts) - tps, ious


def _empty_bucket() -> dict:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "pos_images": 0,
        "pos_correct": 0,
        "empty_images": 0,
        "empty_fp": 0,
        "ious": [],
        "times_ms": [],
    }


def _finalize(bucket: dict, *, iou_thr: float) -> dict:
    tp, fp, fn = bucket["tp"], bucket["fp"], bucket["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    ious = bucket["ious"]
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    pos = bucket["pos_images"]
    img_acc = bucket["pos_correct"] / pos if pos else 0.0
    times = bucket["times_ms"]
    mean_ms = sum(times) / len(times) if times else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mean_iou_matched": round(mean_iou, 4),
        "image_accuracy_pos": round(img_acc, 4),
        "pos_images": pos,
        "pos_correct_iou": bucket["pos_correct"],
        "empty_images": bucket["empty_images"],
        "empty_fp": bucket["empty_fp"],
        "iou_threshold": iou_thr,
        "mean_infer_ms": round(mean_ms, 3),
        "approx_fps": round(1000.0 / mean_ms, 3) if mean_ms > 0 else 0.0,
    }


def evaluate_upright(
    pairs: List[Tuple[Path, Path]],
    detector: RotatedTemplateDetector,
    *,
    iou_thr: float,
    max_images: Optional[int],
) -> dict:
    if max_images is not None:
        pairs = pairs[: max(0, int(max_images))]
    bucket = _empty_bucket()
    for img_path, lbl_path in pairs:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gts = load_yolo_label(lbl_path, w, h) if lbl_path.exists() else []
        is_empty = lbl_path.exists() and (not gts)
        if is_empty:
            bucket["empty_images"] += 1
        elif gts:
            bucket["pos_images"] += 1
        t0 = time.perf_counter()
        preds, stats = detector.predict(bgr)
        ms = stats.inference_ms if stats.inference_ms > 0 else (time.perf_counter() - t0) * 1000
        bucket["times_ms"].append(ms)
        t, f, n, ious = match_dets(preds, gts, iou_thr=iou_thr)
        bucket["tp"] += t
        bucket["fp"] += f
        bucket["fn"] += n
        bucket["ious"].extend(ious)
        if is_empty and preds:
            bucket["empty_fp"] += 1
        if gts and t >= 1:
            bucket["pos_correct"] += 1
    return _finalize(bucket, iou_thr=iou_thr)


def evaluate_sideways(
    items: List[Tuple[Path, object, str]],
    detector: RotatedTemplateDetector,
    *,
    iou_thr: float,
    max_images: Optional[int],
) -> dict:
    if max_images is not None:
        items = items[: max(0, int(max_images))]
    bucket = _empty_bucket()
    by_kind: Dict[str, dict] = {}
    silent_sideways_miss = 0
    for img_path, gts, kind in items:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        gts = list(gts)
        is_pos = bool(gts)
        if is_pos:
            bucket["pos_images"] += 1
        else:
            bucket["empty_images"] += 1
        t0 = time.perf_counter()
        preds, stats = detector.predict(bgr)
        ms = stats.inference_ms if stats.inference_ms > 0 else (time.perf_counter() - t0) * 1000
        bucket["times_ms"].append(ms)
        t, f, n, ious = match_dets(preds, gts, iou_thr=iou_thr)
        bucket["tp"] += t
        bucket["fp"] += f
        bucket["fn"] += n
        bucket["ious"].extend(ious)
        if not is_pos and preds:
            bucket["empty_fp"] += 1
        if is_pos and t >= 1:
            bucket["pos_correct"] += 1
        if is_pos and t == 0:
            # Fire-able offense: sideways GT present, zero TP (silent fail).
            silent_sideways_miss += 1
        kb = by_kind.setdefault(kind, {"pos": 0, "hit": 0, "fp": 0})
        if is_pos:
            kb["pos"] += 1
            if t >= 1:
                kb["hit"] += 1
        elif preds:
            kb["fp"] += 1

    out = _finalize(bucket, iou_thr=iou_thr)
    out["silent_sideways_misses"] = silent_sideways_miss
    out["by_kind"] = {
        k: {
            "pos": v["pos"],
            "hit": v["hit"],
            "recall": round(v["hit"] / v["pos"], 4) if v["pos"] else None,
            "fp": v["fp"],
        }
        for k, v in sorted(by_kind.items())
    }
    return out


def main() -> int:
    camera = default_camera_root()
    ap = argparse.ArgumentParser(description="Evaluate r2_02_rotated_templates")
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
    ap.add_argument(
        "--sideways-dir",
        type=Path,
        default=camera / "sideways_race" / "synth",
    )
    ap.add_argument("--templates-dir", type=Path, default=None)
    ap.add_argument("--conf", type=float, default=0.48)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--max-sideways", type=int, default=None)
    ap.add_argument("--skip-upright", action="store_true")
    ap.add_argument("--skip-sideways", action="store_true")
    ap.add_argument(
        "--angles",
        type=str,
        default="0,45,90,135,180,225,270,315",
        help="Comma-separated template rotation angles (degrees)",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "eval_results.json")
    args = ap.parse_args()

    angles = [float(x.strip()) for x in args.angles.split(",") if x.strip()]

    if args.split == "raw":
        upright_pairs = _pairs_from_raw(camera / "vial_dataset" / "raw")
        upright_source = str(camera / "vial_dataset" / "raw")
    else:
        upright_pairs = _pairs_from_split(args.split_dir, args.split)
        upright_source = f"{args.split_dir} [{args.split}]"

    detector = RotatedTemplateDetector(
        args.templates_dir,
        camera_root=camera,
        template_score_threshold=args.conf,
        angles_deg=angles,
    )
    n_up = sum(1 for t in detector.templates if t.pose == "upright")
    n_side = sum(1 for t in detector.templates if t.pose == "sideways")
    print(
        f"templates={len(detector.templates)} (upright={n_up} sideways={n_side}) "
        f"views={len(detector.rotated_views)} dir={detector.templates_dir}"
    )
    print(f"angles={detector.angles_deg} conf={args.conf}")

    result = {
        "backend": detector.backend,
        "templates_dir": str(detector.templates_dir),
        "n_templates": len(detector.templates),
        "n_upright_templates": n_up,
        "n_sideways_templates": n_side,
        "n_rotated_views": len(detector.rotated_views),
        "angles_deg": list(detector.angles_deg),
        "template_score_threshold": args.conf,
        "orientation_strategy": (
            "multi-scale NCC × multi-angle (0/45/90/…/315) on gray+edge templates; "
            "bank built from upright train GT crops + sideways_race synth GT crops; "
            "orientation-aware white-cap strips; aspect 0.35–3.0"
        ),
    }

    if not args.skip_upright:
        if not upright_pairs:
            print(f"WARNING: no upright images for {upright_source}", file=sys.stderr)
            result["upright"] = {"error": "no images"}
        else:
            print(f"evaluating upright: {upright_source} n={len(upright_pairs)}")
            result["upright_source"] = upright_source
            result["upright"] = evaluate_upright(
                upright_pairs,
                detector,
                iou_thr=args.iou,
                max_images=args.max_images,
            )
            u = result["upright"]
            print(
                f"  UPRIGHT  recall={u['recall']:.3f} precision={u['precision']:.3f} "
                f"img_acc={u['image_accuracy_pos']:.3f} "
                f"tp/fp/fn={u['tp']}/{u['fp']}/{u['fn']} "
                f"empty_fp={u['empty_fp']} ms={u['mean_infer_ms']:.1f}"
            )

    if not args.skip_sideways:
        exclude = _template_source_names(Path(detector.templates_dir))
        side_items = _sideways_pairs(args.sideways_dir, exclude_sources=exclude)
        if not side_items:
            print(f"WARNING: no sideways synth at {args.sideways_dir}", file=sys.stderr)
            result["sideways"] = {"error": "no images"}
        else:
            print(f"evaluating sideways: {args.sideways_dir} n={len(side_items)}")
            result["sideways_source"] = str(args.sideways_dir)
            result["sideways"] = evaluate_sideways(
                side_items,
                detector,
                iou_thr=args.iou,
                max_images=args.max_sideways if args.max_sideways is not None else args.max_images,
            )
            s = result["sideways"]
            print(
                f"  SIDEWAYS recall={s['recall']:.3f} precision={s['precision']:.3f} "
                f"img_acc={s['image_accuracy_pos']:.3f} "
                f"tp/fp/fn={s['tp']}/{s['fp']}/{s['fn']} "
                f"empty_fp={s['empty_fp']} silent_miss={s['silent_sideways_misses']} "
                f"ms={s['mean_infer_ms']:.1f}"
            )

    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.out}")

    # Hard fail signal for the race: sideways must not be silent-zero.
    side = result.get("sideways") or {}
    if isinstance(side, dict) and side.get("pos_images", 0) > 0:
        if side.get("recall", 0.0) <= 0.0:
            print("FAIL: sideways recall is 0 — upright-only / silent sideways", file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
