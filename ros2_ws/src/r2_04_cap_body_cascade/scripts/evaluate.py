#!/usr/bin/env python3
"""Evaluate r2_04 cap→body cascade — upright vs sideways metrics.

Sources
-------
- Upright / general: YOLO split ``vial_dataset/split_70_10_20`` (images+labels)
- Sideways: ``sideways_race/synth`` positives (role=positive, kind=*sideways*)
  with optional depth; also reports upright/topdown kinds from the same synth.

Usage (no ROS required):
  cd E:\\Desktop\\Jessica\\Camera\\ros2_ws\\src\\r2_04_cap_body_cascade
  python scripts/evaluate.py
  python scripts/evaluate.py --json-out results/metrics.json
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

from r2_04_cap_body_cascade.cascade_detector import (  # noqa: E402
    CapBodyCascadeDetector,
    CascadeParams,
    list_images,
    load_yolo_label,
)
from r2_04_cap_body_cascade.types import Det, box_iou, xywh_iou  # noqa: E402

DEFAULT_CAMERA = Path(r"E:\Desktop\Jessica\robotics")
DEFAULT_SPLIT = DEFAULT_CAMERA / "vial_dataset" / "split_70_10_20"
DEFAULT_SYNTH = DEFAULT_CAMERA / "sideways_race" / "synth"


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


def _summarize(tp: int, fp: int, fn: int, ious: List[float], times_ms: List[float]) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    mean_ms = sum(times_ms) / len(times_ms) if times_ms else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mean_iou_matched": round(mean_iou, 4),
        "mean_infer_ms": round(mean_ms, 3),
        "n_images": len(times_ms),
    }


def eval_yolo_split(
    pairs: List[Tuple[Path, Path]],
    detector: CapBodyCascadeDetector,
    *,
    iou_thr: float,
    max_images: Optional[int],
) -> dict:
    if max_images is not None:
        pairs = pairs[: max(0, int(max_images))]
    tp = fp = fn = 0
    ious: List[float] = []
    times_ms: List[float] = []
    axis_methods: Dict[str, int] = {}
    for img_path, lbl_path in pairs:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gts = load_yolo_label(lbl_path, w, h) if lbl_path.exists() else []
        t0 = time.perf_counter()
        preds, stats = detector.predict(bgr)
        times_ms.append(stats.inference_ms or (time.perf_counter() - t0) * 1000.0)
        for m, c in stats.axis_methods.items():
            axis_methods[m] = axis_methods.get(m, 0) + c
        t, f, n, ii = match_dets(preds, gts, iou_thr=iou_thr)
        tp += t
        fp += f
        fn += n
        ious.extend(ii)
    out = _summarize(tp, fp, fn, ious, times_ms)
    out["orientation_bucket"] = "upright_dataset"
    out["axis_methods"] = axis_methods
    out["note"] = "vial_dataset split — mostly upright / held (not sideways-labeled)"
    return out


def _load_synth_manifest(synth_dir: Path) -> dict:
    path = synth_dir / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    # Fallback: rebuild from per-sample JSON sidecars.
    samples = []
    for jp in sorted(synth_dir.glob("*.json")):
        if jp.name == "manifest.json":
            continue
        samples.append(json.loads(jp.read_text(encoding="utf-8")))
    return {"samples": samples}


def _control_box_from_note(note: str) -> Optional[List[int]]:
    """Parse 'control GT upright box=[x, y, w, h]' from sideways-race notes."""
    import re

    if not note:
        return None
    m = re.search(r"box\s*=\s*\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]", note)
    if not m:
        return None
    return [int(m.group(i)) for i in range(1, 5)]


def eval_synth_by_orientation(
    synth_dir: Path,
    detector: CapBodyCascadeDetector,
    *,
    iou_thr: float,
    max_per_kind: Optional[int],
) -> dict:
    manifest = _load_synth_manifest(synth_dir)
    samples = manifest.get("samples") or []
    buckets: Dict[str, List[dict]] = {
        "sideways": [],
        "upright": [],
        "topdown": [],
        "negative": [],
    }
    for entry in samples:
        kind = str(entry.get("kind", ""))
        boxes = list(entry.get("boxes") or [])
        # Sideways-race marks upright/topdown as role=negative (must be empty for
        # sideways-only methods). We recover control GT from the note for
        # orientation-agnostic scoring.
        if kind in ("upright", "topdown") and not boxes:
            ctrl = _control_box_from_note(str(entry.get("note", "")))
            if ctrl:
                entry = {**entry, "boxes": [ctrl], "role": "positive"}
                boxes = [ctrl]
        if "sideways" in kind and boxes:
            buckets["sideways"].append(entry)
        elif kind == "upright" and boxes:
            buckets["upright"].append(entry)
        elif kind == "topdown" and boxes:
            buckets["topdown"].append(entry)
        elif kind in ("empty", "hard_negative", "real_wide"):
            buckets["negative"].append(entry)
        elif boxes:
            buckets["sideways"].append(entry)
        else:
            buckets["negative"].append(entry)

    def _run(entries: List[dict], *, positives: bool) -> dict:
        if max_per_kind is not None and positives:
            entries = entries[: max(0, int(max_per_kind))]
        tp = fp = fn = 0
        ious: List[float] = []
        times_ms: List[float] = []
        axis_methods: Dict[str, int] = {}
        n_fp_images = 0
        for entry in entries:
            color_path = synth_dir / entry["color"]
            bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            depth = None
            dpath = entry.get("depth")
            if dpath:
                dp = synth_dir / dpath
                if dp.exists():
                    depth = np.load(dp).astype(np.float32)
            t0 = time.perf_counter()
            preds, stats = detector.predict(bgr, depth_mm=depth)
            times_ms.append(stats.inference_ms or (time.perf_counter() - t0) * 1000.0)
            for m, c in stats.axis_methods.items():
                axis_methods[m] = axis_methods.get(m, 0) + c
            pred_xywh = [d.as_xywh() for d in preds]
            if positives:
                gts = [tuple(b) for b in (entry.get("boxes") or [])]
                if not gts:
                    fn += 1
                    continue
                # Match best IoU per GT (single-vial synth).
                best = 0.0
                for g in gts:
                    for p in pred_xywh:
                        best = max(best, xywh_iou(p, g))
                if best >= iou_thr:
                    tp += 1
                    ious.append(best)
                    # extra preds beyond one count as FP
                    if len(pred_xywh) > 1:
                        fp += len(pred_xywh) - 1
                else:
                    fn += 1
                    fp += len(pred_xywh)
            else:
                if pred_xywh:
                    fp += len(pred_xywh)
                    n_fp_images += 1
        if positives:
            out = _summarize(tp, fp, fn, ious, times_ms)
        else:
            out = {
                "n_images": len(times_ms),
                "fp_count": fp,
                "fp_image_rate": round(n_fp_images / len(times_ms), 4) if times_ms else 0.0,
                "mean_infer_ms": round(sum(times_ms) / len(times_ms), 3) if times_ms else 0.0,
            }
        out["axis_methods"] = axis_methods
        return out

    return {
        "sideways": {**_run(buckets["sideways"], positives=True), "orientation_bucket": "sideways"},
        "upright_synth": {
            **_run(buckets["upright"], positives=True),
            "orientation_bucket": "upright",
        },
        "topdown_synth": {
            **_run(buckets["topdown"], positives=True),
            "orientation_bucket": "topdown",
        },
        "negatives": {
            **_run(buckets["negative"], positives=False),
            "orientation_bucket": "negative",
        },
        "counts": {k: len(v) for k, v in buckets.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate r2_04 cap→body cascade")
    ap.add_argument("--split-dir", type=str, default=str(DEFAULT_SPLIT))
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--synth", type=str, default=str(DEFAULT_SYNTH))
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--no-yolo", action="store_true", help="Disable YOLO RGB proposals")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-images", type=int, default=0, help="0 = all (YOLO split)")
    ap.add_argument("--max-per-kind", type=int, default=0, help="0 = all (synth buckets)")
    ap.add_argument("--json-out", type=str, default="")
    ap.add_argument("--skip-dataset", action="store_true")
    ap.add_argument("--skip-synth", action="store_true")
    args = ap.parse_args()

    params = CascadeParams(
        confidence_threshold=float(args.conf),
        use_yolo_rgb=not bool(args.no_yolo),
    )
    detector = CapBodyCascadeDetector(params)
    iou_thr = float(args.iou)
    max_images = int(args.max_images) if args.max_images > 0 else None
    max_per_kind = int(args.max_per_kind) if args.max_per_kind > 0 else None

    result: dict = {
        "agent": "r2_04_cap_body_cascade",
        "algorithm": "cap→body geometric cascade (PCA / ellipse / dual-blob link)",
        "axis_estimation": {
            "primary": "dual-blob link (depth/sideways) + PCA / cardinal probes along major axis",
            "rgb_rescue": "YOLO proposals fused via NMS when depth absent (upright stills)",
            "fallbacks": [
                "PCA perpendicular to thin end-cap",
                "cardinal probes (up/down/left/right)",
                "topdown padded-cap",
            ],
            "expansion": "geometric prior along major axis both ways; body-mass side bias (NOT down-only)",
        },
        "iou_threshold": iou_thr,
        "confidence_threshold": params.confidence_threshold,
        "backend": detector.backend,
    }

    if not args.skip_dataset:
        pairs = _pairs_from_split(Path(args.split_dir), args.split)
        if pairs:
            result["upright_dataset"] = eval_yolo_split(
                pairs, detector, iou_thr=iou_thr, max_images=max_images
            )
            result["upright_dataset"]["source"] = f"{args.split_dir} [{args.split}]"
        else:
            result["upright_dataset"] = {"error": "no images", "source": args.split_dir}

    if not args.skip_synth:
        synth_dir = Path(args.synth)
        if synth_dir.exists():
            result["synth_by_orientation"] = eval_synth_by_orientation(
                synth_dir, detector, iou_thr=iou_thr, max_per_kind=max_per_kind
            )
            result["synth_by_orientation"]["source"] = str(synth_dir.resolve())
        else:
            result["synth_by_orientation"] = {"error": f"missing {synth_dir}"}

    # Compact comparison table for the round-2 gate.
    cmp_rows = {}
    if "upright_dataset" in result and "recall" in result.get("upright_dataset", {}):
        u = result["upright_dataset"]
        cmp_rows["upright_dataset"] = {
            "recall": u["recall"],
            "precision": u["precision"],
            "mean_iou": u["mean_iou_matched"],
        }
    syn = result.get("synth_by_orientation") or {}
    for key in ("sideways", "upright_synth", "topdown_synth"):
        if key in syn and "recall" in syn[key]:
            cmp_rows[key] = {
                "recall": syn[key]["recall"],
                "precision": syn[key]["precision"],
                "mean_iou": syn[key]["mean_iou_matched"],
            }
    result["comparison"] = cmp_rows
    side_ok = cmp_rows.get("sideways", {}).get("recall", 0) >= 0.70
    up_ds = cmp_rows.get("upright_dataset", {}).get("recall", 0)
    up_ok = up_ds >= 0.50
    up_target = up_ds >= 0.70
    result["orientation_agnostic_gate"] = {
        "sideways_recall_ge_0_70": side_ok,
        "upright_dataset_recall_ge_0_50": up_ok,
        "upright_dataset_recall_ge_0_70": up_target,
        "pass": bool(side_ok and up_ok),
        "fired_if_upright_dataset_fails": not up_ok,
    }
    result["yolo_rgb_enabled"] = bool(params.use_yolo_rgb)

    print(json.dumps(result, indent=2))
    if args.json_out:
        outp = Path(args.json_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {outp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
