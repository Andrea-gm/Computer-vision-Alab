#!/usr/bin/env python3
"""Offline eval with per-orientation metrics (upright / sideways / other).

Data sources
------------
1. ``vial_dataset/split_70_10_20`` test/val — treated as **upright** (desk stills;
   labels are YOLO xywh; almost all standing / near-upright poses).
2. ``sideways_race/synth`` — **sideways** positives from manifest boxes;
   **upright** from control notes ``box=[x,y,w,h]``; empties / hard_negatives
   as empty negatives; topdown / real_wide as **other** (optional).

Metrics per orientation bucket (and overall):
  precision / recall / mean IoU @ IoU≥0.5, plus FP on empty frames.

Usage (no ROS required):
  cd E:\\Desktop\\Jessica\\Camera\\ros2_ws\\src\\r2_06_two_stage_orient
  python scripts/evaluate.py
  python scripts/evaluate.py --no-yolo          # classical Stage-A only
  python scripts/evaluate.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
CAMERA_ROOT = PKG_ROOT.parents[2]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from r2_06_two_stage_orient.pipeline import TwoStageOrientDetector  # noqa: E402
from r2_06_two_stage_orient.types import box_iou_xywh  # noqa: E402

NOTE_BOX_RE = re.compile(
    r"box\s*=\s*\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]", re.I
)


def load_yolo_boxes(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
    if not label_path.is_file():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, nw, nh = map(float, parts[1:5])
        bw, bh = nw * img_w, nh * img_h
        x = cx * img_w - bw / 2
        y = cy * img_h - bh / 2
        boxes.append((int(round(x)), int(round(y)), int(round(bw)), int(round(bh))))
    return boxes


def best_iou(preds, gts) -> float:
    if not preds or not gts:
        return 0.0
    return max(box_iou_xywh(p, g) for p in preds for g in gts)


def match_counts(preds, gts, iou_thr: float) -> Tuple[int, int, int, List[float]]:
    """Greedy TP/FP/FN + matched IoUs."""
    if not gts and not preds:
        return 0, 0, 0, []
    if not gts:
        return 0, len(preds), 0, []
    if not preds:
        return 0, 0, len(gts), []
    used = set()
    tps = 0
    ious: List[float] = []
    ordered = sorted(range(len(preds)), key=lambda i: i)  # already scored upstream
    # Prefer higher overlap first
    pairs = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            pairs.append((box_iou_xywh(p, g), pi, gi))
    pairs.sort(reverse=True)
    matched_p = set()
    for iou, pi, gi in pairs:
        if pi in matched_p or gi in used:
            continue
        if iou >= iou_thr:
            matched_p.add(pi)
            used.add(gi)
            tps += 1
            ious.append(iou)
    fp = len(preds) - len(matched_p)
    fn = len(gts) - len(used)
    return tps, fp, fn, ious


class Bucket:
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.ious: List[float] = []
        self.n_pos_images = 0
        self.n_hit_images = 0
        self.n_empty = 0
        self.n_empty_fp = 0

    def add_positive(self, preds, gts, iou_thr: float) -> None:
        self.n_pos_images += 1
        tp, fp, fn, ious = match_counts(preds, gts, iou_thr)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.ious.extend(ious)
        if best_iou(preds, gts) >= iou_thr:
            self.n_hit_images += 1

    def add_empty(self, preds) -> None:
        self.n_empty += 1
        if preds:
            self.n_empty_fp += 1
            self.fp += len(preds)

    def summary(self) -> dict:
        prec = self.tp / max(1, self.tp + self.fp)
        rec = self.tp / max(1, self.tp + self.fn)
        img_rec = self.n_hit_images / max(1, self.n_pos_images)
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "image_recall_at_iou": round(img_rec, 4),
            "mean_iou_matched": round(float(np.mean(self.ious)) if self.ious else 0.0, 4),
            "n_pos_images": self.n_pos_images,
            "n_hit_images": self.n_hit_images,
            "n_empty": self.n_empty,
            "empty_fp_rate": round(self.n_empty_fp / max(1, self.n_empty), 4),
        }


def collect_vial_dataset(
    split_dir: Path, split: str
) -> List[dict]:
    name = "val" if split in ("val", "validate") else split
    img_dir = split_dir / "images" / name
    lbl_dir = split_dir / "labels" / name
    if not img_dir.is_dir():
        return []
    samples = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        samples.append(
            {
                "id": img.stem,
                "orient_gt": "upright",
                "color_path": img,
                "depth_path": None,
                "label_path": lbl_dir / f"{img.stem}.txt",
                "boxes": None,  # filled after load
                "role": "positive",
                "source": "vial_dataset",
                "intrinsics": None,
            }
        )
    return samples


def collect_synth(synth_dir: Path) -> List[dict]:
    man_path = synth_dir / "manifest.json"
    if not man_path.is_file():
        return []
    man = json.loads(man_path.read_text(encoding="utf-8"))
    samples = []
    for entry in man.get("samples", []):
        kind = entry.get("kind", "")
        sid = entry["id"]
        color = synth_dir / entry["color"]
        depth = synth_dir / entry["depth"]
        jpath = synth_dir / f"{sid}.json"
        note = ""
        if jpath.is_file():
            note = json.loads(jpath.read_text(encoding="utf-8")).get("note", "") or ""

        if kind.startswith("sideways") or kind.startswith("hard_sideways"):
            orient = "sideways"
            boxes = [tuple(b) for b in entry.get("boxes") or []]
            role = "positive" if boxes else "empty"
        elif kind == "upright":
            orient = "upright"
            m = NOTE_BOX_RE.search(note)
            boxes = [tuple(map(int, m.groups()))] if m else []
            role = "positive" if boxes else "empty"
        elif kind in ("topdown", "real_wide"):
            orient = "other"
            # Prefer note control box if present; else empty (hard neg for vial AABB).
            m = NOTE_BOX_RE.search(note)
            if m:
                boxes = [tuple(map(int, m.groups()))]
                role = "positive"
            else:
                boxes = [tuple(b) for b in entry.get("boxes") or []]
                role = "positive" if boxes else "empty"
        else:
            # empty / hard_negative
            orient = "empty"
            boxes = []
            role = "empty"

        samples.append(
            {
                "id": sid,
                "orient_gt": orient,
                "color_path": color,
                "depth_path": depth if depth.is_file() else None,
                "label_path": None,
                "boxes": boxes,
                "role": role,
                "source": "sideways_race_synth",
                "intrinsics": entry.get("intrinsics"),
                "kind": kind,
            }
        )
    return samples


def run_eval(args: argparse.Namespace) -> dict:
    detector = TwoStageOrientDetector(
        camera_root=args.camera_root,
        model_path=args.weights or None,
        conf=args.conf,
        use_yolo=not args.no_yolo,
        use_rotation_tta=not args.no_tta,
        use_cap_proposals=not args.no_cap,
        use_sideways_method=not args.no_sideways_method,
        min_conf=args.min_conf,
        max_dets=args.max_dets,
    )
    print(f"Loading detector (use_yolo={not args.no_yolo})...", flush=True)
    t_load = time.perf_counter()
    detector.load()
    print(
        f"Ready backend={detector.backend} weights={detector.model_path!r} "
        f"load_ms={(time.perf_counter()-t_load)*1000:.0f}",
        flush=True,
    )

    samples: List[dict] = []
    if not args.synth_only:
        samples.extend(collect_vial_dataset(Path(args.split_dir), args.split))
    if not args.dataset_only:
        samples.extend(collect_synth(Path(args.synth_dir)))
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    buckets: Dict[str, Bucket] = defaultdict(Bucket)
    overall = Bucket()
    per_sample = []
    t0 = time.perf_counter()

    for i, s in enumerate(samples):
        bgr = cv2.imread(str(s["color_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"skip missing {s['color_path']}", flush=True)
            continue
        h, w = bgr.shape[:2]
        depth = None
        if s.get("depth_path") is not None:
            depth = np.load(s["depth_path"]).astype(np.float32)

        if s["boxes"] is None:
            gts = load_yolo_boxes(s["label_path"], w, h) if s.get("label_path") else []
        else:
            gts = list(s["boxes"])

        dets, stats = detector.detect(bgr, depth_mm=depth, intrinsics=s.get("intrinsics"))
        # Predictions: optionally filter by predicted orientation matching GT bucket.
        preds_all = [d.as_xywh() for d in dets]
        if args.filter_pred_orient and s["orient_gt"] in ("upright", "sideways", "other"):
            preds = [d.as_xywh() for d in dets if d.orientation == s["orient_gt"]]
            # Fall back to all preds for recall if filter empties a true positive frame.
            if not preds and gts:
                preds = preds_all
        else:
            preds = preds_all

        orient = s["orient_gt"]
        role = s["role"]
        if role == "empty" or orient == "empty":
            buckets["empty"].add_empty(preds_all)
            overall.add_empty(preds_all)
            hit = False
            iou = 0.0
        else:
            buckets[orient].add_positive(preds, gts, args.iou)
            overall.add_positive(preds, gts, args.iou)
            iou = best_iou(preds, gts)
            hit = iou >= args.iou

        per_sample.append(
            {
                "id": s["id"],
                "source": s["source"],
                "orient_gt": orient,
                "n_gt": len(gts),
                "n_pred": len(preds_all),
                "pred_orients": [d.orientation for d in dets],
                "best_iou": round(iou, 4),
                "hit": hit,
                "infer_ms": round(stats.inference_ms, 2),
            }
        )
        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f"  [{i+1}/{len(samples)}] last={s['id']} hit={hit} iou={iou:.3f}", flush=True)

    elapsed = time.perf_counter() - t0
    result = {
        "agent": "r2_06_two_stage_orient",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backend": detector.backend,
        "weights": detector.model_path,
        "iou_threshold": args.iou,
        "n_samples": len(per_sample),
        "elapsed_s": round(elapsed, 2),
        "overall": overall.summary(),
        "by_orientation": {k: v.summary() for k, v in sorted(buckets.items())},
        "round1_notes": {
            "agent_06_seg_test_bbox_recall_at_0.5": 0.9,
            "agent_06_failure_modes": [
                "prefer_seg could resolve COCO yolov8n-seg before vial-live det",
                "no rotation TTA / sideways proposals",
                "classical peers expand cap→body downward only",
            ],
        },
        "config": {
            "conf": args.conf,
            "min_conf": args.min_conf,
            "use_yolo": not args.no_yolo,
            "use_rotation_tta": not args.no_tta,
            "use_cap_proposals": not args.no_cap,
            "use_sideways_method": not args.no_sideways_method,
            "filter_pred_orient": args.filter_pred_orient,
            "split": args.split,
            "synth_dir": str(args.synth_dir),
            "split_dir": str(args.split_dir),
        },
        "per_sample": per_sample,
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera-root", type=Path, default=CAMERA_ROOT)
    ap.add_argument(
        "--split-dir",
        type=Path,
        default=CAMERA_ROOT / "vial_dataset" / "split_70_10_20",
    )
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument(
        "--synth-dir",
        type=Path,
        default=CAMERA_ROOT / "sideways_race" / "synth",
    )
    ap.add_argument("--weights", type=str, default="", help="Optional YOLO det weights")
    ap.add_argument("--conf", type=float, default=0.12, help="Stage-A YOLO conf")
    ap.add_argument("--min-conf", type=float, default=0.22, help="Stage-B keep threshold")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-dets", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-cap", action="store_true")
    ap.add_argument("--no-sideways-method", action="store_true")
    ap.add_argument("--synth-only", action="store_true")
    ap.add_argument("--dataset-only", action="store_true")
    ap.add_argument(
        "--filter-pred-orient",
        action="store_true",
        help="Only count preds whose Stage-B class matches GT orientation",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=PKG_ROOT / "results" / "eval_latest.json",
    )
    args = ap.parse_args()

    result = run_eval(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== r2_06_two_stage_orient eval ===")
    print(f"backend: {result['backend']}")
    print(f"weights: {result['weights']}")
    print(f"samples: {result['n_samples']}  elapsed_s: {result['elapsed_s']}")
    print("\nOverall:", json.dumps(result["overall"], indent=2))
    print("\nBy orientation:")
    for k, v in result["by_orientation"].items():
        print(f"  {k}: recall={v['recall']}  precision={v['precision']}  "
              f"img_recall={v['image_recall_at_iou']}  mean_iou={v['mean_iou_matched']}  "
              f"tp/fp/fn={v['tp']}/{v['fp']}/{v['fn']}")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
