#!/usr/bin/env python3
"""Offline eval of Agent 08 depth-geometry on sideways_race/synth (no ROS needed).

Metrics (same spirit as sideways_race/eval_sideways.py):
  - recall@IoU>=0.5 on positives
  - false-positive count / rate on negatives
  - mean IoU on matched positives

Usage (from Camera or package root):
  python evaluate.py
  python evaluate.py --synth-dir E:\\Desktop\\Jessica\\Camera\\sideways_race\\synth
  python evaluate.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PKG_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PKG_ROOT
# PKG_ROOT = .../ros2_ws/src/agent_08_depth_geometry
# parents[0]=src, [1]=ros2_ws, [2]=Camera
CAMERA_ROOT = PKG_ROOT.parents[2]
DEFAULT_SYNTH = CAMERA_ROOT / "sideways_race" / "synth"

# Allow `import agent_08_depth_geometry` without install.
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_08_depth_geometry.geometry_detector import (  # noqa: E402
    NAME,
    detect_vials,
)


def box_iou(a, b) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - overlap
    return 0.0 if union <= 0 else overlap / union


def best_iou(pred_boxes, gt_boxes) -> float:
    if not pred_boxes or not gt_boxes:
        return 0.0
    return max(box_iou(p, g) for p in pred_boxes for g in gt_boxes)


def load_manifest(synth_dir: Path) -> dict:
    path = synth_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synth-dir", type=Path, default=DEFAULT_SYNTH)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="0 = all samples")
    ap.add_argument("--min-score", type=float, default=0.25)
    ap.add_argument(
        "--out",
        type=Path,
        default=PKG_ROOT / "results" / "eval_latest.json",
    )
    args = ap.parse_args()

    synth_dir = args.synth_dir.resolve()
    manifest = load_manifest(synth_dir)
    samples = list(manifest.get("samples", []))
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    tp = 0
    fn = 0
    fp_total = 0
    neg_n = 0
    pos_n = 0
    ious: list[float] = []
    t0 = time.perf_counter()
    per_sample = []

    for entry in samples:
        color = cv2.imread(str(synth_dir / entry["color"]), cv2.IMREAD_COLOR)
        if color is None:
            raise FileNotFoundError(entry["color"])
        depth = np.load(synth_dir / entry["depth"]).astype(np.float32)
        K = entry.get("intrinsics")
        dets = detect_vials(
            color, depth, K, allow_pseudo_depth=False, min_score=args.min_score
        )
        # For race parity, count all boxes (upright+sideways) against GT.
        preds = [d.box_xywh for d in dets]
        gts = [tuple(b) for b in entry.get("boxes", [])]
        role = entry.get("role", "positive")
        iou = best_iou(preds, gts) if gts else 0.0
        hit = bool(gts) and iou >= args.iou

        if role == "positive":
            pos_n += 1
            if hit:
                tp += 1
                ious.append(iou)
            else:
                fn += 1
        else:
            neg_n += 1
            # Any prediction on a negative is a false positive sample.
            if preds:
                fp_total += len(preds)

        per_sample.append(
            {
                "id": entry.get("id"),
                "role": role,
                "kind": entry.get("kind"),
                "n_pred": len(preds),
                "best_iou": iou,
                "hit": hit,
            }
        )

    elapsed = time.perf_counter() - t0
    recall = tp / max(pos_n, 1)
    fp_rate = fp_total / max(neg_n, 1)
    mean_iou = float(np.mean(ious)) if ious else 0.0

    report = {
        "method": NAME,
        "agent": "agent_08_depth_geometry",
        "synth_dir": str(synth_dir),
        "n_samples": len(samples),
        "n_positive": pos_n,
        "n_negative": neg_n,
        "recall_at_iou": recall,
        "tp": tp,
        "fn": fn,
        "fp_count": fp_total,
        "fp_rate_per_negative": fp_rate,
        "mean_iou_matched": mean_iou,
        "iou_thresh": args.iou,
        "min_score": args.min_score,
        "seconds": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "suitability": (
            "HIGH — Orbbec RGB-D depth exists (live topics + synth *_depth.npy). "
            "Geometric plane/cylinder pipeline is the intended use of this agent."
        ),
        "samples": per_sample,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"method={NAME}")
    print(f"samples={len(samples)} pos={pos_n} neg={neg_n}")
    print(f"recall@{args.iou}={recall:.4f} ({tp}/{pos_n})")
    print(f"fp_count={fp_total} fp_rate={fp_rate:.4f}")
    print(f"mean_iou_matched={mean_iou:.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
