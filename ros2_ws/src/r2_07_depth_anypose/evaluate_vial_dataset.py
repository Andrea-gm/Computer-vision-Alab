#!/usr/bin/env python3
"""Evaluate RGB (and optional depth) any-pose detector on vial_dataset upright stills.

Uses YOLO split_70_10_20 val+test images with aspect(h/w) >= 1.05 as upright GT.
No depth available in this split — RGB path only.

Usage:
  python evaluate_vial_dataset.py
  python evaluate_vial_dataset.py --no-yolo   # classical blob only
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
CAMERA_ROOT = PKG_ROOT.parents[2]
DEFAULT_SPLIT = CAMERA_ROOT / "vial_dataset" / "split_70_10_20"

if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from r2_07_depth_anypose.anypose_detector import (  # noqa: E402
    NAME,
    detect_vials,
)


def load_yolo_boxes(lbl: Path, w: int, h: int) -> list[tuple[int, int, int, int]]:
    if not lbl.is_file():
        return []
    out: list[tuple[int, int, int, int]] = []
    for line in lbl.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = (float(v) for v in parts[:5])
        x = int(round((cx - bw / 2.0) * w))
        y = int(round((cy - bh / 2.0) * h))
        bw_i = int(round(bw * w))
        bh_i = int(round(bh * h))
        if bw_i > 1 and bh_i > 1:
            out.append((x, y, bw_i, bh_i))
    return out


def box_iou(a, b) -> float:
    ax, ay, aw, ah = (float(v) for v in a)
    bx, by, bw, bh = (float(v) for v in b)
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    uni = aw * ah + bw * bh - inter
    return 0.0 if uni <= 0 else inter / uni


def eval_split(
    split_dir: Path,
    split: str,
    *,
    iou_thr: float,
    min_score: float,
    use_yolo: bool,
    min_aspect: float,
) -> dict:
    img_dir = split_dir / "images" / split
    lbl_dir = split_dir / "labels" / split
    imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    tp = fp = fn = 0
    ious: list[float] = []
    n_pos = n_neg = 0
    per = []
    for p in imgs:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        gts = [
            g
            for g in load_yolo_boxes(lbl_dir / f"{p.stem}.txt", w, h)
            if g[3] / float(max(g[2], 1)) >= min_aspect
        ]
        # Still split has no depth npy.
        dets = detect_vials(
            im,
            None,
            None,
            allow_rgb_fallback=True,
            use_yolo_rgb=use_yolo,
            min_score=min_score,
        )
        preds = [d.box_xywh for d in dets]
        if not gts:
            n_neg += 1
            fp += len(preds)
            per.append({"id": p.name, "role": "neg", "n_pred": len(preds)})
            continue
        n_pos += 1
        used: set[int] = set()
        hits = 0
        best = 0.0
        for g in gts:
            bi, bj = -1, 0.0
            for i, pr in enumerate(preds):
                if i in used:
                    continue
                v = box_iou(pr, g)
                if v > bj:
                    bi, bj = i, v
            best = max(best, bj)
            if bi >= 0 and bj >= iou_thr:
                used.add(bi)
                hits += 1
                ious.append(bj)
        tp += hits
        fn += len(gts) - hits
        fp += len(preds) - len(used)
        per.append(
            {
                "id": p.name,
                "role": "pos",
                "n_pred": len(preds),
                "n_gt": len(gts),
                "hits": hits,
                "best_iou": best,
            }
        )
    return {
        "split": split,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall_at_iou": tp / max(tp + fn, 1),
        "precision_at_iou": tp / max(tp + fp, 1),
        "mean_iou_matched": float(np.mean(ious)) if ious else 0.0,
        "samples": per,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-score", type=float, default=0.25)
    ap.add_argument("--min-aspect", type=float, default=1.05, help="upright GT filter")
    ap.add_argument("--no-yolo", action="store_true", help="classical RGB only")
    ap.add_argument(
        "--out",
        type=Path,
        default=PKG_ROOT / "results" / "eval_vial_dataset_upright.json",
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    use_yolo = not args.no_yolo
    val = eval_split(
        args.split_dir,
        "val",
        iou_thr=args.iou,
        min_score=args.min_score,
        use_yolo=use_yolo,
        min_aspect=args.min_aspect,
    )
    test = eval_split(
        args.split_dir,
        "test",
        iou_thr=args.iou,
        min_score=args.min_score,
        use_yolo=use_yolo,
        min_aspect=args.min_aspect,
    )
    tp = val["tp"] + test["tp"]
    fp = val["fp"] + test["fp"]
    fn = val["fn"] + test["fn"]
    report = {
        "method": NAME,
        "agent": "r2_07_depth_anypose",
        "split_dir": str(args.split_dir.resolve()),
        "mode": "rgb_yolo" if use_yolo else "rgb_classical",
        "depth_available": False,
        "note": (
            "vial_dataset/split_70_10_20 has RGB stills only (no depth npy). "
            "Upright GT = boxes with h/w >= min_aspect."
        ),
        "iou_thresh": args.iou,
        "min_score": args.min_score,
        "min_aspect": args.min_aspect,
        "val": {k: v for k, v in val.items() if k != "samples"},
        "test": {k: v for k, v in test.items() if k != "samples"},
        "combined": {
            "recall_at_iou": tp / max(tp + fn, 1),
            "precision_at_iou": tp / max(tp + fp, 1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_positive": val["n_positive"] + test["n_positive"],
            "n_negative": val["n_negative"] + test["n_negative"],
        },
        "seconds": time.perf_counter() - t0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samples_val": val["samples"],
        "samples_test": test["samples"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    c = report["combined"]
    print(f"method={NAME} mode={report['mode']}")
    print(
        f"val  recall={val['recall_at_iou']:.3f} prec={val['precision_at_iou']:.3f} "
        f"({val['tp']}/{val['tp']+val['fn']})"
    )
    print(
        f"test recall={test['recall_at_iou']:.3f} prec={test['precision_at_iou']:.3f} "
        f"({test['tp']}/{test['tp']+test['fn']})"
    )
    print(
        f"combined recall@{args.iou}={c['recall_at_iou']:.3f} "
        f"prec={c['precision_at_iou']:.3f} tp={c['tp']} fp={c['fp']} fn={c['fn']}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
