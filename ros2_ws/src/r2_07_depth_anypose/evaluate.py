#!/usr/bin/env python3
"""Offline eval of r2_07_depth_anypose on sideways_race/synth (no ROS).

Reports:
  - race_sideways: recall@0.5 / FP on manifest roles (sideways race parity)
  - anypose: sideways positives + upright/topdown control GT from notes
  - per-kind breakdown

Usage:
  python evaluate.py
  python evaluate.py --synth-dir E:\\Desktop\\Jessica\\Camera\\sideways_race\\synth
  python evaluate.py --limit 20
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

import cv2
import numpy as np

PKG_ROOT = Path(__file__).resolve().parent
CAMERA_ROOT = PKG_ROOT.parents[2]
DEFAULT_SYNTH = CAMERA_ROOT / "sideways_race" / "synth"

if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from r2_07_depth_anypose.anypose_detector import (  # noqa: E402
    NAME,
    detect_vials,
)

_NOTE_BOX_RE = re.compile(
    r"control GT[^=]*=\[([^\]]+)\]", re.IGNORECASE
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


def parse_control_gt(entry: dict) -> list[tuple[int, int, int, int]]:
    """Upright/topdown samples store control boxes in the note field."""
    note = entry.get("note") or ""
    m = _NOTE_BOX_RE.search(note)
    if not m:
        return []
    try:
        vals = [int(float(x.strip())) for x in m.group(1).split(",")]
    except ValueError:
        return []
    if len(vals) != 4:
        return []
    return [tuple(vals)]  # type: ignore[list-item]


def load_manifest(synth_dir: Path) -> dict:
    path = synth_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synth-dir", type=Path, default=DEFAULT_SYNTH)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
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

    # --- Race metrics (manifest role) ---
    race_tp = race_fn = race_fp = race_pos = race_neg = 0
    race_ious: list[float] = []

    # --- Any-pose metrics (sideways + upright/topdown control GT) ---
    ap_tp = ap_fn = ap_fp = ap_pos = ap_neg = 0
    ap_ious: list[float] = []
    by_kind: dict[str, dict] = defaultdict(
        lambda: {"tp": 0, "fn": 0, "fp": 0, "pos": 0, "neg": 0, "ious": []}
    )

    t0 = time.perf_counter()
    per_sample = []

    for entry in samples:
        # Prefer per-file JSON (has note with control GT).
        meta_path = synth_dir / f"{entry['id']}.json"
        if meta_path.is_file():
            entry = json.loads(meta_path.read_text(encoding="utf-8"))

        color = cv2.imread(str(synth_dir / entry["color"]), cv2.IMREAD_COLOR)
        if color is None:
            raise FileNotFoundError(entry["color"])
        depth = np.load(synth_dir / entry["depth"]).astype(np.float32)
        K = entry.get("intrinsics")
        dets = detect_vials(
            color,
            depth,
            K,
            allow_rgb_fallback=True,
            min_score=args.min_score,
        )
        preds = [d.box_xywh for d in dets]
        # Sideways-race parity: only lying / angled boxes count (upright on
        # upright control frames is correct for any-pose, not a race FP).
        preds_race = [
            d.box_xywh
            for d in dets
            if d.orientation in ("sideways", "angled")
        ]
        kind = entry.get("kind", "?")
        role = entry.get("role", "positive")
        gts_race = [tuple(b) for b in entry.get("boxes", [])]

        # Race scoring
        iou_race = best_iou(preds_race, gts_race) if gts_race else 0.0
        hit_race = bool(gts_race) and iou_race >= args.iou
        if role == "positive":
            race_pos += 1
            if hit_race:
                race_tp += 1
                race_ious.append(iou_race)
            else:
                race_fn += 1
        else:
            race_neg += 1
            if preds_race:
                race_fp += len(preds_race)

        # Any-pose GT: race boxes OR upright/topdown control notes
        gts_ap = list(gts_race)
        if not gts_ap and kind in ("upright", "topdown"):
            gts_ap = parse_control_gt(entry)

        is_ap_positive = bool(gts_ap)
        iou_ap = best_iou(preds, gts_ap) if gts_ap else 0.0
        hit_ap = is_ap_positive and iou_ap >= args.iou
        bk = by_kind[kind]
        if is_ap_positive:
            ap_pos += 1
            bk["pos"] += 1
            if hit_ap:
                ap_tp += 1
                ap_ious.append(iou_ap)
                bk["tp"] += 1
                bk["ious"].append(iou_ap)
            else:
                ap_fn += 1
                bk["fn"] += 1
        else:
            ap_neg += 1
            bk["neg"] += 1
            if preds:
                ap_fp += len(preds)
                bk["fp"] += len(preds)

        per_sample.append(
            {
                "id": entry.get("id"),
                "role": role,
                "kind": kind,
                "n_pred": len(preds),
                "orientations": [d.orientation for d in dets],
                "best_iou_race": iou_race,
                "hit_race": hit_race,
                "best_iou_anypose": iou_ap,
                "hit_anypose": hit_ap,
                "n_gt_anypose": len(gts_ap),
            }
        )

    elapsed = time.perf_counter() - t0
    kind_summary = {}
    for k, v in sorted(by_kind.items()):
        kind_summary[k] = {
            "pos": v["pos"],
            "neg": v["neg"],
            "recall": v["tp"] / max(v["pos"], 1),
            "tp": v["tp"],
            "fn": v["fn"],
            "fp_count": v["fp"],
            "mean_iou": float(np.mean(v["ious"])) if v["ious"] else 0.0,
        }

    report = {
        "method": NAME,
        "agent": "r2_07_depth_anypose",
        "synth_dir": str(synth_dir),
        "n_samples": len(samples),
        "race_sideways": {
            "n_positive": race_pos,
            "n_negative": race_neg,
            "recall_at_iou": race_tp / max(race_pos, 1),
            "tp": race_tp,
            "fn": race_fn,
            "fp_count": race_fp,
            "fp_rate_per_negative": race_fp / max(race_neg, 1),
            "mean_iou_matched": float(np.mean(race_ious)) if race_ious else 0.0,
        },
        "anypose": {
            "n_positive": ap_pos,
            "n_negative": ap_neg,
            "recall_at_iou": ap_tp / max(ap_pos, 1),
            "tp": ap_tp,
            "fn": ap_fn,
            "fp_count": ap_fp,
            "fp_rate_per_negative": ap_fp / max(ap_neg, 1),
            "mean_iou_matched": float(np.mean(ap_ious)) if ap_ious else 0.0,
            "note": (
                "Positives = sideways race boxes + upright/topdown control GT "
                "parsed from sample notes."
            ),
        },
        "by_kind": kind_summary,
        "iou_thresh": args.iou,
        "min_score": args.min_score,
        "seconds": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pose_approach": (
            "White-cap seed → clear-depth/dark-RGB body scan (8 dirs) → "
            "3D PCA major axis on Z-band cluster (vertical OR horizontal OR angled) → "
            "cylinder prior Ø≈54×H≈73 mm projected to AABB; RGB YOLO-or-classical "
            "fallback when depth sparse/missing."
        ),
        "samples": per_sample,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rs = report["race_sideways"]
    apm = report["anypose"]
    print(f"method={NAME}")
    print(f"samples={len(samples)}")
    print(
        f"RACE  recall@{args.iou}={rs['recall_at_iou']:.4f} "
        f"({rs['tp']}/{rs['n_positive']}) fp={rs['fp_count']} "
        f"mean_iou={rs['mean_iou_matched']:.4f}"
    )
    print(
        f"ANYPOSE recall@{args.iou}={apm['recall_at_iou']:.4f} "
        f"({apm['tp']}/{apm['n_positive']}) fp={apm['fp_count']} "
        f"mean_iou={apm['mean_iou_matched']:.4f}"
    )
    for k, v in kind_summary.items():
        if v["pos"]:
            print(
                f"  {k}: recall={v['recall']:.3f} ({v['tp']}/{v['pos']}) "
                f"mean_iou={v['mean_iou']:.3f}"
            )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
