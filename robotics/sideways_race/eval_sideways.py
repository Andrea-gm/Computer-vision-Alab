"""Score sideways-vial method modules for the Best-of-N race (eval only — not live).

Method API
----------
Each module under sideways_race/methods/ must expose:

    NAME: str  # optional; defaults to stem
    def detect_sideways(color_bgr, depth_mm, intrinsics=None) -> list[tuple[x, y, w, h]]

Metrics
-------
- recall@IoU>=0.5 on positives (sideways left/right)
- false-positive rate on negatives (empty / upright / top-down / real_wide)
- mean IoU on matched positives

Writes JSON rank tables under sideways_race/results/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
CAMERA = ROOT.parent
METHODS_DIR = ROOT / "methods"
SYNTH_DIR = ROOT / "synth"
RESULTS_DIR = ROOT / "results"
IOU_THRESH = 0.5

if str(CAMERA) not in sys.path:
    sys.path.insert(0, str(CAMERA))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    if not path.exists():
        from generate_synth import generate

        print(f"No {path}; generating synth...")
        return generate()
    return json.loads(path.read_text(encoding="utf-8"))


def load_sample(synth_dir: Path, entry: dict) -> tuple[np.ndarray, np.ndarray, dict | None]:
    color = cv2.imread(str(synth_dir / entry["color"]), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(entry["color"])
    depth = np.load(synth_dir / entry["depth"]).astype(np.float32)
    K = entry.get("intrinsics")
    return color, depth, K


def discover_methods(methods_dir: Path, only: list[str] | None = None) -> list[Path]:
    paths = sorted(methods_dir.glob("*.py"))
    out = []
    for p in paths:
        if p.name.startswith("_"):
            continue
        if only and p.stem not in only and p.name not in only:
            # Also allow NAME-less filter by substring
            if not any(tok in p.stem for tok in only):
                continue
        out.append(p)
    return out


def load_method(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "detect_sideways"):
        raise AttributeError(f"{path.name} missing detect_sideways(...)")
    name = getattr(mod, "NAME", path.stem)
    return name, mod.detect_sideways


def score_method(name: str, detect_fn, samples: list[dict], synth_dir: Path) -> dict:
    n_pos = 0
    n_pos_hit = 0
    ious: list[float] = []
    n_neg = 0
    n_fp = 0
    per_sample = []
    t0 = time.perf_counter()
    errors = 0

    for entry in samples:
        role = entry.get("role", "positive")
        gt = [tuple(b) for b in entry.get("boxes") or []]
        try:
            color, depth, K = load_sample(synth_dir, entry)
            preds = detect_fn(color, depth, K) or []
            preds = [tuple(int(v) for v in b) for b in preds]
        except Exception as exc:
            errors += 1
            per_sample.append(
                {
                    "id": entry["id"],
                    "role": role,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=4),
                }
            )
            if role == "positive":
                n_pos += 1
                ious.append(0.0)
            else:
                n_neg += 1
                n_fp += 1  # failure on negative counts as FP (conservative)
            continue

        if role == "positive":
            n_pos += 1
            iou = best_iou(preds, gt)
            ious.append(iou)
            hit = iou >= IOU_THRESH
            if hit:
                n_pos_hit += 1
            per_sample.append(
                {
                    "id": entry["id"],
                    "role": role,
                    "kind": entry.get("kind"),
                    "gt": [list(b) for b in gt],
                    "pred": [list(b) for b in preds],
                    "best_iou": round(iou, 4),
                    "hit": hit,
                }
            )
        else:
            n_neg += 1
            fp = len(preds) > 0
            if fp:
                n_fp += 1
            per_sample.append(
                {
                    "id": entry["id"],
                    "role": role,
                    "kind": entry.get("kind"),
                    "gt": [],
                    "pred": [list(b) for b in preds],
                    "fp": fp,
                }
            )

    elapsed = time.perf_counter() - t0
    recall = (n_pos_hit / n_pos) if n_pos else 0.0
    fp_rate = (n_fp / n_neg) if n_neg else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    # Rank score: prioritize recall, then low FP, then mean IoU
    rank_score = recall - 0.35 * fp_rate + 0.15 * mean_iou

    return {
        "name": name,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "recall_at_iou_0_5": round(recall, 4),
        "fp_count": n_fp,
        "fp_rate": round(fp_rate, 4),
        "mean_iou": round(mean_iou, 4),
        "rank_score": round(rank_score, 4),
        "errors": errors,
        "seconds": round(elapsed, 3),
        "per_sample": per_sample,
    }


def print_table(rows: list[dict]) -> None:
    headers = ("rank", "name", "recall@0.5", "FP", "fp_rate", "mean_IoU", "score", "sec")
    print("  ".join(f"{h:>10}" for h in headers))
    for i, r in enumerate(rows, 1):
        print(
            f"{i:>10}"
            f"  {r['name']:>10}"
            f"  {r['recall_at_iou_0_5']:>10.3f}"
            f"  {r['fp_count']:>10}"
            f"  {r['fp_rate']:>10.3f}"
            f"  {r['mean_iou']:>10.3f}"
            f"  {r['rank_score']:>10.3f}"
            f"  {r['seconds']:>10.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score sideways-vial race methods")
    parser.add_argument("--synth", type=Path, default=SYNTH_DIR, help="Synth directory with manifest.json")
    parser.add_argument("--methods", type=Path, default=METHODS_DIR, help="Methods directory")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR, help="Output JSON directory")
    parser.add_argument("--only", nargs="*", default=None, help="Only score these method stems")
    parser.add_argument("--regen", action="store_true", help="Regenerate synth before eval")
    parser.add_argument("--no-detail", action="store_true", help="Omit per-sample rows in JSON")
    args = parser.parse_args(argv)

    if args.regen or not (args.synth / "manifest.json").exists():
        from generate_synth import generate

        generate()

    manifest = load_manifest(args.synth)
    samples = manifest["samples"]
    method_paths = discover_methods(args.methods, args.only)
    if not method_paths:
        print(f"No methods found in {args.methods}", file=sys.stderr)
        return 2

    args.results.mkdir(parents=True, exist_ok=True)
    scored = []
    for path in method_paths:
        try:
            name, fn = load_method(path)
        except Exception as exc:
            print(f"SKIP {path.name}: {exc}", file=sys.stderr)
            scored.append(
                {
                    "name": path.stem,
                    "error": str(exc),
                    "recall_at_iou_0_5": 0.0,
                    "fp_count": 999,
                    "fp_rate": 1.0,
                    "mean_iou": 0.0,
                    "rank_score": -1.0,
                    "seconds": 0.0,
                }
            )
            continue
        print(f"Scoring {name} ({path.name})...")
        result = score_method(name, fn, samples, args.synth)
        scored.append(result)

    ranked = sorted(
        scored,
        key=lambda r: (
            -float(r.get("recall_at_iou_0_5", 0)),
            int(r.get("fp_count", 999)),
            -float(r.get("mean_iou", 0)),
        ),
    )
    print()
    print_table(ranked)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = {
        "created_utc": stamp,
        "iou_threshold": IOU_THRESH,
        "synth": str(args.synth.resolve()),
        "n_samples": len(samples),
        "manifest_counts": {
            "positive": manifest.get("n_positive"),
            "negative": manifest.get("n_negative"),
            "by_kind": manifest.get("counts_by_kind"),
        },
        "ranking": [
            {
                k: v
                for k, v in r.items()
                if k != "per_sample" or not args.no_detail
            }
            for r in ranked
        ],
        "winner": ranked[0]["name"] if ranked else None,
        "note": "Eval harness only — do not promote to live depthcamera without supervisor gate.",
    }
    out_path = args.results / f"race_{stamp}.json"
    latest = args.results / "latest.json"
    # Compact latest without huge per_sample unless useful — keep per_sample in stamped file
    latest_body = {
        **summary,
        "ranking": [
            {k: v for k, v in r.items() if k != "per_sample"} for r in ranked
        ],
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    latest.write_text(json.dumps(latest_body, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"Wrote {latest}")
    if ranked:
        print(f"Winner (eval only, not live): {ranked[0]['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
