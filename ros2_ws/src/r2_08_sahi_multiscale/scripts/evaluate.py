#!/usr/bin/env python3
"""Evaluate SAHI/multi-scale vs single-scale baseline — upright AND sideways.

Datasets
--------
- upright: sideways_race/synth upright_* (control GT parsed from JSON notes)
           + optional vial_dataset split test (real upright-ish crops)
- sideways: sideways_race/synth role=positive (sideways / hard_sideways)

Usage (no ROS required):
  cd E:\\Desktop\\Jessica\\Camera\\ros2_ws\\src\\r2_08_sahi_multiscale
  python scripts/evaluate.py
  python scripts/evaluate.py --max-images 20   # smoke
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2_08_sahi_multiscale.detector import (  # noqa: E402
    SahiMultiscaleDetector,
    list_images,
    load_yolo_label,
)
from r2_08_sahi_multiscale.small_object_set import (  # noqa: E402
    build_small_object_items,
    maybe_write_preview,
)
from r2_08_sahi_multiscale.types import Det, box_iou  # noqa: E402
from r2_08_sahi_multiscale.weights import default_camera_root, resolve_model_path  # noqa: E402

NOTE_BOX_RE = re.compile(
    r"box\s*=\s*\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]",
    re.IGNORECASE,
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


def xywh_to_det(box: Sequence[float]) -> Det:
    x, y, w, h = [float(v) for v in box]
    return Det(x, y, x + w, y + h, conf=1.0, label="vial", source="gt")


def load_synth_samples(synth_dir: Path) -> List[dict]:
    manifest = synth_dir / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return list(data.get("samples") or [])
    samples = []
    for p in sorted(synth_dir.glob("*.json")):
        if p.name == "manifest.json":
            continue
        samples.append(json.loads(p.read_text(encoding="utf-8")))
    return samples


def upright_gt_from_entry(entry: dict, synth_dir: Path) -> List[Det]:
    """Parse control upright GT from JSON note (boxes[] is intentionally empty)."""
    note = entry.get("note") or ""
    m = NOTE_BOX_RE.search(note)
    if m:
        return [xywh_to_det([int(m.group(i)) for i in range(1, 5)])]
    # Fall back to per-file JSON if note missing
    jpath = synth_dir / f"{entry['id']}.json"
    if jpath.exists():
        data = json.loads(jpath.read_text(encoding="utf-8"))
        note = data.get("note") or ""
        m = NOTE_BOX_RE.search(note)
        if m:
            return [xywh_to_det([int(m.group(i)) for i in range(1, 5)])]
    return []


def collect_eval_items(
    camera_root: Path,
    *,
    suite: str,
    include_vial_dataset: bool,
    max_images: Optional[int],
    small_per_orient: int,
) -> Dict[str, List[dict]]:
    """Return {orientation: [{path|bgr, gts, id}, ...]}."""
    items: Dict[str, List[dict]] = {"upright": [], "sideways": []}
    suite = (suite or "both").lower()

    if suite in ("synth", "both"):
        synth_dir = camera_root / "sideways_race" / "synth"
        if synth_dir.is_dir():
            for entry in load_synth_samples(synth_dir):
                kind = entry.get("kind") or ""
                color = entry.get("color") or f"{entry.get('id', '')}_color.png"
                path = synth_dir / color
                if not path.exists():
                    continue
                if entry.get("role") == "positive" or kind.startswith("sideways") or kind.startswith(
                    "hard_sideways"
                ):
                    gts = [xywh_to_det(b) for b in (entry.get("boxes") or [])]
                    if gts:
                        items["sideways"].append(
                            {
                                "path": path,
                                "gts": gts,
                                "id": entry.get("id", path.stem),
                                "kind": kind,
                            }
                        )
                elif kind == "upright":
                    gts = upright_gt_from_entry(entry, synth_dir)
                    if gts:
                        items["upright"].append(
                            {
                                "path": path,
                                "gts": gts,
                                "id": entry.get("id", path.stem),
                                "kind": kind,
                            }
                        )

    if suite in ("small", "both"):
        small = build_small_object_items(
            camera_root,
            per_orient=int(small_per_orient),
            target_max_side=64,
            canvas_w=1280,
            canvas_h=720,
        )
        preview_dir = Path(__file__).resolve().parents[1] / "eval_previews"
        maybe_write_preview(small, preview_dir, n=2)
        for orient in ("upright", "sideways"):
            items[orient].extend(small.get(orient) or [])

    if include_vial_dataset:
        split = camera_root / "vial_dataset" / "split_70_10_20" / "images" / "test"
        lbl_dir = camera_root / "vial_dataset" / "split_70_10_20" / "labels" / "test"
        if not split.is_dir():
            split = camera_root / "vial_dataset" / "raw"
            lbl_dir = split
        for img in list_images(split):
            bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            lbl = lbl_dir / f"{img.stem}.txt" if lbl_dir != split else img.with_suffix(".txt")
            gts = load_yolo_label(lbl, w, h)
            if not gts:
                continue
            items["upright"].append(
                {"path": img, "gts": gts, "id": img.stem, "kind": "vial_dataset"}
            )

    if max_images is not None:
        for key in items:
            items[key] = items[key][: max(0, int(max_images))]
    return items


def evaluate_bucket(
    items: List[dict],
    detector: SahiMultiscaleDetector,
    *,
    iou_thr: float,
) -> dict:
    tp = fp = fn = 0
    all_ious: List[float] = []
    times_ms: List[float] = []
    image_hits = 0
    n = 0
    for it in items:
        if it.get("bgr") is not None:
            bgr = it["bgr"]
        else:
            bgr = cv2.imread(str(it["path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        n += 1
        gts: List[Det] = it["gts"]
        t0 = time.perf_counter()
        preds, stats = detector.predict(bgr)
        ms = (time.perf_counter() - t0) * 1000.0
        if stats.inference_ms > 0:
            ms = stats.inference_ms
        times_ms.append(ms)
        t, f, miss, ious = match_dets(preds, gts, iou_thr=iou_thr)
        tp += t
        fp += f
        fn += miss
        all_ious.extend(ious)
        if t > 0:
            image_hits += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    mean_ms = sum(times_ms) / len(times_ms) if times_ms else 0.0
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return {
        "n_images": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mean_iou": round(mean_iou, 4),
        "image_recall": round(image_hits / n, 4) if n else 0.0,
        "mean_infer_ms": round(mean_ms, 2),
        "fps": round(fps, 3),
    }


def build_detector(mode: str, args: argparse.Namespace) -> SahiMultiscaleDetector:
    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    return SahiMultiscaleDetector(
        args.model or None,
        camera_root=args.camera_root,
        confidence_threshold=args.conf,
        iou_threshold=args.iou_nms,
        imgsz=args.imgsz,
        mode=mode,
        slice_width=args.slice_w,
        slice_height=args.slice_h,
        overlap_ratio=args.overlap,
        image_scales=scales,
        rotation_tta=(mode == "sahi") and (not args.no_rotation_tta),
    )


def main() -> int:
    cam = default_camera_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera-root", default=str(cam))
    ap.add_argument(
        "--model",
        default="",
        help="YOLO .pt (default: auto-resolve bon-winner → r1 → r3 → vial-live)",
    )
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-thr", type=float, default=0.5, help="Match IoU gate")
    ap.add_argument("--iou-nms", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--slice-w", type=int, default=320)
    ap.add_argument("--slice-h", type=int, default=320)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--scales", default="1.0,1.5")
    ap.add_argument("--no-rotation-tta", action="store_true")
    ap.add_argument(
        "--suite",
        default="both",
        choices=("synth", "small", "both"),
        help="synth=sideways_race; small=pasted real crops; both=union",
    )
    ap.add_argument("--small-per-orient", type=int, default=20)
    ap.add_argument("--include-vial-dataset", action="store_true")
    ap.add_argument("--max-images", type=int, default=None, help="Per-orientation cap")
    ap.add_argument(
        "--out",
        default=str(ROOT / "eval_results.json"),
        help="Write metrics JSON here",
    )
    args = ap.parse_args()
    camera_root = Path(args.camera_root)

    model = resolve_model_path(args.model or None, camera_root)
    if model is None:
        print("ERROR: no YOLO weights found", file=sys.stderr)
        return 2

    items = collect_eval_items(
        camera_root,
        suite=args.suite,
        include_vial_dataset=bool(args.include_vial_dataset),
        max_images=args.max_images,
        small_per_orient=args.small_per_orient,
    )
    n_up = len(items["upright"])
    n_side = len(items["sideways"])
    if n_up == 0 or n_side == 0:
        print(
            f"ERROR: need both upright and sideways samples (got upright={n_up}, sideways={n_side})",
            file=sys.stderr,
        )
        return 3

    print(f"weights: {model}")
    print(f"upright samples: {n_up} | sideways samples: {n_side}")
    print(f"tile: {args.slice_w}x{args.slice_h} overlap={args.overlap} scales={args.scales}")

    results = {
        "model": str(model),
        "camera_root": str(camera_root),
        "iou_match": args.iou_thr,
        "conf": args.conf,
        "tile": {
            "slice_w": args.slice_w,
            "slice_h": args.slice_h,
            "overlap": args.overlap,
            "scales": args.scales,
            "rotation_tta": not args.no_rotation_tta,
        },
        "baseline": {},
        "sahi": {},
        "delta_recall": {},
    }

    for mode in ("baseline", "sahi"):
        print(f"\n=== mode={mode} ===")
        det = build_detector(mode, args)
        for orient in ("upright", "sideways"):
            metrics = evaluate_bucket(items[orient], det, iou_thr=args.iou_thr)
            results[mode][orient] = metrics
            print(
                f"  {orient:9s}  recall={metrics['recall']:.3f}  "
                f"precision={metrics['precision']:.3f}  "
                f"mean_IoU={metrics['mean_iou']:.3f}  "
                f"image_recall={metrics['image_recall']:.3f}  "
                f"ms={metrics['mean_infer_ms']:.1f}  fps={metrics['fps']:.2f}  "
                f"(tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} n={metrics['n_images']})"
            )

    for orient in ("upright", "sideways"):
        br = results["baseline"][orient]["recall"]
        sr = results["sahi"][orient]["recall"]
        results["delta_recall"][orient] = round(sr - br, 4)
        print(f"delta recall ({orient}): {sr - br:+.4f}  (sahi {sr:.4f} - baseline {br:.4f})")

    # FPS tradeoff summary
    base_ms = (
        results["baseline"]["upright"]["mean_infer_ms"]
        + results["baseline"]["sideways"]["mean_infer_ms"]
    ) / 2.0
    sahi_ms = (
        results["sahi"]["upright"]["mean_infer_ms"]
        + results["sahi"]["sideways"]["mean_infer_ms"]
    ) / 2.0
    results["fps_tradeoff"] = {
        "baseline_mean_ms": round(base_ms, 2),
        "sahi_mean_ms": round(sahi_ms, 2),
        "slowdown_x": round(sahi_ms / base_ms, 2) if base_ms > 0 else None,
        "baseline_fps": round(1000.0 / base_ms, 3) if base_ms > 0 else 0.0,
        "sahi_fps": round(1000.0 / sahi_ms, 3) if sahi_ms > 0 else 0.0,
    }
    print(
        f"\nFPS tradeoff: baseline {results['fps_tradeoff']['baseline_fps']:.2f} fps "
        f"({base_ms:.1f} ms) vs sahi {results['fps_tradeoff']['sahi_fps']:.2f} fps "
        f"({sahi_ms:.1f} ms) → {results['fps_tradeoff']['slowdown_x']}× slower"
    )

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
