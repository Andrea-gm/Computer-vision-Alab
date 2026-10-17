#!/usr/bin/env python3
"""Offline OBB evaluation — upright vs sideways, polygon IoU @ 0.5.

No ROS required. Sources:
  - upright: vial_dataset/raw (AABB→OBB GT) + sideways_race upright controls
  - sideways: sideways_race synth positives (incl. hard angled)

Also reports AABB envelope IoU for comparison with Round-1 detectors.
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

from r2_03_obb.obb_detector import VialOBBDetector  # noqa: E402
from r2_03_obb.obb_geometry import (  # noqa: E402
    OBBDet,
    aabb_iou,
    aabb_xywh_to_obb,
    load_yolo_aabb_label,
    parse_angle_from_note,
    polygon_iou,
)

CAMERA = ROOT.parents[2]
IOU_THR = 0.5


def _match(
    preds: Sequence[OBBDet],
    gts: Sequence[OBBDet],
    *,
    iou_thr: float,
    use_obb: bool,
) -> Tuple[int, int, int, List[float]]:
    if not gts and not preds:
        return 0, 0, 0, []
    if not gts:
        return 0, len(preds), 0, []
    if not preds:
        return 0, 0, len(gts), []
    used = set()
    tps = 0
    ious: List[float] = []
    ordered = sorted(preds, key=lambda d: d.conf, reverse=True)
    for p in ordered:
        best_i, best = -1, 0.0
        for gi, g in enumerate(gts):
            if gi in used:
                continue
            iou = polygon_iou(p, g) if use_obb else aabb_iou(p.aabb(), g.aabb())
            if iou > best:
                best, best_i = iou, gi
        if best_i >= 0 and best >= iou_thr:
            used.add(best_i)
            tps += 1
            ious.append(best)
    return tps, len(preds) - tps, len(gts) - tps, ious


def _load_upright_raw(raw_dir: Path, max_n: Optional[int]) -> List[dict]:
    items = []
    for img in sorted(raw_dir.glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gts = load_yolo_aabb_label(img.with_suffix(".txt"), w, h)
        # Skip empty negatives here; counted separately
        if not gts:
            items.append({"path": img, "gts": [], "bucket": "empty", "kind": "empty_raw"})
        else:
            # Most raw labels are upright; tag by geometry
            ori = gts[0].orientation
            bucket = "upright" if ori != "sideways" else "sideways"
            items.append({"path": img, "gts": gts, "bucket": bucket, "kind": "raw"})
        if max_n and len(items) >= max_n:
            break
    return items


def _load_split(split_dir: Path, split: str) -> List[dict]:
    """Load vial_dataset split_70_10_20/{images,labels}/{train,val,test}."""
    name = "val" if split in ("val", "validate") else split
    img_dir = split_dir / "images" / name
    lbl_dir = split_dir / "labels" / name
    if not img_dir.is_dir():
        return []
    items = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        lbl = lbl_dir / f"{img.stem}.txt"
        gts = load_yolo_aabb_label(lbl, w, h) if lbl.exists() else []
        if not gts:
            items.append({"path": img, "gts": [], "bucket": "empty", "kind": f"split_{name}_empty"})
        else:
            ori = gts[0].orientation
            bucket = "upright" if ori != "sideways" else "sideways"
            items.append({"path": img, "gts": gts, "bucket": bucket, "kind": f"split_{name}"})
    return items


def _load_sideways_synth(synth_dir: Path) -> List[dict]:
    import numpy as np

    man_path = synth_dir / "manifest.json"
    entries = []
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        entries = man.get("samples") or man.get("entries") or []
    if not entries:
        for jp in sorted(synth_dir.glob("*.json")):
            if jp.name == "manifest.json":
                continue
            entries.append(json.loads(jp.read_text(encoding="utf-8")))

    items = []
    for entry in entries:
        kind = entry.get("kind", "")
        color = entry.get("color")
        if not color:
            continue
        img = synth_dir / color
        if not img.exists():
            continue
        depth = None
        depth_name = entry.get("depth")
        if depth_name:
            dpath = synth_dir / depth_name
            if dpath.exists():
                depth = np.load(dpath).astype(np.float32)
        boxes = entry.get("boxes") or []
        note = entry.get("note") or ""
        angle = parse_angle_from_note(note)
        gts: List[OBBDet] = []
        for b in boxes:
            x, y, bw, bh = [float(v) for v in b[:4]]
            if "sideways" in kind:
                ang = 0.0 if angle is None else angle
                if angle is None and kind.startswith("hard_sideways"):
                    ang = -20.0
                gts.append(aabb_xywh_to_obb(x, y, bw, bh, angle_deg=ang, source="gt_synth"))
            elif kind == "upright":
                gts.append(aabb_xywh_to_obb(x, y, bw, bh, angle_deg=None, source="gt_synth"))
        if kind == "upright" and not gts:
            m = re.search(r"box=\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", note)
            if m:
                x, y, bw, bh = map(int, m.groups())
                gts.append(aabb_xywh_to_obb(x, y, bw, bh, angle_deg=None, source="gt_synth"))

        if "sideways" in kind:
            bucket = "sideways"
        elif kind in {"upright", "topdown"}:
            bucket = "upright"
            if kind == "topdown" and not gts and boxes:
                for b in boxes:
                    x, y, bw, bh = [float(v) for v in b[:4]]
                    gts.append(aabb_xywh_to_obb(x, y, bw, bh, angle_deg=None, source="gt_synth"))
        elif kind == "real_wide":
            # Real vials — positives for a general detector (hard-neg only for sideways-only race).
            bucket = "upright"
            if not gts and boxes:
                for b in boxes:
                    x, y, bw, bh = [float(v) for v in b[:4]]
                    gts.append(aabb_xywh_to_obb(x, y, bw, bh, angle_deg=None, source="gt_synth"))
            if not gts:
                m = re.search(r"gt=\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", note)
                if m:
                    x, y, bw, bh = map(int, m.groups())
                    gts.append(aabb_xywh_to_obb(x, y, bw, bh, angle_deg=None, source="gt_synth"))
        elif kind in {"empty", "hard_negative"}:
            bucket = "empty"
            gts = []
        else:
            bucket = "other"
        items.append(
            {
                "path": img,
                "gts": gts,
                "bucket": bucket,
                "kind": kind,
                "depth": depth,
                "intrinsics": entry.get("intrinsics"),
            }
        )
    return items


def _bucket_stats() -> dict:
    return {
        "images": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "ious": [],
        "image_hits": 0,
        "aabb_ious": [],
        "aabb_hits": 0,
    }


def evaluate(
    detector: VialOBBDetector,
    items: List[dict],
    *,
    iou_thr: float,
) -> dict:
    buckets: Dict[str, dict] = {
        "upright": _bucket_stats(),
        "upright_synth_strict": _bucket_stats(),  # kind == upright only (contract)
        "upright_dataset": _bucket_stats(),  # raw / split stills
        "sideways": _bucket_stats(),
        "empty": _bucket_stats(),
        "all_pos": _bucket_stats(),
    }
    times_ms: List[float] = []

    for it in items:
        bgr = cv2.imread(str(it["path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        gts: List[OBBDet] = it["gts"]
        bucket = it["bucket"]
        kind = str(it.get("kind") or "")
        t0 = time.perf_counter()
        preds, stats = detector.predict(
            bgr,
            depth_mm=it.get("depth"),
            intrinsics=it.get("intrinsics"),
        )
        times_ms.append(stats.inference_ms or (time.perf_counter() - t0) * 1000.0)

        tp, fp, fn, ious = _match(preds, gts, iou_thr=iou_thr, use_obb=True)
        _, _, _, aabb_ious = _match(preds, gts, iou_thr=iou_thr, use_obb=False)

        targets = [bucket]
        if bucket in {"upright", "sideways"}:
            targets.append("all_pos")
        if kind == "upright" and gts:
            targets.append("upright_synth_strict")
        if bucket == "upright" and gts and (
            kind.startswith("raw") or kind.startswith("split_") or kind == "raw"
        ):
            targets.append("upright_dataset")
        for key in targets:
            if key not in buckets:
                continue
            st = buckets[key]
            st["images"] += 1
            st["tp"] += tp
            st["fp"] += fp
            st["fn"] += fn
            st["ious"].extend(ious)
            if gts:
                if tp >= 1:
                    st["image_hits"] += 1
                if aabb_ious:
                    st["aabb_ious"].extend(aabb_ious)
                    st["aabb_hits"] += 1 if max(aabb_ious) >= iou_thr or tp >= 1 else 0
            elif key == "empty" and preds:
                # Only empty bucket treats any prediction as an FP image.
                st["image_hits"] += 1

    def _summarize(st: dict, *, empty: bool = False) -> dict:
        images = st["images"]
        if empty:
            fp_images = st["image_hits"]
            return {
                "images": images,
                "fp_images": fp_images,
                "fp_rate": (fp_images / images) if images else 0.0,
                "det_fp": st["fp"],
            }
        tp, fp, fn = st["tp"], st["fp"], st["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        mean_iou = sum(st["ious"]) / len(st["ious"]) if st["ious"] else 0.0
        img_rec = st["image_hits"] / images if images else 0.0
        mean_aabb = sum(st["aabb_ious"]) / len(st["aabb_ious"]) if st["aabb_ious"] else 0.0
        return {
            "images": images,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "image_recall@IoU": round(img_rec, 4),
            "mean_obb_iou": round(mean_iou, 4),
            "mean_aabb_iou": round(mean_aabb, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return {
        "iou_threshold": iou_thr,
        "backend": detector.backend,
        "model_path": str(detector.model_path or ""),
        "mean_inference_ms": round(sum(times_ms) / len(times_ms), 2) if times_ms else 0.0,
        "upright": _summarize(buckets["upright"]),
        "upright_synth_strict": _summarize(buckets["upright_synth_strict"]),
        "upright_dataset": _summarize(buckets["upright_dataset"]),
        "sideways": _summarize(buckets["sideways"]),
        "empty_negatives": _summarize(buckets["empty"], empty=True),
        "all_positives": _summarize(buckets["all_pos"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera-root", type=Path, default=CAMERA)
    ap.add_argument("--model", type=str, default="", help="OBB weights path (optional)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=IOU_THR)
    ap.add_argument("--max-raw", type=int, default=0, help="0 = all raw images")
    ap.add_argument("--no-synth", action="store_true")
    ap.add_argument("--no-raw", action="store_true")
    ap.add_argument(
        "--split",
        type=str,
        default="",
        help="If set (train|val|test), evaluate vial_dataset/split_70_10_20 that split",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "eval_results.json",
    )
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()

    detector = VialOBBDetector(
        args.model or None,
        camera_root=args.camera_root,
        confidence_threshold=args.conf,
        allow_classical_fallback=True,
        allow_aabb_refine=True,
        device=args.device or None,
    )

    items: List[dict] = []
    if args.split:
        split_dir = args.camera_root / "vial_dataset" / "split_70_10_20"
        items.extend(_load_split(split_dir, args.split))
    else:
        if not args.no_raw:
            max_n = args.max_raw if args.max_raw > 0 else None
            items.extend(_load_upright_raw(args.camera_root / "vial_dataset" / "raw", max_n))
        if not args.no_synth:
            items.extend(_load_sideways_synth(args.camera_root / "sideways_race" / "synth"))

    print(
        f"Evaluating backend={detector.backend} model={detector.model_path} "
        f"n_images={len(items)} conf={args.conf} split={args.split or 'raw+synth'}"
    )
    metrics = evaluate(detector, items, iou_thr=args.iou)
    metrics["n_images"] = len(items)
    metrics["split"] = args.split or "raw+synth"
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.out}")

    # Concise upright vs sideways headline
    u = metrics["upright"]
    us = metrics.get("upright_synth_strict") or {}
    ud = metrics.get("upright_dataset") or {}
    s = metrics["sideways"]
    print(
        f"\nHEADLINE  upright_bucket={u.get('image_recall@IoU')} "
        f"upright_synth_strict={us.get('image_recall@IoU')} "
        f"upright_dataset={ud.get('image_recall@IoU')} | "
        f"sideways={s.get('image_recall@IoU')} "
        f"mean_obb_iou={s.get('mean_obb_iou')} | "
        f"empty_fp={metrics['empty_negatives'].get('fp_rate')}"
    )


if __name__ == "__main__":
    main()
