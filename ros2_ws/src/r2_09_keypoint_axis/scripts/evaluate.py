#!/usr/bin/env python3
"""Evaluate cap/base keypoint axis detector — upright vs sideways.

Reports AABB IoU@0.5 plus axis angle error (FIRE metric: box must follow vial axis).

Usage (no ROS required):
  cd E:\\Desktop\\Jessica\\Camera\\ros2_ws\\src\\r2_09_keypoint_axis
  python scripts/evaluate.py
  python scripts/evaluate.py --synth-only
  python scripts/evaluate.py --split test
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2_09_keypoint_axis.axis_skeleton import far_end_from_box  # noqa: E402
from r2_09_keypoint_axis.cap_finder import localize_cap_in_box  # noqa: E402
from r2_09_keypoint_axis.detector import (  # noqa: E402
    AxisParams,
    DEFAULT_CAMERA,
    KeypointAxisDetector,
)
from r2_09_keypoint_axis.types import Det, angle_diff_deg, box_iou, list_images  # noqa: E402
from r2_09_keypoint_axis.weak_labels import load_yolo_label  # noqa: E402

SYNTH_DIR = DEFAULT_CAMERA / "sideways_race" / "synth"
SPLIT_DIR = DEFAULT_CAMERA / "vial_dataset" / "split_70_10_20"
NOTE_BOX_RE = re.compile(
    r"box\s*=\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", re.I
)
NOTE_GT_RE = re.compile(
    r"gt\s*=\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", re.I
)
NOTE_ANGLE_RE = re.compile(r"angle\s*=\s*([+-]?\d+(?:\.\d+)?)", re.I)


def _gt_axis_from_box(bgr: np.ndarray, xyxy: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    """Return (angle_deg, cap_x, cap_y) using white-cap localization inside GT."""
    cap = localize_cap_in_box(bgr, xyxy)
    if cap is None:
        x1, y1, x2, y2 = xyxy
        # Heuristic: upright → top is cap; sideways → left/right by aspect.
        if (y2 - y1) >= (x2 - x1):
            cx, cy = 0.5 * (x1 + x2), y1 + 0.15 * (y2 - y1)
        else:
            cx, cy = x1 + 0.15 * (x2 - x1), 0.5 * (y1 + y2)
    else:
        cx, cy = cap.cx, cap.cy
    base = far_end_from_box((cx, cy), xyxy)
    ang = math.degrees(math.atan2(base[1] - cy, base[0] - cx))
    return float(ang), float(cx), float(cy)


def _parse_upright_note(note: str) -> Optional[Tuple[int, int, int, int]]:
    m = NOTE_BOX_RE.search(note or "")
    if not m:
        return None
    return tuple(int(v) for v in m.groups())  # type: ignore[return-value]


def load_synth_samples(synth_dir: Path) -> List[Dict[str, Any]]:
    manifest_path = synth_dir / "manifest.json"
    samples: List[Dict[str, Any]] = []
    if manifest_path.is_file():
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw = man.get("samples", man if isinstance(man, list) else [])
    else:
        raw = []
        for jf in sorted(synth_dir.glob("*.json")):
            if jf.name == "manifest.json":
                continue
            raw.append(json.loads(jf.read_text(encoding="utf-8")))

    for s in raw:
        kind = str(s.get("kind", ""))
        sid = str(s.get("id", ""))
        color = s.get("color") or f"{sid}_color.png"
        depth = s.get("depth") or f"{sid}_depth.npy"
        # Per-file note may hold upright GT.
        note = str(s.get("note", ""))
        jpath = synth_dir / f"{sid}.json"
        if jpath.is_file():
            try:
                jd = json.loads(jpath.read_text(encoding="utf-8"))
                note = str(jd.get("note", note))
            except (OSError, json.JSONDecodeError):
                pass

        boxes = list(s.get("boxes") or [])
        group = "other"
        if kind.startswith("sideways") or kind.startswith("hard_sideways"):
            group = "sideways"
        elif kind == "upright":
            group = "upright"
            ub = _parse_upright_note(note)
            if ub is not None:
                boxes = [list(ub)]
        elif kind == "real_wide":
            # Real vial stills used as sideways-race negatives; for axis eval they are positives.
            group = "real_wide"
            mgt = NOTE_GT_RE.search(note)
            if mgt:
                boxes = [[int(v) for v in mgt.groups()]]
        elif kind in ("empty", "hard_negative", "topdown"):
            group = "negative"

        ang_note = None
        m = NOTE_ANGLE_RE.search(note)
        if m:
            ang_note = float(m.group(1))

        samples.append(
            {
                "id": sid,
                "kind": kind,
                "group": group,
                "boxes_xywh": boxes,
                "color": synth_dir / color,
                "depth": synth_dir / depth,
                "note_angle": ang_note,
                "role": s.get("role", ""),
            }
        )
    return samples


def match_best(
    preds: Sequence[Det], gts: Sequence[Det], *, iou_thr: float
) -> Tuple[int, int, int, List[float], List[float]]:
    """Returns tp, fp, fn, matched_ious, matched_angle_errs."""
    if not gts and not preds:
        return 0, 0, 0, [], []
    if not gts:
        return 0, len(preds), 0, [], []
    if not preds:
        return 0, 0, len(gts), [], []
    used = set()
    tps = 0
    ious: List[float] = []
    angs: List[float] = []
    for p in sorted(preds, key=lambda d: d.conf, reverse=True):
        best_i, best_iou = -1, 0.0
        for gi, g in enumerate(gts):
            if gi in used:
                continue
            iou = box_iou(p, g)
            if iou > best_iou:
                best_iou, best_i = iou, gi
        if best_i >= 0 and best_iou >= iou_thr:
            used.add(best_i)
            tps += 1
            ious.append(best_iou)
            angs.append(angle_diff_deg(p.angle_deg, gts[best_i].angle_deg))
    return tps, len(preds) - tps, len(gts) - len(used), ious, angs


def evaluate_synth(
    detector: KeypointAxisDetector,
    synth_dir: Path,
    *,
    iou_thr: float,
    angle_thr_deg: float,
) -> Dict[str, Any]:
    samples = load_synth_samples(synth_dir)
    buckets = {
        "upright": _empty_bucket(),
        "sideways": _empty_bucket(),
        "real_wide": _empty_bucket(),
        "negative": _empty_bucket(),
        "all_pos": _empty_bucket(),
    }

    per: List[Dict[str, Any]] = []
    for s in samples:
        bgr = cv2.imread(str(s["color"]), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        depth = None
        if s["depth"].is_file():
            try:
                depth = np.load(str(s["depth"]))
            except Exception:  # noqa: BLE001
                depth = None
        t0 = time.perf_counter()
        preds, stats = detector.predict(bgr, depth_mm=depth)
        dt = (time.perf_counter() - t0) * 1000.0

        gts: List[Det] = []
        for xywh in s["boxes_xywh"]:
            x, y, w, h = [float(v) for v in xywh]
            xyxy = (x, y, x + w, y + h)
            ang, _, _ = _gt_axis_from_box(bgr, xyxy)
            if s.get("note_angle") is not None and s["group"] == "sideways":
                # Synth hard tilt is signed degrees from horizontal.
                tilt = float(s["note_angle"])
                if "right" in s["kind"]:
                    ang = 180.0 + tilt
                else:
                    ang = tilt
            gts.append(
                Det(
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + h,
                    conf=1.0,
                    angle_deg=ang,
                    pose=s["group"],
                )
            )

        group = s["group"]
        if group == "negative":
            fp = 1 if preds else 0
            buckets["negative"]["n"] += 1
            buckets["negative"]["fp"] += fp
            buckets["negative"]["times_ms"].append(dt)
            per.append({"id": s["id"], "group": group, "fp": fp, "n_pred": len(preds)})
            continue

        if group == "other" or not gts:
            continue

        tp, fp, fn, ious, angs = match_best(preds, gts, iou_thr=iou_thr)
        # FIRE gate: matched boxes must follow GT vial axis (undirected, mod 180).
        axis_ok = bool(angs) and all(a <= angle_thr_deg for a in angs)

        rec = {
            "id": s["id"],
            "group": group,
            "kind": s["kind"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "ious": ious,
            "angle_errs": angs,
            "axis_ok": axis_ok,
            "pred_angles": [d.angle_deg for d in preds],
            "gt_angles": [g.angle_deg for g in gts],
            "infer_ms": dt,
            "backend": stats.backend,
        }
        per.append(rec)
        for key in (group, "all_pos"):
            if key not in buckets:
                continue
            b = buckets[key]
            b["n"] += 1
            b["tp"] += tp
            b["fp"] += fp
            b["fn"] += fn
            b["ious"].extend(ious)
            b["angle_errs"].extend(angs)
            b["axis_ok"] += int(axis_ok and tp > 0)
            b["times_ms"].append(dt)

    summary = {k: _finalize_bucket(v) for k, v in buckets.items()}
    # FIRE rule: fired if axis does not follow vial on positives with a detection.
    upright_fire = summary["upright"]["axis_follow_rate"] < 0.75
    sideways_fire = summary["sideways"]["axis_follow_rate"] < 0.75
    return {
        "dataset": str(synth_dir),
        "iou_thr": iou_thr,
        "angle_thr_deg": angle_thr_deg,
        "summary": summary,
        "fire": {
            "upright_orientation_fail": upright_fire,
            "sideways_orientation_fail": sideways_fire,
            "fired": upright_fire or sideways_fire,
        },
        "per_sample": per,
    }


def evaluate_split(
    detector: KeypointAxisDetector,
    split_dir: Path,
    split: str,
    *,
    iou_thr: float,
    angle_thr_deg: float,
) -> Dict[str, Any]:
    name = "val" if split in ("val", "validate") else split
    img_dir = split_dir / "images" / name
    lbl_dir = split_dir / "labels" / name
    pairs = []
    for img in list_images(img_dir):
        pairs.append((img, lbl_dir / f"{img.stem}.txt"))

    upright = _empty_bucket()
    sideways = _empty_bucket()
    overall = _empty_bucket()
    per: List[Dict[str, Any]] = []

    for img_path, lbl_path in pairs:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gts_raw = load_yolo_label(lbl_path, w, h)
        gts: List[Det] = []
        for g in gts_raw:
            ang, _, _ = _gt_axis_from_box(bgr, g.as_xyxy())
            g.angle_deg = ang
            bw = g.x2 - g.x1
            bh = g.y2 - g.y1
            g.pose = "upright" if bh >= bw * 0.95 else "sideways"
            gts.append(g)

        t0 = time.perf_counter()
        preds, stats = detector.predict(bgr)
        dt = (time.perf_counter() - t0) * 1000.0

        if not gts:
            overall["n"] += 1
            overall["fp"] += len(preds)
            overall["times_ms"].append(dt)
            continue

        tp, fp, fn, ious, angs = match_best(preds, gts, iou_thr=iou_thr)
        # Bucket by majority GT pose.
        u = sum(1 for g in gts if g.pose == "upright")
        group = "upright" if u >= len(gts) / 2 else "sideways"
        axis_ok = bool(angs) and all(a <= angle_thr_deg for a in angs)
        for b in (overall, upright if group == "upright" else sideways):
            b["n"] += 1
            b["tp"] += tp
            b["fp"] += fp
            b["fn"] += fn
            b["ious"].extend(ious)
            b["angle_errs"].extend(angs)
            b["axis_ok"] += int(axis_ok and tp > 0)
            b["times_ms"].append(dt)
        per.append(
            {
                "id": img_path.stem,
                "group": group,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "ious": ious,
                "angle_errs": angs,
                "axis_ok": axis_ok,
                "infer_ms": dt,
            }
        )

    return {
        "dataset": str(split_dir / "images" / name),
        "iou_thr": iou_thr,
        "angle_thr_deg": angle_thr_deg,
        "summary": {
            "upright": _finalize_bucket(upright),
            "sideways": _finalize_bucket(sideways),
            "overall": _finalize_bucket(overall),
        },
        "n_images": len(pairs),
        "per_sample": per[:50],
        "backend": detector.backend,
    }


def _empty_bucket() -> Dict[str, Any]:
    return {
        "n": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "ious": [],
        "angle_errs": [],
        "axis_ok": 0,
        "times_ms": [],
    }


def _finalize_bucket(b: Dict[str, Any]) -> Dict[str, Any]:
    n = int(b["n"])
    tp, fp, fn = int(b["tp"]), int(b["fp"]), int(b["fn"])
    denom_p = tp + fp
    denom_r = tp + fn
    ious = b["ious"]
    angs = b["angle_errs"]
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": (tp / denom_p) if denom_p else 0.0,
        "recall": (tp / denom_r) if denom_r else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_angle_err_deg": float(np.mean(angs)) if angs else 0.0,
        "median_angle_err_deg": float(np.median(angs)) if angs else 0.0,
        "axis_follow_rate": (b["axis_ok"] / n) if n else 0.0,
        "mean_ms": float(np.mean(b["times_ms"])) if b["times_ms"] else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synth", type=Path, default=SYNTH_DIR)
    ap.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--synth-only", action="store_true")
    ap.add_argument("--split-only", action="store_true")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--angle-thr", type=float, default=25.0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--no-yolo-seed", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "eval_latest.json",
    )
    args = ap.parse_args()

    params = AxisParams(
        confidence_threshold=args.conf,
        use_yolo_seed=not args.no_yolo_seed,
    )
    detector = KeypointAxisDetector(params)
    report: Dict[str, Any] = {
        "method": "cap_base_keypoint_axis",
        "backend": detector.backend,
        "params": {
            "conf": args.conf,
            "use_yolo_seed": params.use_yolo_seed,
            "angle_thr_deg": args.angle_thr,
            "iou_thr": args.iou,
        },
    }

    if not args.split_only:
        print(f"Evaluating synth: {args.synth}")
        report["synth"] = evaluate_synth(
            detector, args.synth, iou_thr=args.iou, angle_thr_deg=args.angle_thr
        )
        s = report["synth"]["summary"]
        print(
            "  upright:  "
            f"recall={s['upright']['recall']:.3f} meanIoU={s['upright']['mean_iou']:.3f} "
            f"angErr={s['upright']['mean_angle_err_deg']:.1f} "
            f"axisFollow={s['upright']['axis_follow_rate']:.3f}"
        )
        print(
            "  sideways: "
            f"recall={s['sideways']['recall']:.3f} meanIoU={s['sideways']['mean_iou']:.3f} "
            f"angErr={s['sideways']['mean_angle_err_deg']:.1f} "
            f"axisFollow={s['sideways']['axis_follow_rate']:.3f}"
        )
        print(
            "  real_wide: "
            f"recall={s['real_wide']['recall']:.3f} meanIoU={s['real_wide']['mean_iou']:.3f} "
            f"angErr={s['real_wide']['mean_angle_err_deg']:.1f} "
            f"axisFollow={s['real_wide']['axis_follow_rate']:.3f}"
        )
        print(
            "  negative: "
            f"fp={s['negative']['fp']}/{s['negative']['n']} "
            f"fp_rate={s['negative']['fp']/max(s['negative']['n'],1):.3f}"
        )
        print(f"  FIRE={report['synth']['fire']}")

    if not args.synth_only:
        print(f"Evaluating split {args.split}: {args.split_dir}")
        report["split"] = evaluate_split(
            detector,
            args.split_dir,
            args.split,
            iou_thr=args.iou,
            angle_thr_deg=args.angle_thr,
        )
        s = report["split"]["summary"]
        print(
            "  overall:  "
            f"P={s['overall']['precision']:.3f} R={s['overall']['recall']:.3f} "
            f"IoU={s['overall']['mean_iou']:.3f} angErr={s['overall']['mean_angle_err_deg']:.1f} "
            f"axisFollow={s['overall']['axis_follow_rate']:.3f}"
        )
        print(
            "  upright:  "
            f"R={s['upright']['recall']:.3f} IoU={s['upright']['mean_iou']:.3f} "
            f"angErr={s['upright']['mean_angle_err_deg']:.1f} "
            f"axisFollow={s['upright']['axis_follow_rate']:.3f} n={s['upright']['n']}"
        )
        print(
            "  sideways: "
            f"R={s['sideways']['recall']:.3f} IoU={s['sideways']['mean_iou']:.3f} "
            f"angErr={s['sideways']['mean_angle_err_deg']:.1f} "
            f"axisFollow={s['sideways']['axis_follow_rate']:.3f} n={s['sideways']['n']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
