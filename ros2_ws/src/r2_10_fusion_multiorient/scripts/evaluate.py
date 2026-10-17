#!/usr/bin/env python3
"""Evaluate multi-orientation fusion: upright vs sideways precision/recall/IoU.

Upright set: vial_dataset split test (YOLO labels).
Sideways set: sideways_race/synth positives (manifest boxes) + negatives for FP.

Usage (no ROS required):
  cd E:\\Desktop\\Jessica\\Camera\\ros2_ws\\src\\r2_10_fusion_multiorient
  python scripts/evaluate.py
  python scripts/evaluate.py --max-upright 40 --no-tta   # ablation
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

from r2_10_fusion_multiorient.fusion import FusionConfig, FusionVialDetector  # noqa: E402
from r2_10_fusion_multiorient.types import Det, box_iou, xywh_to_det  # noqa: E402
from r2_10_fusion_multiorient.yolo_tta import default_camera_root  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yolo_label(path: Path, img_w: int, img_h: int) -> List[Det]:
    if not path.is_file():
        return []
    dets: List[Det] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = parts[:5]
        cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
        w, h = bw * img_w, bh * img_h
        x1 = (cx * img_w) - w / 2
        y1 = (cy * img_h) - h / 2
        dets.append(Det(x1, y1, x1 + w, y1 + h, 1.0, source="gt", orientation="upright"))
    return dets


def match_image(
    preds: Sequence[Det],
    gts: Sequence[Det],
    iou_thr: float,
) -> Tuple[int, int, int, List[float], float]:
    """Per-image: tp/fp/fn, matched ious, best_iou over GTs."""
    if not gts and not preds:
        return 0, 0, 0, [], 1.0
    if not gts:
        return 0, len(preds), 0, [], 0.0
    if not preds:
        return 0, 0, len(gts), [], 0.0
    used = set()
    ious: List[float] = []
    tp = 0
    for p in sorted(preds, key=lambda d: d.conf, reverse=True):
        best_i, best = -1, 0.0
        for gi, g in enumerate(gts):
            if gi in used:
                continue
            v = box_iou(p, g)
            if v > best:
                best, best_i = v, gi
        if best_i >= 0 and best >= iou_thr:
            used.add(best_i)
            tp += 1
            ious.append(best)
    fp = len(preds) - tp
    fn = len(gts) - tp
    # best IoU any pred↔any gt (for mean_iou on positives)
    best_any = 0.0
    for g in gts:
        for p in preds:
            best_any = max(best_any, box_iou(p, g))
    return tp, fp, fn, ious, best_any


def upright_pairs(split_dir: Path, subset: str) -> List[Tuple[Path, Path]]:
    name = "val" if subset in ("val", "validate") else subset
    img_dir = split_dir / "images" / name
    lbl_dir = split_dir / "labels" / name
    if not img_dir.is_dir():
        return []
    pairs = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        pairs.append((img, lbl_dir / f"{img.stem}.txt"))
    return pairs


def load_sideways_samples(synth_dir: Path) -> List[dict]:
    manifest = synth_dir / "manifest.json"
    if not manifest.is_file():
        return []
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return list(data.get("samples", []))


def summarize_bucket(
    name: str,
    tp: int,
    fp: int,
    fn: int,
    ious: List[float],
    best_ious: List[float],
    pos_images: int,
    empty_images: int,
    empty_fp: int,
    seconds: float,
    n_images: int,
) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    # Image-level recall: fraction of positive images with any GT matched
    img_recall = (
        sum(1 for v in best_ious if v >= 0.5) / pos_images if pos_images else 0.0
    )
    mean_iou = float(np.mean(best_ious)) if best_ious else 0.0
    mean_matched = float(np.mean(ious)) if ious else 0.0
    return {
        "bucket": name,
        "n_images": n_images,
        "n_positives": pos_images,
        "n_empties_or_negatives": empty_images,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "image_recall_iou0.5": round(img_recall, 4),
        "mean_best_iou": round(mean_iou, 4),
        "mean_matched_iou": round(mean_matched, 4),
        "empty_fp": empty_fp,
        "fps": round(n_images / seconds, 3) if seconds > 0 else 0.0,
        "seconds": round(seconds, 3),
    }


def eval_upright(
    detector: FusionVialDetector,
    pairs: List[Tuple[Path, Path]],
    iou_thr: float,
    max_images: Optional[int],
) -> dict:
    if max_images is not None:
        pairs = pairs[:max_images]
    tp = fp = fn = 0
    ious: List[float] = []
    best_ious: List[float] = []
    pos_images = empty_images = empty_fp = 0
    t0 = time.perf_counter()
    for img_path, lbl_path in pairs:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gts = load_yolo_label(lbl_path, w, h)
        preds = detector.detect(bgr, None)
        tpi, fpi, fni, matched, best = match_image(preds, gts, iou_thr)
        tp += tpi
        fp += fpi
        fn += fni
        ious.extend(matched)
        if gts:
            pos_images += 1
            best_ious.append(best)
        else:
            empty_images += 1
            if preds:
                empty_fp += 1
    seconds = time.perf_counter() - t0
    return summarize_bucket(
        "upright",
        tp,
        fp,
        fn,
        ious,
        best_ious,
        pos_images,
        empty_images,
        empty_fp,
        seconds,
        len(pairs),
    )


def _is_sideways_positive(sample: dict) -> bool:
    kind = str(sample.get("kind", ""))
    return sample.get("role") == "positive" or kind.startswith("sideways") or kind.startswith(
        "hard_sideways"
    )


def _is_true_negative_for_multiorient(sample: dict) -> bool:
    """Empty / hard clutter only — never top-down or upright vials (ERROR_AUDIT)."""
    kind = str(sample.get("kind", ""))
    if kind.startswith("topdown") or kind.startswith("upright"):
        return False
    if kind.startswith("real_wide"):
        return False  # incomplete GT; exclude from headline metrics
    return kind.startswith("empty") or kind.startswith("hard_negative")


def _is_synth_pose_control(sample: dict) -> Optional[str]:
    """Return control bucket name for synth vials that must not score as negatives."""
    kind = str(sample.get("kind", ""))
    if kind.startswith("topdown"):
        return "topdown"
    if kind.startswith("upright"):
        return "upright_synth"
    return None


def _note_boxes(sample: dict) -> List[List[int]]:
    import re

    match = re.search(r"(?:box|gt)=\[([0-9, ]+)\]", str(sample.get("note", "")), re.I)
    if not match:
        return []
    values = [int(v.strip()) for v in match.group(1).split(",")]
    return [values] if len(values) == 4 else []


def _sample_gt_boxes(sample: dict, default_orient: str) -> List[Det]:
    boxes = list(sample.get("boxes") or [])
    if not boxes:
        boxes = _note_boxes(sample)
    return [
        xywh_to_det(int(b[0]), int(b[1]), int(b[2]), int(b[3]), orientation=default_orient)
        for b in boxes
    ]


def eval_pose_controls(
    detector: FusionVialDetector,
    synth_dir: Path,
    iou_thr: float,
) -> Dict[str, dict]:
    """Score topdown_* and upright_* synth as vial positives (not negatives)."""
    samples = load_sideways_samples(synth_dir)
    buckets: Dict[str, List[dict]] = {"topdown": [], "upright_synth": []}
    for s in samples:
        name = _is_synth_pose_control(s)
        if name:
            buckets[name].append(s)

    out: Dict[str, dict] = {}
    for name, scored in buckets.items():
        tp = fp = fn = 0
        ious: List[float] = []
        best_ious: List[float] = []
        orient = "topdown" if name == "topdown" else "upright"
        t0 = time.perf_counter()
        for s in scored:
            color = synth_dir / s["color"]
            depth_path = synth_dir / s["depth"]
            bgr = cv2.imread(str(color), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            depth = None
            if depth_path.is_file():
                depth = np.load(str(depth_path)).astype(np.float32)
            gts = _sample_gt_boxes(s, orient)
            preds = detector.detect(bgr, depth)
            tpi, fpi, fni, matched, best = match_image(preds, gts, iou_thr)
            tp += tpi
            fp += fpi
            fn += fni
            ious.extend(matched)
            best_ious.append(best)
        seconds = time.perf_counter() - t0
        out[name] = summarize_bucket(
            name,
            tp,
            fp,
            fn,
            ious,
            best_ious,
            len(scored),
            0,
            0,
            seconds,
            len(scored),
        )
        out[name]["note"] = (
            "Synth pose control scored as vial positives; not used as empty negatives."
        )
    return out


def eval_sideways(
    detector: FusionVialDetector,
    synth_dir: Path,
    iou_thr: float,
    max_images: Optional[int],
    positives_only: bool = False,
) -> dict:
    samples = load_sideways_samples(synth_dir)
    scored = []
    skipped_real_wide = 0
    for s in samples:
        kind = str(s.get("kind", ""))
        if kind.startswith("real_wide"):
            skipped_real_wide += 1
            continue
        if _is_sideways_positive(s):
            scored.append(s)
        elif (not positives_only) and _is_true_negative_for_multiorient(s):
            scored.append(s)
        # topdown/upright handled separately as pose controls — not here
    if max_images is not None:
        scored = scored[:max_images]
    tp = fp = fn = 0
    ious: List[float] = []
    best_ious: List[float] = []
    pos_images = empty_images = empty_fp = 0
    t0 = time.perf_counter()
    for s in scored:
        color = synth_dir / s["color"]
        depth_path = synth_dir / s["depth"]
        bgr = cv2.imread(str(color), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        depth = None
        if depth_path.is_file():
            depth = np.load(str(depth_path)).astype(np.float32)
        gts = [
            xywh_to_det(int(b[0]), int(b[1]), int(b[2]), int(b[3]), orientation="sideways")
            for b in s.get("boxes") or []
        ]
        preds = detector.detect(bgr, depth)
        if _is_sideways_positive(s):
            # Score sideways detections primarily; allow any orient that matches GT.
            tpi, fpi, fni, matched, best = match_image(preds, gts, iou_thr)
            tp += tpi
            fp += fpi
            fn += fni
            ious.extend(matched)
            pos_images += 1
            best_ious.append(best)
        else:
            empty_images += 1
            # True empty/clutter: any detection is FP
            if preds:
                empty_fp += 1
                fp += len(preds)
    seconds = time.perf_counter() - t0
    result = summarize_bucket(
        "sideways",
        tp,
        fp,
        fn,
        ious,
        best_ious,
        pos_images,
        empty_images,
        empty_fp,
        seconds,
        len(scored),
    )
    result["excluded_real_wide_incomplete_gt"] = skipped_real_wide
    result["note"] = (
        "Headline sideways excludes topdown_*/upright_* (reported separately) "
        "and real_wide_* (incomplete GT)."
    )
    return result


def print_table(
    upright: dict,
    sideways: dict,
    overall: dict,
    pose_controls: Optional[Dict[str, dict]] = None,
) -> None:
    def row(d: dict) -> str:
        return (
            f"{d['bucket']:<14}  P={d['precision']:.3f}  R={d['recall']:.3f}  "
            f"imgR@0.5={d['image_recall_iou0.5']:.3f}  "
            f"mIoU={d['mean_best_iou']:.3f}  "
            f"emptyFP={d['empty_fp']}  n={d['n_images']}  fps={d['fps']:.1f}"
        )

    print("\n=== r2_10_fusion_multiorient evaluate ===")
    print(row(upright))
    print(row(sideways))
    if pose_controls:
        for name in ("topdown", "upright_synth"):
            if name in pose_controls:
                print(row(pose_controls[name]))
    print(row(overall))
    if sideways.get("excluded_real_wide_incomplete_gt"):
        print(
            f"(excluded real_wide_* incomplete GT: "
            f"{sideways['excluded_real_wide_incomplete_gt']} frames)"
        )
    print()


def pass_fail(upright: dict, sideways: dict) -> dict:
    """Compare against Round-1 failure bar.

    Round-1 agent_01 YOLO smoke: upright recall ~0.90, no sideways support.
    Sideways race baseline_depthcamera: recall 0.375.
    Pass if upright image-recall ≥ 0.85 AND sideways image-recall ≥ 0.80
    with sideways empty FP rate ≤ 0.15.
    """
    u_ok = upright["image_recall_iou0.5"] >= 0.85 and upright["precision"] >= 0.70
    neg = max(1, sideways["n_empties_or_negatives"])
    fp_rate = sideways["empty_fp"] / neg
    s_ok = sideways["image_recall_iou0.5"] >= 0.80 and fp_rate <= 0.15
    return {
        "upright_pass": bool(u_ok),
        "sideways_pass": bool(s_ok),
        "overall_pass": bool(u_ok and s_ok),
        "criteria": {
            "upright_image_recall_ge": 0.85,
            "upright_precision_ge": 0.70,
            "sideways_image_recall_ge": 0.80,
            "sideways_neg_fp_rate_le": 0.15,
        },
        "round1_reference": {
            "agent_01_yolo_upright_recall_smoke": 0.90,
            "agent_01_yolo_sideways": "unsupported / poor",
            "sideways_race_baseline_depthcamera_recall": 0.375,
            "agent_10_ensemble_upright_test_recall": 0.907,
            "agent_10_ensemble_sideways": "not evaluated; upright-biased shape gates",
        },
    }


def main() -> int:
    cam = default_camera_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split",
        type=Path,
        default=cam / "vial_dataset" / "split_70_10_20_latest2",
    )
    ap.add_argument("--subset", default="test", choices=("train", "val", "test"))
    ap.add_argument(
        "--synth",
        type=Path,
        default=cam / "sideways_race" / "synth",
    )
    ap.add_argument("--camera-root", type=Path, default=cam)
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-upright", type=int, default=None)
    ap.add_argument("--max-sideways", type=int, default=None)
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--no-classical", action="store_true")
    ap.add_argument("--yolo-only", action="store_true")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "models" / "eval_multiorient.json",
    )
    args = ap.parse_args()

    cfg = FusionConfig(
        camera_root=str(args.camera_root),
        model_path=args.model or None,
        yolo_conf=args.conf,
        use_yolo_tta=not args.no_tta,
        use_classical_upright=not (args.no_classical or args.yolo_only),
        use_classical_sideways=not (args.no_classical or args.yolo_only),
        enable_rgb_classical_sideways=False,
        use_cap_refine=not args.yolo_only,
    )
    detector = FusionVialDetector(cfg)
    print(
        f"backend={detector.backend} model={detector.model_path} "
        f"tta={cfg.use_yolo_tta} classical_u={cfg.use_classical_upright} "
        f"classical_s={cfg.use_classical_sideways} "
        f"rgb_sideways={cfg.enable_rgb_classical_sideways} "
        f"extra_yolo_gate={cfg.extra_yolo_conf_after_method_i} "
        f"yolo_floor={cfg.empty_scene_yolo_floor}"
    )

    pairs = upright_pairs(args.split, args.subset)
    if not pairs:
        print(f"WARN: no upright pairs under {args.split}", file=sys.stderr)
    upright = eval_upright(detector, pairs, args.iou, args.max_upright)

    pose_controls: Dict[str, dict] = {}
    if not (args.synth / "manifest.json").is_file():
        print(f"WARN: no synth manifest at {args.synth}", file=sys.stderr)
        sideways = summarize_bucket("sideways", 0, 0, 0, [], [], 0, 0, 0, 0.0, 0)
    else:
        sideways = eval_sideways(detector, args.synth, args.iou, args.max_sideways)
        pose_controls = eval_pose_controls(detector, args.synth, args.iou)

    # Headline overall: real upright holdout + sideways pos/neg only
    # (pose controls reported separately; real_wide excluded).
    n_pos = upright["n_positives"] + sideways["n_positives"]
    overall = summarize_bucket(
        "overall",
        upright["tp"] + sideways["tp"],
        upright["fp"] + sideways["fp"],
        upright["fn"] + sideways["fn"],
        [],
        [],
        n_pos,
        upright["n_empties_or_negatives"] + sideways["n_empties_or_negatives"],
        upright["empty_fp"] + sideways["empty_fp"],
        upright["seconds"] + sideways["seconds"],
        upright["n_images"] + sideways["n_images"],
    )
    if n_pos:
        overall["image_recall_iou0.5"] = round(
            (
                upright["image_recall_iou0.5"] * upright["n_positives"]
                + sideways["image_recall_iou0.5"] * sideways["n_positives"]
            )
            / n_pos,
            4,
        )
        overall["mean_best_iou"] = round(
            (
                upright["mean_best_iou"] * upright["n_positives"]
                + sideways["mean_best_iou"] * sideways["n_positives"]
            )
            / n_pos,
            4,
        )

    verdict = pass_fail(upright, sideways)
    print_table(upright, sideways, overall, pose_controls)
    print(
        f"PASS/FAIL vs Round-1 bar: upright={verdict['upright_pass']} "
        f"sideways={verdict['sideways_pass']} overall={verdict['overall_pass']}"
    )

    payload = {
        "agent": "r2_10_fusion_multiorient",
        "backend": detector.backend,
        "model_path": str(detector.model_path or ""),
        "config": {
            "yolo_conf": cfg.yolo_conf,
            "use_yolo_tta": cfg.use_yolo_tta,
            "use_classical_upright": cfg.use_classical_upright,
            "use_classical_sideways": cfg.use_classical_sideways,
            "enable_rgb_classical_sideways": cfg.enable_rgb_classical_sideways,
            "extra_yolo_conf_after_method_i": cfg.extra_yolo_conf_after_method_i,
            "empty_scene_yolo_floor": cfg.empty_scene_yolo_floor,
            "empty_method_i_keep_conf": cfg.empty_method_i_keep_conf,
            "use_cap_refine": cfg.use_cap_refine,
        },
        "upright": upright,
        "sideways": sideways,
        "pose_controls": pose_controls,
        "overall": overall,
        "pass_fail": verdict,
        "paths": {
            "split": str(args.split),
            "subset": args.subset,
            "synth": str(args.synth),
        },
        "scoring_notes": {
            "topdown_not_negatives": True,
            "upright_synth_reported_separately": True,
            "real_wide_excluded_incomplete_gt": True,
            "audit": "ERROR_AUDIT.md short-term gates",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0 if verdict["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
