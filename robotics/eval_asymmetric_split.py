"""Refresh 70/20/10 (seed 42) and score YOLO IoU≥0.5 under asymmetric confs.

Does NOT wipe vial_dataset/raw — only rebuilds split_70_10_20 from current raw.
Dataset path defaults to the main Camera tree so worktrees never junction/delete it.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
# Prefer main Camera dataset (worktrees must not own/junction vial_dataset).
MAIN_CAMERA = Path(r"E:\Desktop\Jessica\Camera")
DEFAULT_DATASET = MAIN_CAMERA / "vial_dataset"
DEFAULT_WEIGHTS = MAIN_CAMERA / "runs" / "vial-live" / "weights" / "best.pt"
FALLBACK_WEIGHTS = MAIN_CAMERA / "runs" / "bon-winner" / "weights" / "best.pt"

sys.path.insert(0, str(ROOT))
import depthcamera as dc  # noqa: E402


def refresh_split(dataset: Path, seed: int = 42) -> dict:
    raw = dataset / "raw"
    out = dataset / "split_70_10_20"
    pairs = []
    for img in sorted(raw.glob("*.jpg")):
        txt = img.with_suffix(".txt")
        if not txt.exists():
            continue
        has_box = bool(txt.read_text(encoding="utf-8").strip())
        pairs.append((img.name, txt.name, has_box))
    if len(pairs) < 3:
        raise SystemExit(f"Not enough labeled pairs in {raw}: {len(pairs)}")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_test = max(1, round(n * 0.20))
    n_val = max(1, round(n * 0.10))
    n_train = n - n_test - n_val
    if n_train < 1:
        raise SystemExit(f"Not enough images after split: {n}")

    test = pairs[:n_test]
    val = pairs[n_test : n_test + n_val]
    train = pairs[n_test + n_val :]

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    manifest = {
        "seed": seed,
        "counts": {
            "train": len(train),
            "validate": len(val),
            "test": len(test),
            "total": n,
        },
        "train": [p[0] for p in train],
        "validate": [p[0] for p in val],
        "test": [p[0] for p in test],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def materialize(split: str, items) -> None:
        idir = out / "images" / split
        ldir = out / "labels" / split
        idir.mkdir(parents=True)
        ldir.mkdir(parents=True)
        for img_name, txt_name, _ in items:
            shutil.copy2(raw / img_name, idir / img_name)
            shutil.copy2(raw / txt_name, ldir / Path(txt_name).name)

    materialize("train", train)
    materialize("val", val)
    materialize("test", test)
    yaml_path = out / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: vial",
                "",
            ]
        ),
        encoding="utf-8",
    )
    pos = sum(1 for _, _, p in pairs if p)
    neg = sum(1 for _, _, p in pairs if not p)
    manifest["positives"] = pos
    manifest["empty_negatives"] = neg
    manifest["yaml"] = str(yaml_path)
    return manifest


def yolo_xyxy_to_xywh(x1, y1, x2, y2):
    return (float(x1), float(y1), float(x2 - x1), float(y2 - y1))


def load_gt_boxes(label_path: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, xc, yc, bw, bh = map(float, parts)
        w = bw * width
        h = bh * height
        x = xc * width - w / 2
        y = yc * height - h / 2
        boxes.append((x, y, w, h))
    return boxes


def box_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - overlap
    return 0.0 if union <= 0 else overlap / union


def predict_boxes(model, image, conf: float) -> list[tuple[float, float, float, float]]:
    h, w = image.shape[:2]
    results = model.predict(image, verbose=False, conf=conf)
    out = []
    if not results:
        return out
    for result in results:
        detected = getattr(result, "boxes", None)
        if detected is None:
            continue
        for box in detected:
            xyxy = getattr(box, "xyxy", None)
            if xyxy is None:
                continue
            x1, y1, x2, y2 = xyxy[0].tolist()
            out.append(yolo_xyxy_to_xywh(x1, y1, x2, y2))
        break
    return out


def score_split(model, split_dir: Path, split: str, conf: float, iou_thresh: float = 0.5) -> dict:
    images = sorted((split_dir / "images" / split).glob("*.jpg"))
    correct = 0
    details = []
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            details.append({"image": img_path.name, "ok": False, "reason": "read_fail"})
            continue
        h, w = image.shape[:2]
        gt = load_gt_boxes(split_dir / "labels" / split / f"{img_path.stem}.txt", w, h)
        preds = predict_boxes(model, image, conf)
        if not gt:
            ok = len(preds) == 0
            details.append(
                {
                    "image": img_path.name,
                    "ok": ok,
                    "kind": "negative",
                    "n_pred": len(preds),
                }
            )
        else:
            # Positive: any pred IoU≥thresh with any GT counts as hit; prefer best.
            best = 0.0
            for g in gt:
                for p in preds:
                    best = max(best, box_iou(g, p))
            ok = best >= iou_thresh
            details.append(
                {
                    "image": img_path.name,
                    "ok": ok,
                    "kind": "positive",
                    "best_iou": round(best, 4),
                    "n_pred": len(preds),
                    "n_gt": len(gt),
                }
            )
        if ok:
            correct += 1
    total = len(images)
    return {
        "split": split,
        "conf": conf,
        "total": total,
        "correct": correct,
        "accuracy_pct": round(100.0 * correct / total, 1) if total else 0.0,
        "details": details,
    }


def tune_conf_on_val(
    model, split_dir: Path, candidates: list[float], *, prefer_lower: bool = False
) -> tuple[float, dict]:
    best_conf = candidates[0]
    best = None
    grid = []
    for conf in candidates:
        metrics = score_split(model, split_dir, "val", conf)
        grid.append({"conf": conf, "accuracy_pct": metrics["accuracy_pct"], "correct": metrics["correct"]})
        if best is None or metrics["accuracy_pct"] > best["accuracy_pct"]:
            best = metrics
            best_conf = conf
        elif metrics["accuracy_pct"] == best["accuracy_pct"]:
            # Strong cam: higher conf on ties (precision). Weak cam: lower conf (recall).
            if prefer_lower and conf < best_conf:
                best = metrics
                best_conf = conf
            elif not prefer_lower and conf > best_conf:
                best = metrics
                best_conf = conf
    return best_conf, {"grid": grid, "best_conf": best_conf, "best": best}


def main() -> None:
    parser = argparse.ArgumentParser(description="Asymmetric-gate split eval (BoN #3)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-refresh", action="store_true", help="Keep existing split_70_10_20")
    parser.add_argument("--out", type=Path, default=ROOT / "metrics.json")
    args = parser.parse_args()

    dataset = args.dataset
    if not args.no_refresh:
        manifest = refresh_split(dataset, seed=args.seed)
    else:
        manifest = json.loads((dataset / "split_70_10_20" / "manifest.json").read_text(encoding="utf-8"))

    weights = args.weights
    if not weights.exists():
        if FALLBACK_WEIGHTS.exists():
            weights = FALLBACK_WEIGHTS
        else:
            runs = MAIN_CAMERA / "runs"
            cands = list(runs.rglob("best.pt")) if runs.exists() else []
            if not cands:
                raise SystemExit(f"No weights at {weights}")
            # Prefer vial-live / bon-winner names over newest scratch run.
            preferred = [
                p
                for p in cands
                if "vial-live" in str(p) or "bon-winner" in str(p)
            ]
            weights = max(preferred or cands, key=lambda p: p.stat().st_mtime)

    from ultralytics import YOLO

    model = YOLO(str(weights))
    split_dir = dataset / "split_70_10_20"

    strong = dc.strong_gate_profile(yolo_conf=0.25)
    weak = dc.weak_gate_profile(base_conf=0.25)

    # Learn per-profile conf on validate (color-only IoU race).
    strong_conf, strong_tune = tune_conf_on_val(
        model, split_dir, [0.15, 0.20, 0.25, 0.30, 0.35], prefer_lower=False
    )
    weak_conf, weak_tune = tune_conf_on_val(
        model, split_dir, [0.05, 0.10, 0.12, 0.15, 0.20, 0.25], prefer_lower=True
    )
    # Keep weak ≤ strong conf (asymmetric intent).
    if weak_conf > strong_conf:
        weak_conf = min(weak_conf, strong_conf)

    strong = dc.CameraGateProfile(**{**asdict(strong), "yolo_conf": strong_conf, "name": "strong"})
    weak = dc.CameraGateProfile(**{**asdict(weak), "yolo_conf": weak_conf, "name": "weak"})

    baseline_test = score_split(model, split_dir, "test", 0.25)
    strong_test = score_split(model, split_dir, "test", strong.yolo_conf)
    weak_test = score_split(model, split_dir, "test", weak.yolo_conf)
    # Race metric: report weak-cam (recall-oriented) test accuracy as primary for this strategy,
    # plus strong and baseline for comparison.
    primary = weak_test

    result = {
        "strategy": "asymmetric_per_camera_thresholds",
        "agent": 3,
        "worktree_id": "bon-vid-03-ba9c8430",
        "seed": args.seed,
        "weights": str(weights),
        "dataset": str(dataset),
        "split_manifest": {
            "counts": manifest.get("counts"),
            "positives": manifest.get("positives"),
            "empty_negatives": manifest.get("empty_negatives"),
            "test": manifest.get("test"),
            "validate": manifest.get("validate"),
        },
        "profiles": {
            "cam0_strong": asdict(strong),
            "cam1_weak": asdict(weak),
        },
        "tune": {"strong": strong_tune, "weak": weak_tune},
        "metrics": {
            "test_accuracy_pct_primary_weak_conf": primary["accuracy_pct"],
            "test_accuracy_pct_baseline_conf_0.25": baseline_test["accuracy_pct"],
            "test_accuracy_pct_strong_conf": strong_test["accuracy_pct"],
            "test_accuracy_pct_weak_conf": weak_test["accuracy_pct"],
            "iou_threshold": 0.5,
            "baseline_test": {k: v for k, v in baseline_test.items() if k != "details"},
            "strong_test": {k: v for k, v in strong_test.items() if k != "details"},
            "weak_test": {k: v for k, v in weak_test.items() if k != "details"},
            "baseline_details": baseline_test["details"],
            "weak_details": weak_test["details"],
        },
        "constraints": {
            "locate_pair_preserved": True,
            "no_mask_rcnn_sam": True,
            "distances_independent": True,
            "label": "vial",
            "no_depth_thumbnail": True,
            "no_dataset_wipe": True,
            "no_merge": True,
        },
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Wrote {args.out}")
    print(
        f"PRIMARY test accuracy (weak conf={weak.yolo_conf}): "
        f"{primary['accuracy_pct']}%  ({primary['correct']}/{primary['total']})"
    )


if __name__ == "__main__":
    main()
