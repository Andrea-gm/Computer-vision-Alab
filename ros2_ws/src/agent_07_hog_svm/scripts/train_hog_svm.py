#!/usr/bin/env python3
"""Train HOG + Linear SVM vial detector with hard-negative mining.

Labeling assumptions (documented for competition):
- Positives: YOLO class-0 boxes from vial_dataset (clear plastic vial + white cap).
- Empty .txt files are full-image negatives (no vial).
- Additional negatives: random crops from positive images with IoU < 0.1 vs GT.
- Hard negatives: sliding-window false positives (score > 0) on train images
  that do not overlap GT (IoU < 0.3), then one retrain pass.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.svm import LinearSVC

# Allow running without installing the package.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from agent_07_hog_svm.hog_detector import (  # noqa: E402
    DEFAULT_WIN_SIZE,
    HogSvmDetector,
    compute_hog_features,
    crop_resize_gray,
    iou_xyxy,
    make_hog,
    parse_yolo_label,
)


def _find_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def list_split_pairs(split_root: Path, split: str) -> List[Tuple[Path, Path]]:
    images_dir = split_root / "images" / split
    labels_dir = split_root / "labels" / split
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    pairs: List[Tuple[Path, Path]] = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        label = labels_dir / f"{img.stem}.txt"
        pairs.append((img, label))
    return pairs


def sample_negative_boxes(
    img_w: int,
    img_h: int,
    gt: Sequence[Sequence[int]],
    win_size: Tuple[int, int],
    n: int,
    rng: random.Random,
    max_iou: float = 0.1,
) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    # Prefer GT-like scales but allow some diversity.
    base_w, base_h = win_size
    attempts = 0
    while len(boxes) < n and attempts < n * 40:
        attempts += 1
        scale = rng.uniform(0.7, 1.8)
        bw = int(round(base_w * scale))
        bh = int(round(base_h * scale))
        if bw >= img_w or bh >= img_h:
            continue
        x1 = rng.randint(0, img_w - bw)
        y1 = rng.randint(0, img_h - bh)
        cand = (x1, y1, x1 + bw, y1 + bh)
        if any(iou_xyxy(cand, g) > max_iou for g in gt):
            continue
        boxes.append(cand)
    return boxes


def extract_features_for_split(
    pairs: List[Tuple[Path, Path]],
    hog,
    win_size: Tuple[int, int],
    neg_per_image: int,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    labels: List[int] = []

    for img_path, label_path in pairs:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gt = parse_yolo_label(label_path, w, h)

        for box in gt:
            patch = crop_resize_gray(bgr, box, win_size)
            if patch is None:
                continue
            # Light augmentation: horizontal flip (vials roughly symmetric).
            for variant in (patch, cv2.flip(patch, 1)):
                feats.append(compute_hog_features(hog, variant))
                labels.append(1)

        n_neg = neg_per_image if gt else max(neg_per_image, 6)
        for box in sample_negative_boxes(w, h, gt, win_size, n_neg, rng):
            patch = crop_resize_gray(bgr, box, win_size)
            if patch is None:
                continue
            feats.append(compute_hog_features(hog, patch))
            labels.append(0)

    if not feats:
        raise RuntimeError("No training features extracted — check dataset path")
    X = np.stack(feats, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)
    return X, y


def hard_negative_mine(
    pairs: List[Tuple[Path, Path]],
    detector: HogSvmDetector,
    hog,
    win_size: Tuple[int, int],
    max_per_image: int,
    score_thresh: float,
) -> Tuple[np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    # Temporary lower threshold / coarser step for mining speed.
    old_step = detector.step_size
    old_thr = detector.confidence_threshold
    old_scale = detector.scale_factor
    detector.step_size = max(12, old_step)
    detector.confidence_threshold = score_thresh
    detector.scale_factor = 1.35

    for img_path, label_path in pairs:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gt = parse_yolo_label(label_path, w, h)
        dets = detector.detect(bgr)
        hard = []
        for d in dets:
            if any(iou_xyxy(d.xyxy, g) >= 0.3 for g in gt):
                continue
            hard.append(d)
            if len(hard) >= max_per_image:
                break
        for d in hard:
            patch = crop_resize_gray(bgr, d.xyxy, win_size)
            if patch is None:
                continue
            feats.append(compute_hog_features(hog, patch))

    detector.step_size = old_step
    detector.confidence_threshold = old_thr
    detector.scale_factor = old_scale

    if not feats:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    X = np.stack(feats, axis=0).astype(np.float32)
    y = np.zeros((X.shape[0],), dtype=np.int32)
    return X, y


def train_linear_svc(X: np.ndarray, y: np.ndarray, C: float, seed: int) -> Tuple[np.ndarray, float]:
    # Balance classes — empty negatives are scarce relative to random crops.
    clf = LinearSVC(
        C=C,
        class_weight="balanced",
        max_iter=20000,
        dual="auto",
        random_state=seed,
    )
    clf.fit(X, y)
    weights = np.asarray(clf.coef_, dtype=np.float32).reshape(-1)
    bias = float(clf.intercept_[0])
    return weights, bias


def evaluate_simple(
    pairs: List[Tuple[Path, Path]],
    detector: HogSvmDetector,
    iou_thresh: float = 0.5,
) -> dict:
    tp = fp = fn = 0
    empty_fp = 0
    n_pos = n_empty = 0
    for img_path, label_path in pairs:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gt = parse_yolo_label(label_path, w, h)
        dets = detector.detect(bgr)
        if not gt:
            n_empty += 1
            empty_fp += len(dets)
            fp += len(dets)
            continue
        n_pos += 1
        matched = set()
        for d in dets:
            best_i, best_iou = -1, 0.0
            for i, g in enumerate(gt):
                if i in matched:
                    continue
                v = iou_xyxy(d.xyxy, g)
                if v > best_iou:
                    best_iou, best_i = v, i
            if best_iou >= iou_thresh and best_i >= 0:
                tp += 1
                matched.add(best_i)
            else:
                fp += 1
        fn += len(gt) - len(matched)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "n_pos_images": n_pos,
        "n_empty_images": n_empty,
        "empty_fp": empty_fp,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train HOG+SVM vial detector")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"E:\Desktop\Jessica\robotics\vial_dataset\split_70_10_20"),
        help="YOLO split root with images/{train,val,test} and labels/...",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_PKG_ROOT / "models" / "hog_svm_vial.json",
    )
    parser.add_argument("--win-w", type=int, default=DEFAULT_WIN_SIZE[0])
    parser.add_argument("--win-h", type=int, default=DEFAULT_WIN_SIZE[1])
    parser.add_argument("--neg-per-image", type=int, default=8)
    parser.add_argument("--C", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-neg-rounds", type=int, default=1)
    parser.add_argument("--hard-neg-max-per-image", type=int, default=8)
    parser.add_argument("--score-thresh", type=float, default=0.0)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    win_size = (args.win_w, args.win_h)
    hog = make_hog(win_size)

    train_pairs = list_split_pairs(args.dataset, "train")
    val_name = "val" if (args.dataset / "images" / "val").exists() else "validate"
    val_pairs = list_split_pairs(args.dataset, val_name) if (args.dataset / "images" / val_name).exists() else []
    test_pairs = (
        list_split_pairs(args.dataset, "test")
        if (args.dataset / "images" / "test").exists()
        else []
    )

    print(f"[train] dataset={args.dataset}")
    print(f"[train] train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")
    print(f"[train] win_size={win_size}")

    t0 = time.time()
    X, y = extract_features_for_split(
        train_pairs, hog, win_size, args.neg_per_image, rng
    )
    print(f"[train] initial features: X={X.shape} pos={(y == 1).sum()} neg={(y == 0).sum()}")

    weights, bias = train_linear_svc(X, y, args.C, args.seed)
    detector = HogSvmDetector(
        confidence_threshold=args.score_thresh,
        scale_factor=1.25,
        step_size=8,
        max_image_side=480,
        nms_iou=0.3,
    )
    detector.set_linear_svm(weights, bias, win_size)

    for round_i in range(args.hard_neg_rounds):
        print(f"[train] hard-negative mining round {round_i + 1}...")
        Xh, yh = hard_negative_mine(
            train_pairs,
            detector,
            hog,
            win_size,
            args.hard_neg_max_per_image,
            score_thresh=0.0,
        )
        print(f"[train] mined hard negatives: {Xh.shape[0]}")
        if Xh.shape[0] == 0:
            break
        X = np.concatenate([X, Xh], axis=0)
        y = np.concatenate([y, yh], axis=0)
        weights, bias = train_linear_svc(X, y, args.C, args.seed)
        detector.set_linear_svm(weights, bias, win_size)

    meta = {
        "algorithm": "HOG + LinearSVC",
        "dataset": str(args.dataset),
        "win_size": list(win_size),
        "C": args.C,
        "seed": args.seed,
        "n_train_images": len(train_pairs),
        "n_features": int(X.shape[1]),
        "n_samples_final": int(X.shape[0]),
        "pos_samples": int((y == 1).sum()),
        "neg_samples": int((y == 0).sum()),
        "hard_neg_rounds": args.hard_neg_rounds,
        "labeling_assumptions": [
            "Positives = YOLO class-0 full-vial boxes (lid through cup).",
            "Empty label files = background-only negatives.",
            "Random crops with IoU<0.1 vs GT used as easy negatives.",
            "Hard negatives = high-scoring windows with IoU<0.3 vs GT.",
        ],
        "train_seconds": time.time() - t0,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    detector.meta = meta
    detector.save(args.out, meta=meta)
    print(f"[train] wrote {args.out}")
    print(f"[train] wrote {args.out.with_suffix('.yml')}")

    if not args.skip_eval:
        for name, pairs in (("val", val_pairs), ("test", test_pairs)):
            if not pairs:
                continue
            metrics = evaluate_simple(pairs, detector, iou_thresh=0.5)
            print(f"[eval:{name}] {json.dumps(metrics)}")
            meta[f"metrics_{name}"] = metrics
        detector.save(args.out, meta=meta)

    metrics_path = args.out.with_name("train_metrics.json")
    metrics_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[train] metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
