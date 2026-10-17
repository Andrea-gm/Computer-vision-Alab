#!/usr/bin/env python3
"""Train the Stage-3 logistic verifier from YOLO-format vial labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running without install: add package parent to path.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from agent_10_ensemble.features import FEATURE_DIM, Verifier, extract_features  # noqa: E402
from agent_10_ensemble.proposals import propose_cap, propose_edges  # noqa: E402
from agent_10_ensemble.types import Detection  # noqa: E402
from agent_10_ensemble.nms import box_iou  # noqa: E402


def yolo_to_xyxy(line: str, w: int, h: int) -> tuple[float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    _, xc, yc, bw, bh = map(float, parts[:5])
    x1 = (xc - bw / 2.0) * w
    y1 = (yc - bh / 2.0) * h
    x2 = (xc + bw / 2.0) * w
    y2 = (yc + bh / 2.0) * h
    return x1, y1, x2, y2


def load_split(images_dir: Path, labels_dir: Path):
    pairs = []
    for img_path in sorted(images_dir.glob("*.*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        gts = []
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                # Need image size — deferred until read
                gts.append(line)
        pairs.append((img_path, gts))
    return pairs


def random_neg_box(w: int, h: int, rng: np.random.Generator) -> Detection:
    bw = int(rng.integers(max(12, w // 12), max(13, w // 2)))
    bh = int(rng.integers(max(12, h // 12), max(13, h // 2)))
    x1 = int(rng.integers(0, max(1, w - bw)))
    y1 = int(rng.integers(0, max(1, h - bh)))
    return Detection(float(x1), float(y1), float(x1 + bw), float(y1 + bh), 0.2, source="neg")


def build_dataset(split_root: Path, max_per_image: int = 8, seed: int = 42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    images_dir = split_root / "images" / "train"
    labels_dir = split_root / "labels" / "train"
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)
    pairs = load_split(images_dir, labels_dir)
    for img_path, gt_lines in pairs:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gts = []
        for line in gt_lines:
            box = yolo_to_xyxy(line, w, h)
            if box:
                gts.append(box)

        # Positives from GT
        for box in gts:
            det = Detection(*box, score=0.9, source="gt")
            X.append(extract_features(bgr, det))
            y.append(1)
            # Light jitter positives
            for _ in range(2):
                jx = float(rng.normal(0, 0.03 * (box[2] - box[0])))
                jy = float(rng.normal(0, 0.03 * (box[3] - box[1])))
                js = float(rng.uniform(0.92, 1.08))
                cx = 0.5 * (box[0] + box[2]) + jx
                cy = 0.5 * (box[1] + box[3]) + jy
                bw = (box[2] - box[0]) * js
                bh = (box[3] - box[1]) * js
                det_j = Detection(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, 0.8, source="gt_jit")
                X.append(extract_features(bgr, det_j))
                y.append(1)

        # Proposal-based: high-IoU → pos, low-IoU → hard neg
        props = propose_cap(bgr, max_proposals=max_per_image) + propose_edges(
            bgr, max_proposals=max_per_image
        )
        for det in props:
            if not gts:
                X.append(extract_features(bgr, det))
                y.append(0)
                continue
            best = max(box_iou(det.xyxy, g) for g in gts)
            if best >= 0.5:
                X.append(extract_features(bgr, det))
                y.append(1)
            elif best <= 0.15:
                X.append(extract_features(bgr, det))
                y.append(0)

        # Random background negatives
        n_rand = 3 if gts else 5
        for _ in range(n_rand):
            det = random_neg_box(w, h, rng)
            if gts and max(box_iou(det.xyxy, g) for g in gts) > 0.2:
                continue
            X.append(extract_features(bgr, det))
            y.append(0)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split",
        default=r"E:\Desktop\Jessica\robotics\vial_dataset\split_70_10_20_latest2",
        help="YOLO split root with images/train + labels/train",
    )
    ap.add_argument(
        "--out",
        default=str(_PKG_ROOT / "models" / "verifier.pkl"),
        help="Output pickle path",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    print(f"Building features from {args.split} ...")
    X, y = build_dataset(Path(args.split), seed=args.seed)
    print(f"samples={len(y)} positives={int(y.sum())} negatives={int((y == 0).sum())} dim={FEATURE_DIM}")
    if len(y) < 20 or y.sum() == 0 or (y == 0).sum() == 0:
        print("ERROR: insufficient class balance", file=sys.stderr)
        return 2

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    C=1.0,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print(classification_report(yte, pred, digits=3))

    out = Path(args.out)
    Verifier().save(out, clf, meta={"split": str(args.split), "n": int(len(y)), "pos": int(y.sum())})
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
