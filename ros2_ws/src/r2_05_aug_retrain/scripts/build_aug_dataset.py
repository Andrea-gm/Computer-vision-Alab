#!/usr/bin/env python3
"""Audit upright vs sideways label coverage and build a rotation-heavy YOLO set.

Pipeline:
  1. Audit vial_dataset/raw box aspect ratios (h/w).
  2. Seed train from split_70_10_20 train + val copies.
  3. Offline aug: random ±180° rotate, perspective warp, scale jitter
     (labels transformed via corner AABB).
  4. Inject sideways_race/synth positives (sideways_left/right + hard_*) into
     train; hold out a fraction for sideways eval.
  5. Write upright holdout = original split test; sideways holdout = synth holdout.

Usage:
  python scripts/build_aug_dataset.py
  python scripts/build_aug_dataset.py --augs-per-image 3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

PKG = Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIDEWAYS_KINDS = {
    "sideways_left",
    "sideways_right",
    "hard_sideways_left",
    "hard_sideways_right",
}
UPRIGHT_KINDS = {"upright", "topdown", "real_wide"}


def camera_root() -> Path:
    for parent in [PKG, *PKG.parents]:
        if (parent / "vial_dataset").is_dir():
            return parent
    return PKG.parents[3]


def parse_yolo_labels(path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = parts
        boxes.append((int(float(cls)), float(cx), float(cy), float(w), float(h)))
    return boxes


def write_yolo_labels(path: Path, boxes: Sequence[Tuple[int, float, float, float, float]]) -> None:
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes]
    path.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")


def yolo_to_corners(
    boxes: Sequence[Tuple[int, float, float, float, float]], w: int, h: int
) -> List[Tuple[int, np.ndarray]]:
    out = []
    for cls, cx, cy, bw, bh in boxes:
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h
        corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
        )
        out.append((cls, corners))
    return out


def _clamp_yolo_box(
    cls: int, cx: float, cy: float, bw: float, bh: float
) -> Tuple[int, float, float, float, float] | None:
    """Clamp YOLO box so all coords stay in [0,1] (Ultralytics rejects OOB)."""
    bw = float(np.clip(bw, 1e-4, 1.0))
    bh = float(np.clip(bh, 1e-4, 1.0))
    cx = float(np.clip(cx, bw / 2.0, 1.0 - bw / 2.0))
    cy = float(np.clip(cy, bh / 2.0, 1.0 - bh / 2.0))
    if bw < 0.01 or bh < 0.01:
        return None
    return (cls, cx, cy, bw, bh)


def corners_to_yolo(
    items: Sequence[Tuple[int, np.ndarray]], w: int, h: int, min_area_frac: float = 1e-4
) -> List[Tuple[int, float, float, float, float]]:
    boxes = []
    for cls, corners in items:
        xs = corners[:, 0]
        ys = corners[:, 1]
        x1 = float(np.clip(xs.min(), 0, w - 1))
        y1 = float(np.clip(ys.min(), 0, h - 1))
        x2 = float(np.clip(xs.max(), 0, w - 1))
        y2 = float(np.clip(ys.max(), 0, h - 1))
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 1 or bh <= 1:
            continue
        if (bw * bh) / float(w * h) < min_area_frac:
            continue
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        clamped = _clamp_yolo_box(cls, cx, cy, bw / w, bh / h)
        if clamped is not None:
            boxes.append(clamped)
    return boxes


def apply_affine_to_corners(corners: np.ndarray, M: np.ndarray) -> np.ndarray:
    ones = np.ones((corners.shape[0], 1), dtype=np.float32)
    hom = np.hstack([corners, ones])
    return (M @ hom.T).T


def rotate_image_and_boxes(
    bgr: np.ndarray,
    boxes: Sequence[Tuple[int, float, float, float, float]],
    angle_deg: float,
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    h, w = bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += (nw / 2.0) - center[0]
    M[1, 2] += (nh / 2.0) - center[1]
    warped = cv2.warpAffine(
        bgr, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    items = []
    for cls, corners in yolo_to_corners(boxes, w, h):
        items.append((cls, apply_affine_to_corners(corners, M)))
    return warped, corners_to_yolo(items, nw, nh)


def perspective_image_and_boxes(
    bgr: np.ndarray,
    boxes: Sequence[Tuple[int, float, float, float, float]],
    strength: float,
    rng: random.Random,
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    h, w = bgr.shape[:2]
    jitter = strength * min(w, h)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = src.copy()
    for i in range(4):
        dst[i, 0] += rng.uniform(-jitter, jitter)
        dst[i, 1] += rng.uniform(-jitter, jitter)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    items = []
    for cls, corners in yolo_to_corners(boxes, w, h):
        ones = np.ones((4, 1), dtype=np.float32)
        hom = np.hstack([corners, ones])
        proj = (M @ hom.T).T
        proj = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-6)
        items.append((cls, proj.astype(np.float32)))
    return warped, corners_to_yolo(items, w, h)


def scale_image_and_boxes(
    bgr: np.ndarray,
    boxes: Sequence[Tuple[int, float, float, float, float]],
    scale: float,
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    h, w = bgr.shape[:2]
    nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # Letterbox back to original size so YOLO imgsz stays consistent.
    canvas = np.zeros_like(bgr)
    y0 = max(0, (h - nh) // 2)
    x0 = max(0, (w - nw) // 2)
    y1 = min(h, y0 + nh)
    x1 = min(w, x0 + nw)
    crop = resized[0 : y1 - y0, 0 : x1 - x0]
    canvas[y0:y1, x0:x1] = crop
    new_boxes = []
    for cls, cx, cy, bw, bh in boxes:
        # Map normalized box through resize+pad.
        abs_cx = cx * w * scale + x0
        abs_cy = cy * h * scale + y0
        abs_bw = bw * w * scale
        abs_bh = bh * h * scale
        ncx = abs_cx / w
        ncy = abs_cy / h
        nbw = abs_bw / w
        nbh = abs_bh / h
        clamped = _clamp_yolo_box(cls, ncx, ncy, nbw, nbh)
        if clamped is not None:
            new_boxes.append(clamped)
    return canvas, new_boxes


def augment_once(
    bgr: np.ndarray,
    boxes: Sequence[Tuple[int, float, float, float, float]],
    rng: random.Random,
    *,
    force_sideways: bool = False,
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]], dict]:
    meta = {}
    if force_sideways:
        # Bias toward lying-down orientations (≈±90°) with jitter.
        base = 90.0 if rng.random() < 0.5 else -90.0
        angle = base + rng.uniform(-25.0, 25.0)
    else:
        angle = rng.uniform(-180.0, 180.0)
    meta["angle"] = angle
    img, boxes = rotate_image_and_boxes(bgr, boxes, angle)

    if rng.random() < 0.7:
        strength = rng.uniform(0.02, 0.08)
        meta["perspective"] = strength
        img, boxes = perspective_image_and_boxes(img, boxes, strength, rng)

    if rng.random() < 0.8:
        scale = rng.uniform(0.75, 1.25)
        meta["scale"] = scale
        # If rotate expanded canvas, scale within current frame.
        img, boxes = scale_image_and_boxes(img, boxes, scale)

    # Mild photometric jitter (does not affect boxes).
    if rng.random() < 0.5:
        alpha = rng.uniform(0.85, 1.15)
        beta = rng.uniform(-20, 20)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        meta["photo"] = {"alpha": alpha, "beta": beta}

    return img, boxes, meta


def audit_raw(raw_dir: Path) -> dict:
    aspects = []
    empty = 0
    images = 0
    sideways = upright = square = 0
    for img in sorted(raw_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        images += 1
        boxes = parse_yolo_labels(img.with_suffix(".txt"))
        if not boxes:
            empty += 1
            continue
        for _, _, _, bw, bh in boxes:
            ah = bh / max(bw, 1e-9)
            aspects.append(ah)
            if ah < 0.85:
                sideways += 1
            elif ah > 1.0:
                upright += 1
            else:
                square += 1
    aspects_s = sorted(aspects)
    return {
        "raw_dir": str(raw_dir),
        "images": images,
        "empty_negatives": empty,
        "positive_boxes": len(aspects),
        "sideways_like_h_over_w_lt_0.85": sideways,
        "squareish_0.85_to_1.0": square,
        "upright_like_h_over_w_gt_1.0": upright,
        "aspect_h_over_w_min_med_max": (
            [round(aspects_s[0], 4), round(aspects_s[len(aspects_s) // 2], 4), round(aspects_s[-1], 4)]
            if aspects_s
            else None
        ),
        "verdict": (
            "SEVERE upright bias — sideways recall will collapse without rotation/synth aug"
            if sideways <= max(1, len(aspects) // 50)
            else "mixed orientations present"
        ),
    }


def copy_pair(img_src: Path, lbl_src: Path, img_dst: Path, lbl_dst: Path) -> None:
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_src, img_dst)
    if lbl_src.exists():
        shutil.copy2(lbl_src, lbl_dst)
    else:
        lbl_dst.write_text("", encoding="utf-8")


def save_pair(
    bgr: np.ndarray,
    boxes: Sequence[Tuple[int, float, float, float, float]],
    img_dst: Path,
    lbl_dst: Path,
) -> None:
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_dst), bgr)
    write_yolo_labels(lbl_dst, boxes)


def synth_xywh_to_yolo(
    boxes_xywh: Sequence[Sequence[float]], w: int, h: int
) -> List[Tuple[int, float, float, float, float]]:
    out = []
    for x, y, bw, bh in boxes_xywh:
        cx = (x + bw / 2.0) / w
        cy = (y + bh / 2.0) / h
        out.append((0, cx, cy, bw / w, bh / h))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augs-per-image", type=int, default=3)
    ap.add_argument("--sideways-holdout-frac", type=float, default=0.30)
    ap.add_argument(
        "--out",
        type=Path,
        default=PKG / "data" / "aug_split",
    )
    args = ap.parse_args()
    rng = random.Random(args.seed)
    cam = camera_root()
    raw = cam / "vial_dataset" / "raw"
    split = cam / "vial_dataset" / "split_70_10_20"
    synth_dir = cam / "sideways_race" / "synth"
    manifest_path = synth_dir / "manifest.json"

    audit = audit_raw(raw)
    print("=== LABEL AUDIT (raw) ===")
    print(json.dumps(audit, indent=2))

    if args.out.exists():
        shutil.rmtree(args.out)
    for part in ("train", "val", "holdout_upright", "holdout_sideways"):
        (args.out / "images" / part).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / part).mkdir(parents=True, exist_ok=True)

    stats: Dict[str, int] = {
        "train_seed": 0,
        "train_aug_random": 0,
        "train_aug_sideways_bias": 0,
        "train_synth_sideways": 0,
        "val": 0,
        "holdout_upright": 0,
        "holdout_sideways": 0,
        "aug_dropped_empty_boxes": 0,
    }

    # --- seed train/val from official split ---
    for part, dest in (("train", "train"), ("val", "val")):
        img_dir = split / "images" / part
        lbl_dir = split / "labels" / part
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            dst_img = args.out / "images" / dest / f"seed_{img.name}"
            dst_lbl = args.out / "labels" / dest / f"seed_{img.stem}.txt"
            copy_pair(img, lbl, dst_img, dst_lbl)
            stats["train_seed" if dest == "train" else "val"] += 1

    # --- upright holdout = official test ---
    test_img = split / "images" / "test"
    test_lbl = split / "labels" / "test"
    for img in sorted(test_img.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = test_lbl / f"{img.stem}.txt"
        copy_pair(
            img,
            lbl,
            args.out / "images" / "holdout_upright" / img.name,
            args.out / "labels" / "holdout_upright" / f"{img.stem}.txt",
        )
        stats["holdout_upright"] += 1

    # --- offline rotation / perspective / scale augs on train seeds ---
    train_imgs = list((args.out / "images" / "train").glob("seed_*"))
    for img_path in train_imgs:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        boxes = parse_yolo_labels(
            args.out / "labels" / "train" / f"{img_path.stem}.txt"
        )
        if not boxes:
            continue
        for k in range(args.augs_per_image):
            force_side = k == 0  # at least one sideways-biased aug per positive
            aug_img, aug_boxes, meta = augment_once(
                bgr, boxes, rng, force_sideways=force_side
            )
            if not aug_boxes:
                stats["aug_dropped_empty_boxes"] += 1
                continue
            stem = f"aug_{img_path.stem}_{k}_{'side' if force_side else 'rand'}"
            save_pair(
                aug_img,
                aug_boxes,
                args.out / "images" / "train" / f"{stem}.jpg",
                args.out / "labels" / "train" / f"{stem}.txt",
            )
            if force_side:
                stats["train_aug_sideways_bias"] += 1
            else:
                stats["train_aug_random"] += 1

    # --- synth sideways inject + holdout ---
    synth_sideways = []
    synth_other_train = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for sample in manifest["samples"]:
            color = synth_dir / sample["color"]
            if not color.exists():
                continue
            kind = sample.get("kind", "")
            role = sample.get("role", "")
            w = int(sample.get("image_width", 640))
            h = int(sample.get("image_height", 480))
            yolo_boxes = (
                synth_xywh_to_yolo(sample.get("boxes") or [], w, h)
                if role == "positive"
                else []
            )
            entry = {
                "id": sample["id"],
                "kind": kind,
                "color": color,
                "boxes": yolo_boxes,
                "role": role,
            }
            if kind in SIDEWAYS_KINDS and role == "positive":
                synth_sideways.append(entry)
            elif kind in UPRIGHT_KINDS or role == "negative":
                # hard_negative / empty → useful negatives; upright/topdown → train mix
                synth_other_train.append(entry)

        rng.shuffle(synth_sideways)
        n_hold = max(1, int(round(len(synth_sideways) * args.sideways_holdout_frac)))
        hold = synth_sideways[:n_hold]
        train_syn = synth_sideways[n_hold:]

        for entry in hold:
            stem = f"synth_{entry['id']}"
            bgr = cv2.imread(str(entry["color"]), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            save_pair(
                bgr,
                entry["boxes"],
                args.out / "images" / "holdout_sideways" / f"{stem}.png",
                args.out / "labels" / "holdout_sideways" / f"{stem}.txt",
            )
            stats["holdout_sideways"] += 1

        for entry in train_syn + synth_other_train:
            stem = f"synth_{entry['id']}"
            bgr = cv2.imread(str(entry["color"]), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            save_pair(
                bgr,
                entry["boxes"],
                args.out / "images" / "train" / f"{stem}.png",
                args.out / "labels" / "train" / f"{stem}.txt",
            )
            if entry["kind"] in SIDEWAYS_KINDS:
                stats["train_synth_sideways"] += 1
            # Extra sideways-biased augs on synth train positives
            if entry["boxes"] and entry["kind"] in SIDEWAYS_KINDS:
                for k in range(2):
                    aug_img, aug_boxes, _ = augment_once(
                        bgr, entry["boxes"], rng, force_sideways=True
                    )
                    if not aug_boxes:
                        stats["aug_dropped_empty_boxes"] += 1
                        continue
                    save_pair(
                        aug_img,
                        aug_boxes,
                        args.out / "images" / "train" / f"{stem}_aug{k}.jpg",
                        args.out / "labels" / "train" / f"{stem}_aug{k}.txt",
                    )
                    stats["train_aug_sideways_bias"] += 1
    else:
        print(f"WARNING: missing synth manifest at {manifest_path}", file=sys.stderr)

    # Post-aug orientation audit on train set
    train_side = train_up = train_sq = 0
    for lbl in (args.out / "labels" / "train").glob("*.txt"):
        for _, _, _, bw, bh in parse_yolo_labels(lbl):
            ah = bh / max(bw, 1e-9)
            if ah < 0.85:
                train_side += 1
            elif ah > 1.0:
                train_up += 1
            else:
                train_sq += 1

    data_yaml = {
        "path": str(args.out.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "vial"},
    }
    yaml_path = args.out / "data.yaml"
    # Minimal YAML without requiring PyYAML for write
    yaml_text = (
        f"path: {data_yaml['path']}\n"
        f"train: {data_yaml['train']}\n"
        f"val: {data_yaml['val']}\n"
        "names:\n"
        "  0: vial\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")

    report = {
        "audit_raw": audit,
        "build_stats": stats,
        "train_box_orientation_after_aug": {
            "sideways_like": train_side,
            "squareish": train_sq,
            "upright_like": train_up,
            "sideways_fraction": round(train_side / max(1, train_side + train_sq + train_up), 4),
        },
        "data_yaml": str(yaml_path),
        "holdout_upright_dir": str(args.out / "images" / "holdout_upright"),
        "holdout_sideways_dir": str(args.out / "images" / "holdout_sideways"),
        "seed": args.seed,
        "augs_per_image": args.augs_per_image,
    }
    report_path = PKG / "results" / "aug_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== BUILD STATS ===")
    print(json.dumps(report, indent=2))
    print(f"Wrote {yaml_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
