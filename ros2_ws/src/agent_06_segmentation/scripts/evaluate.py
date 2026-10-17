#!/usr/bin/env python3
"""Offline eval for agent_06 segmentation.

Metrics:
  - bbox recall / mean IoU @ IoU≥0.5 (always; GT from YOLO det labels or JSON)
  - mask mean IoU when GT polygons exist (YOLO-seg labels, >5 tokens)
  - else *pseudo* mask IoU vs ellipse/rect filled from GT boxes
    (reported separately as ``pseudo_mask_miou`` — not real GT)

Usage (from package dir or Camera/):
  python scripts/evaluate.py
  python scripts/evaluate.py --split test --weights runs/vial-seg/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

CAMERA_ROOT = Path(r"E:\Desktop\Jessica\robotics")
PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from agent_06_segmentation.segmenter import (  # noqa: E402
    VialSegmenter,
    bbox_to_ellipse_mask,
    bbox_to_rect_mask,
    box_iou_xywh,
    mask_iou,
    polygon_mask_from_yolo_line,
    yolo_bbox_line_to_xywh,
)


def load_gt(
    label_path: Path, img_h: int, img_w: int, *, pseudo_shape: str
) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray], bool]:
    """Return boxes, masks, and whether masks came from real polygons."""
    boxes: List[Tuple[int, int, int, int]] = []
    masks: List[np.ndarray] = []
    real_poly = False
    if not label_path.is_file():
        return boxes, masks, False
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return boxes, masks, False
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        parsed = yolo_bbox_line_to_xywh(line, img_w, img_h)
        if parsed is None:
            continue
        _, x, y, w, h = parsed
        boxes.append((x, y, w, h))
        if len(parts) > 5:
            m = polygon_mask_from_yolo_line(line, img_h, img_w)
            if m is not None:
                masks.append(m)
                real_poly = True
                continue
        if pseudo_shape == "rect":
            masks.append(bbox_to_rect_mask(img_h, img_w, x, y, w, h))
        else:
            masks.append(bbox_to_ellipse_mask(img_h, img_w, x, y, w, h))
    return boxes, masks, real_poly


def best_match_iou(
    pred_boxes: List[Tuple[int, int, int, int]],
    gt_boxes: List[Tuple[int, int, int, int]],
) -> List[float]:
    """Per-GT best IoU (greedy, no double-count for ranking simplicity)."""
    used = set()
    scores: List[float] = []
    for g in gt_boxes:
        best, best_j = 0.0, -1
        for j, p in enumerate(pred_boxes):
            if j in used:
                continue
            v = box_iou_xywh(g, p)
            if v > best:
                best, best_j = v, j
        if best_j >= 0 and best > 0:
            used.add(best_j)
        scores.append(best)
    return scores


def best_mask_ious(
    pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]
) -> List[float]:
    used = set()
    scores: List[float] = []
    for g in gt_masks:
        best, best_j = 0.0, -1
        for j, p in enumerate(pred_masks):
            if j in used or p is None:
                continue
            v = mask_iou(g, p)
            if v > best:
                best, best_j = v, j
        if best_j >= 0 and best > 0:
            used.add(best_j)
        scores.append(best)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--images",
        type=Path,
        default=CAMERA_ROOT / "vial_dataset" / "split_70_10_20" / "images",
    )
    ap.add_argument(
        "--labels",
        type=Path,
        default=CAMERA_ROOT / "vial_dataset" / "split_70_10_20" / "labels",
    )
    ap.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    ap.add_argument("--weights", type=str, default="")
    ap.add_argument("--camera-root", type=Path, default=CAMERA_ROOT)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--mask-fallback", type=str, default="ellipse")
    ap.add_argument(
        "--pseudo-gt-shape",
        type=str,
        default="ellipse",
        choices=("ellipse", "rect"),
        help="Shape used when labels are bbox-only (pseudo mask GT)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=CAMERA_ROOT
        / "ros2_ws"
        / "src"
        / "agent_06_segmentation"
        / "eval_results.json",
    )
    ap.add_argument("--max-images", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    img_dir = args.images / args.split
    lab_dir = args.labels / args.split
    if not img_dir.is_dir():
        raise SystemExit(f"Missing images: {img_dir}")

    seg = VialSegmenter(
        model_path=args.weights or None,
        camera_root=args.camera_root,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        mask_fallback=args.mask_fallback,
    )
    seg.load()
    print(f"model={seg.resolved_path} kind={seg.model_kind}")

    images = sorted(
        [
            p
            for p in img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )
    if args.max_images > 0:
        images = images[: args.max_images]

    box_ious_all: List[float] = []
    mask_ious_real: List[float] = []
    mask_ious_pseudo: List[float] = []
    n_pos = 0
    n_empty = 0
    n_fp_empty = 0
    n_real_poly_images = 0
    hits = 0

    for img_path in images:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        lab = lab_dir / f"{img_path.stem}.txt"
        gt_boxes, gt_masks, real_poly = load_gt(
            lab, h, w, pseudo_shape=args.pseudo_gt_shape
        )
        if real_poly:
            n_real_poly_images += 1

        dets = seg.predict(bgr)
        pred_boxes = [(d.x, d.y, d.w, d.h) for d in dets]
        pred_masks = [d.mask for d in dets if d.mask is not None]

        if not gt_boxes:
            n_empty += 1
            if pred_boxes:
                n_fp_empty += 1
            continue

        n_pos += 1
        scores = best_match_iou(pred_boxes, gt_boxes)
        box_ious_all.extend(scores)
        if scores and max(scores) >= 0.5:
            hits += 1

        if gt_masks and pred_masks:
            m_scores = best_mask_ious(pred_masks, gt_masks)
            if real_poly:
                mask_ious_real.extend(m_scores)
            else:
                mask_ious_pseudo.extend(m_scores)

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    report = {
        "agent": "agent_06_segmentation",
        "split": args.split,
        "model": str(seg.resolved_path),
        "model_kind": seg.model_kind,
        "n_images": len(images),
        "n_positive": n_pos,
        "n_empty": n_empty,
        "bbox_recall_at_0.5": (hits / n_pos) if n_pos else 0.0,
        "bbox_mean_iou": _mean(box_ious_all),
        "empty_fp_rate": (n_fp_empty / n_empty) if n_empty else 0.0,
        "n_images_with_real_polygon_gt": n_real_poly_images,
        "mask_mean_iou_real_gt": _mean(mask_ious_real) if mask_ious_real else None,
        "pseudo_mask_miou": _mean(mask_ious_pseudo) if mask_ious_pseudo else None,
        "note": (
            "Dataset labels are YOLO bbox (5 tokens). Real mask IoU is None "
            "unless evaluating a YOLO-seg polygon split. pseudo_mask_miou "
            "compares pred masks to ellipse/rect filled from GT boxes."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
