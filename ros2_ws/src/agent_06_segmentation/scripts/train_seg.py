#!/usr/bin/env python3
"""Train YOLOv8-seg on the pseudo-mask split (scaffolding).

Prereq:
  python scripts/prepare_seg_dataset.py

Then:
  python scripts/train_seg.py
  python scripts/train_seg.py --epochs 40 --model yolov8n-seg.pt

Weights land in Camera/runs/vial-seg/weights/best.pt (picked up by the ROS node).
"""

from __future__ import annotations

import argparse
from pathlib import Path

CAMERA_ROOT = Path(r"E:\Desktop\Jessica\robotics")
DEFAULT_DATA = CAMERA_ROOT / "vial_dataset" / "split_seg_bbox_poly" / "data.yaml"
DEFAULT_OUT = CAMERA_ROOT / "runs" / "vial-seg"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument(
        "--model",
        type=str,
        default="yolov8n-seg.pt",
        help="Base seg checkpoint (COCO) or prior vial-seg best.pt",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--project", type=Path, default=CAMERA_ROOT / "runs")
    ap.add_argument("--name", type=str, default="vial-seg")
    ap.add_argument(
        "--exist-ok",
        action="store_true",
        help="Overwrite runs/vial-seg if present",
    )
    args = ap.parse_args()

    if not args.data.is_file():
        raise SystemExit(
            f"Missing {args.data}. Run prepare_seg_dataset.py first."
        )

    from ultralytics import YOLO

    model = YOLO(args.model)
    kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        exist_ok=bool(args.exist_ok),
        patience=8,
        pretrained=True,
    )
    if args.device:
        kwargs["device"] = args.device
    results = model.train(**kwargs)
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"Training finished. best → {best} exists={best.is_file()}")
    print(
        "Note: labels are PSEUDO masks from bboxes. Promote only after real "
        "polygon labeling if you need accurate instance masks."
    )
    return results


if __name__ == "__main__":
    main()
