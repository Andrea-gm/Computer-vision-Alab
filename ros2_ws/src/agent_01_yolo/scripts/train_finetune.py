#!/usr/bin/env python3
"""Thin wrapper: fine-tune Ultralytics YOLO on vial_dataset/split_70_10_20.

Prefer the Camera-root trainer for production promote discipline:
  python E:\\Desktop\\Jessica\\Camera\\train_vial_model.py train

This script is a self-contained path for the competition package:
  cd E:\\Desktop\\Jessica\\Camera\\ros2_ws\\src\\agent_01_yolo
  python scripts/train_finetune.py --epochs 40 --model yolov8s.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _camera_root() -> Path:
    for parent in [ROOT, *ROOT.parents]:
        if (parent / "vial_dataset").is_dir():
            return parent
    return ROOT.parents[2]  # .../Camera/ros2_ws/src/agent_01_yolo → Camera


CAMERA = _camera_root()
SPLIT_YAML = CAMERA / "vial_dataset" / "split_70_10_20" / "data.yaml"
DEFAULT_BASE = CAMERA / "runs" / "bon-winner" / "weights" / "best.pt"
OUT_DIR = ROOT / "runs" / "agent01-ft"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=SPLIT_YAML)
    ap.add_argument("--model", type=Path, default=None, help="Base .pt to fine-tune")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="")
    ap.add_argument("--project", type=Path, default=OUT_DIR)
    ap.add_argument("--name", default="exp")
    args = ap.parse_args()

    if not args.data.exists():
        print(f"Missing data yaml: {args.data}", file=sys.stderr)
        print("Rebuild split: python Camera/train_vial_model.py split", file=sys.stderr)
        return 1

    base = args.model
    if base is None:
        for cand in (
            DEFAULT_BASE,
            CAMERA / "runs" / "vial-live" / "weights" / "best.pt",
            CAMERA / "yolov8s.pt",
            CAMERA / "yolov8n.pt",
        ):
            if cand.exists():
                base = cand
                break
    if base is None or not Path(base).exists():
        print("No base weights found.", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    print(f"fine-tune base={base} data={args.data} -> {args.project}/{args.name}")
    model = YOLO(str(base))
    kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        patience=15,
        freeze=5,
    )
    if args.device:
        kwargs["device"] = args.device
    model.train(**kwargs)
    best = args.project / args.name / "weights" / "best.pt"
    print(f"done. best={best}")
    print("Eval: python scripts/evaluate.py --model", best)
    print(
        "Promote only after clear holdout beat "
        "(see Camera/train_vial_model.py --export-live discipline)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
