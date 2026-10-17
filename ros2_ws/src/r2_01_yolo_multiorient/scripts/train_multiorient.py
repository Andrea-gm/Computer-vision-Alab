#!/usr/bin/env python3
"""Fine-tune YOLO on the multi-orientation dataset.

Starts from vial-live (or yolov8s) and trains with strong in-plane rotation
augmentation (degrees=180) plus the sideways/rotated samples from
prepare_multiorient_dataset.py.

Saves best weights to:
  <pkg>/models/best.pt
  and Camera/runs/r2-01-multiorient/weights/best.pt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def _camera_root() -> Path:
    for p in PKG.parents:
        if (p / "vial_dataset").is_dir():
            return p
    return PKG.parents[2]


def main() -> int:
    cam = _camera_root()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=PKG / "data" / "multi_orient" / "data.yaml",
    )
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--degrees", type=float, default=180.0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--freeze", type=int, default=8)
    ap.add_argument("--prepare", action="store_true", help="Run dataset prep first")
    args = ap.parse_args()

    if args.prepare or not args.data.exists():
        prep = PKG / "scripts" / "prepare_multiorient_dataset.py"
        print(f"Preparing dataset via {prep}")
        import subprocess

        rc = subprocess.call([sys.executable, str(prep)])
        if rc != 0:
            return rc

    if not args.data.exists():
        print(f"Missing {args.data}", file=sys.stderr)
        return 1

    base = args.model
    if base is None:
        for cand in (
            cam / "runs" / "vial-live" / "weights" / "best.pt",
            cam / "runs" / "bon-winner" / "weights" / "best.pt",
            cam / "yolov8s.pt",
            cam / "yolov8n.pt",
        ):
            if cand.exists():
                base = cand
                break
    if base is None or not Path(base).exists():
        print("No base weights found", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    project = cam / "runs" / "r2-01-multiorient"
    name = "train"
    print(
        f"fine-tune base={base} data={args.data} epochs={args.epochs} "
        f"degrees={args.degrees} device={args.device} -> {project}/{name}"
    )
    model = YOLO(str(base))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project),
        name=name,
        exist_ok=True,
        patience=args.patience,
        freeze=args.freeze,
        degrees=args.degrees,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.8,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        perspective=0.0,
        shear=5.0,
        workers=0,
    )
    best = project / name / "weights" / "best.pt"
    if not best.exists():
        print(f"Training finished but missing {best}", file=sys.stderr)
        return 1

    models_dir = PKG / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / "best.pt"
    shutil.copy2(best, dest)
    shutil.copy2(best, models_dir / "yolov8_multiorient_best.pt")
    # Also flatten into runs/r2-01-multiorient/weights for resolve_model_path
    flat = project / "weights"
    flat.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, flat / "best.pt")
    print(f"copied best -> {dest}")
    print(f"also -> {flat / 'best.pt'}")
    print("Next: python scripts/evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
