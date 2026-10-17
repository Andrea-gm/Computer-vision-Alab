#!/usr/bin/env python3
"""Fine-tune YOLOv8-OBB on the prepared oriented vial dataset."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "obb_split" / "data_obb.yaml")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", type=str, default="yolov8n-obb.pt", help="Base OBB checkpoint")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--project", type=Path, default=ROOT / "models" / "train_runs")
    ap.add_argument("--name", type=str, default="obb_vial")
    ap.add_argument("--prepare", action="store_true", help="Run prepare_obb_dataset.py first")
    ap.add_argument("--sideways-mult", type=int, default=3)
    ap.add_argument("--upright-mult", type=int, default=2)
    ap.add_argument("--resume-best", action="store_true", help="Continue from models/best.pt")
    args = ap.parse_args()

    if args.prepare or not args.data.exists():
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from prepare_obb_dataset import build_dataset

        camera = ROOT.parents[2]
        build_dataset(
            ROOT / "data" / "obb_split",
            camera_root=camera,
            sideways_mult=args.sideways_mult,
            upright_mult=args.upright_mult,
            seed=42,
            val_frac=0.15,
        )

    from ultralytics import YOLO

    weights = args.model
    if args.resume_best:
        local_best = ROOT / "models" / "best.pt"
        if local_best.exists():
            weights = str(local_best)
    model = YOLO(weights)
    kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        patience=10,
        degrees=180.0,  # strong rotation aug — orientation matters
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.8,
        workers=0,
    )
    if args.device:
        kwargs["device"] = args.device
    results = model.train(**kwargs)

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if not best.exists():
        # ultralytics may nest runs
        candidates = list((Path(args.project) / args.name).rglob("best.pt"))
        best = candidates[0] if candidates else best
    dest = ROOT / "models" / "best.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy2(best, dest)
        print(f"Exported {best} → {dest}")
    else:
        print("WARNING: best.pt not found after train", file=sys.stderr)
    print(results)
    return


if __name__ == "__main__":
    main()
