#!/usr/bin/env python3
"""Train YOLO on the rotation-augmented dataset (never reuse upright-biased weights).

Base weights are COCO-pretrained yolov8n/s only. Live/vial-live/r1/bon weights
are intentionally excluded so Round-1 upright bias is not carried over.

  python scripts/build_aug_dataset.py
  python scripts/train.py --epochs 30
  python scripts/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]


def camera_root() -> Path:
    for parent in [PKG, *PKG.parents]:
        if (parent / "vial_dataset").is_dir():
            return parent
    return PKG.parents[3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=PKG / "data" / "aug_split" / "data.yaml",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", type=Path, default=None, help="COCO .pt only")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--name", default="r2_05")
    args = ap.parse_args()

    if not args.data.exists():
        print(f"Missing {args.data}. Run scripts/build_aug_dataset.py first.", file=sys.stderr)
        return 1

    cam = camera_root()
    # HARD RULE: only COCO base — never vial-live / race winners.
    banned = {
        "vial-live",
        "vial-ft",
        "bon-winner",
        "r1-winner",
        "r3-winner",
        "train-race",
    }
    base = args.model
    if base is None:
        for cand in (cam / "yolov8n.pt", cam / "yolov8s.pt", Path("yolov8n.pt")):
            if cand.exists():
                base = cand
                break
    if base is None or not Path(base).exists():
        print("No COCO base weights (yolov8n.pt / yolov8s.pt).", file=sys.stderr)
        return 1
    base_s = str(base).replace("\\", "/").lower()
    if any(b in base_s for b in banned):
        print(
            f"REFUSING banned upright-biased weights: {base}\n"
            "Use Camera/yolov8n.pt or yolov8s.pt and retrain.",
            file=sys.stderr,
        )
        return 2

    from ultralytics import YOLO

    project = PKG / "runs"
    print(f"TRAIN base={base} data={args.data} epochs={args.epochs} imgsz={args.imgsz}")
    print("Online aug: degrees=180, perspective=0.001, scale=0.5 (plus offline set)")
    model = YOLO(str(base))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        degrees=180.0,
        perspective=0.001,
        scale=0.5,
        shear=5.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.5,
        workers=0,
        plots=False,
    )

    best = project / args.name / "weights" / "best.pt"
    if not best.exists():
        print(f"Training finished but missing {best}", file=sys.stderr)
        return 1

    dest_dir = PKG / "models" / "weights"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "best.pt"
    shutil.copy2(best, dest)
    last = project / args.name / "weights" / "last.pt"
    if last.exists():
        shutil.copy2(last, dest_dir / "last.pt")

    meta = {
        "base_weights": str(base),
        "banned_sources_excluded": sorted(banned),
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "best_run": str(best),
        "exported": str(dest),
        "online_aug": {
            "degrees": 180.0,
            "perspective": 0.001,
            "scale": 0.5,
            "shear": 5.0,
            "flipud": 0.5,
            "fliplr": 0.5,
        },
    }
    meta_path = PKG / "results" / "train_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Exported weights -> {dest}")
    print(f"Wrote {meta_path}")
    print("Next: python scripts/evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
