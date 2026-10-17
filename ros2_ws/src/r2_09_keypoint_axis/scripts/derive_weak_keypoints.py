#!/usr/bin/env python3
"""Derive weak cap/base keypoint labels from YOLO boxes + white-cap localization.

Usage:
  python scripts/derive_weak_keypoints.py
  python scripts/derive_weak_keypoints.py --split train --max 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2_09_keypoint_axis.detector import DEFAULT_CAMERA  # noqa: E402
from r2_09_keypoint_axis.weak_labels import (  # noqa: E402
    derive_dataset_weak_labels,
    save_weak_labels,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_CAMERA / "vial_dataset" / "split_70_10_20",
    )
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "weak_keypoints.json",
    )
    args = ap.parse_args()
    img_dir = args.split_dir / "images" / args.split
    lbl_dir = args.split_dir / "labels" / args.split
    max_images = args.max if args.max > 0 else None
    records = derive_dataset_weak_labels(img_dir, lbl_dir, max_images=max_images)
    save_weak_labels(records, args.out)
    print(f"Wrote {len(records)} weak keypoint records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
