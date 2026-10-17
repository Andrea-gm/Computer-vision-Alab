#!/usr/bin/env python3
"""Convert YOLO bbox labels → YOLO-seg polygon labels (4-corner rectangle).

The Camera vial_dataset has **no real masks**. This scaffold writes a parallel
split under ``vial_dataset/split_seg_bbox_poly/`` so Ultralytics YOLOv8-seg can
train. Pseudo-polygons are the four corners of each GT box (optionally shrunk
to an inscribed ellipse sampled as N points).

Usage (from Camera/ or this package):
  python scripts/prepare_seg_dataset.py
  python scripts/prepare_seg_dataset.py --shape ellipse --points 16
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import yaml

CAMERA_ROOT = Path(r"E:\Desktop\Jessica\robotics")
DEFAULT_SRC = CAMERA_ROOT / "vial_dataset" / "split_70_10_20"
DEFAULT_DST = CAMERA_ROOT / "vial_dataset" / "split_seg_bbox_poly"


def bbox_to_rect_poly(cx: float, cy: float, w: float, h: float) -> list[float]:
    """Normalized YOLO-seg polygon: x1 y1 x2 y2 x3 y3 x4 y4 (TL→TR→BR→BL)."""
    x0, y0 = cx - w / 2, cy - h / 2
    x1, y1 = cx + w / 2, cy + h / 2
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def bbox_to_ellipse_poly(
    cx: float, cy: float, w: float, h: float, n: int = 16
) -> list[float]:
    ax, ay = w / 2, h / 2
    pts: list[float] = []
    for i in range(n):
        t = 2 * math.pi * i / n
        pts.append(cx + ax * math.cos(t))
        pts.append(cy + ay * math.sin(t))
    # clamp to [0,1]
    return [min(1.0, max(0.0, v)) for v in pts]


def convert_label_file(
    src: Path, dst: Path, *, shape: str, points: int
) -> int:
    lines_out: list[str] = []
    n = 0
    text = src.read_text(encoding="utf-8").strip()
    if not text:
        dst.write_text("", encoding="utf-8")
        return 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = parts[0]
        # Already a polygon (≥6 coords after class)
        if len(parts) > 5:
            lines_out.append(line.strip())
            n += 1
            continue
        cx, cy, w, h = map(float, parts[1:5])
        if shape == "ellipse":
            poly = bbox_to_ellipse_poly(cx, cy, w, h, n=points)
        else:
            poly = bbox_to_rect_poly(cx, cy, w, h)
            poly = [min(1.0, max(0.0, v)) for v in poly]
        lines_out.append(cls + " " + " ".join(f"{v:.6f}" for v in poly))
        n += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    ap.add_argument(
        "--shape",
        choices=("rect", "ellipse"),
        default="ellipse",
        help="Pseudo-mask shape written as YOLO-seg polygons",
    )
    ap.add_argument("--points", type=int, default=16, help="Ellipse sample count")
    ap.add_argument(
        "--symlink-images",
        action="store_true",
        help="Symlink images/ instead of copying (faster; Windows may need admin)",
    )
    args = ap.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    if not src.is_dir():
        raise SystemExit(f"Source split missing: {src}")

    dst.mkdir(parents=True, exist_ok=True)
    total = 0
    for split in ("train", "val", "test"):
        lab_src = src / "labels" / split
        lab_dst = dst / "labels" / split
        img_src = src / "images" / split
        img_dst = dst / "images" / split
        if not lab_src.is_dir():
            print(f"skip missing {lab_src}")
            continue
        lab_dst.mkdir(parents=True, exist_ok=True)
        img_dst.mkdir(parents=True, exist_ok=True)

        for lab in sorted(lab_src.glob("*.txt")):
            total += convert_label_file(
                lab, lab_dst / lab.name, shape=args.shape, points=args.points
            )
            # copy / link matching image
            stem = lab.stem
            img = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                cand = img_src / f"{stem}{ext}"
                if cand.is_file():
                    img = cand
                    break
            if img is None:
                continue
            target = img_dst / img.name
            if target.exists() or target.is_symlink():
                continue
            if args.symlink_images:
                try:
                    target.symlink_to(img.resolve())
                except OSError:
                    shutil.copy2(img, target)
            else:
                shutil.copy2(img, target)

    data = {
        "path": str(dst.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "vial"},
    }
    yaml_path = dst / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    meta = {
        "source": str(src),
        "shape": args.shape,
        "points": args.points,
        "note": "PSEUDO masks from bbox — not hand-drawn GT. For train scaffolding only.",
        "instances_converted": total,
    }
    (dst / "manifest_seg.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {total} pseudo-polygons → {dst}")
    print(f"data.yaml → {yaml_path}")


if __name__ == "__main__":
    main()
