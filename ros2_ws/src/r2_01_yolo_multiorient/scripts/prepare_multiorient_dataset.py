#!/usr/bin/env python3
"""Build a multi-orientation YOLO dataset from vial_dataset + sideways_race.

Orientation coverage strategy
-----------------------------
1. Copy split_70_10_20 train/val/test (mostly upright real stills).
2. For every train/val positive, add 90° CW and 90° CCW copies with rotated
   YOLO boxes → synthetic sideways supervision from real imagery.
3. Ingest sideways_race/synth color+JSON:
   - positives with kind containing 'sideways' → train (sideways)
   - kind 'upright' → val extras (upright synth)
   - empty / hard_negative / topdown / real_wide → negatives in train/val

Writes:
  <pkg>/data/multi_orient/{images,labels}/{train,val,test}/
  <pkg>/data/multi_orient/data.yaml
  <pkg>/data/multi_orient/orientation_manifest.json  (subset tags for eval)

Subset rules (used by evaluate.py)
----------------------------------
- upright: stem starts with 'upright_' OR (real still AND median GT h/w >= 1.0)
           OR stem contains '_orig' / no rot suffix from real split
- sideways: stem contains 'sideways' OR stem endswith '_r90' / '_r270'
            OR median GT h/w < 0.95
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

PKG = Path(__file__).resolve().parents[1]
CAMERA = PKG.parents[2]  # .../Camera/ros2_ws/src/<pkg> → Camera
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _camera_root() -> Path:
    for p in [CAMERA, *PKG.parents]:
        if (p / "vial_dataset").is_dir():
            return p
    return CAMERA


def parse_yolo_lines(txt: Path) -> List[Tuple[int, float, float, float, float]]:
    if not txt.exists():
        return []
    text = txt.read_text(encoding="utf-8").strip()
    if not text:
        return []
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:])
        rows.append((cls, cx, cy, w, h))
    return rows


def write_yolo(txt: Path, rows: Iterable[Tuple[int, float, float, float, float]]) -> None:
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in rows]
    txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def rotate_yolo_90_cw(rows, h, w):
    """Image ROTATE_90_CLOCKWISE: new size (w,h)=(H_old, W_old) wait — new=(W,H)=(w,h) from (h,w).
    Original size (h,w). New size (w, h) i.e. (W_new=h, H_new=w)... 
    cv2.ROTATE_90_CLOCKWISE: dst.shape = (src.cols, src.rows) = (w, h).
    Point (x,y) -> (h-1-y, x) in pixel space.
    Normalized: cx' = 1 - cy, cy' = cx, w' = h_box, h' = w_box (normalized dims swap).
    """
    out = []
    for c, cx, cy, bw, bh in rows:
        # normalized center → after 90 CW
        ncx = 1.0 - cy
        ncy = cx
        out.append((c, ncx, ncy, bh, bw))
    return out


def rotate_yolo_90_ccw(rows, h, w):
    """90 CCW: (x,y)->(y, w-1-x); ncx=cy, ncy=1-cx; dims swap."""
    out = []
    for c, cx, cy, bw, bh in rows:
        ncx = cy
        ncy = 1.0 - cx
        out.append((c, ncx, ncy, bh, bw))
    return out


def copy_pair(img: Path, lbl: Path, dst_img: Path, dst_lbl: Path) -> None:
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, dst_img)
    if lbl.exists():
        shutil.copy2(lbl, dst_lbl)
    else:
        dst_lbl.write_text("", encoding="utf-8")


def add_rotated(
    img: Path,
    lbl: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    suffix: str,
    rot_code,
    rotate_fn,
) -> Optional[str]:
    bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    rows = parse_yolo_lines(lbl)
    if not rows:
        return None
    rot = cv2.rotate(bgr, rot_code)
    new_rows = rotate_fn(rows, h, w)
    stem = f"{img.stem}{suffix}"
    out_img = out_img_dir / f"{stem}.jpg"
    out_lbl = out_lbl_dir / f"{stem}.txt"
    cv2.imwrite(str(out_img), rot)
    write_yolo(out_lbl, new_rows)
    return stem


def xywh_abs_to_yolo(box, img_w, img_h):
    x, y, bw, bh = [float(v) for v in box]
    cx = (x + bw / 2.0) / img_w
    cy = (y + bh / 2.0) / img_h
    return (0, cx, cy, bw / img_w, bh / img_h)


def ingest_synth(synth_dir: Path, out_root: Path, manifest_samples: list) -> dict:
    man_path = synth_dir / "manifest.json"
    if not man_path.exists():
        print(f"WARN: missing {man_path}", file=sys.stderr)
        return {"added": 0}
    man = json.loads(man_path.read_text(encoding="utf-8"))
    counts = {"sideways_train": 0, "upright_val": 0, "neg_train": 0, "neg_val": 0}
    for entry in man.get("samples", []):
        kind = entry.get("kind", "")
        role = entry.get("role", "")
        color = synth_dir / entry["color"]
        if not color.exists():
            continue
        bgr = cv2.imread(str(color), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        rows = [xywh_abs_to_yolo(b, w, h) for b in (entry.get("boxes") or [])]
        stem = entry["id"]
        # Route
        if role == "positive" and "sideways" in kind:
            split = "train"
            counts["sideways_train"] += 1
            orient = "sideways"
        elif role == "positive" and kind == "upright":
            split = "val"
            counts["upright_val"] += 1
            orient = "upright"
        elif role == "negative" or kind in ("empty", "hard_negative", "topdown", "real_wide"):
            # put half neg into val for FP monitoring
            split = "val" if (hash(stem) % 3 == 0) else "train"
            counts["neg_val" if split == "val" else "neg_train"] += 1
            orient = "negative"
            rows = []  # force empty for hard FP control
            if kind == "topdown" and entry.get("boxes"):
                # keep boxes if present — topdown is hard; treat as sideways-ish for orient tag
                rows = [xywh_abs_to_yolo(b, w, h) for b in entry["boxes"]]
                orient = "other"
        else:
            continue

        out_img = out_root / "images" / split / f"{stem}.png"
        out_lbl = out_root / "labels" / split / f"{stem}.txt"
        out_img.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img), bgr)
        write_yolo(out_lbl, rows)
        manifest_samples.append(
            {
                "stem": stem,
                "split": split,
                "orient": orient,
                "source": "sideways_race_synth",
                "kind": kind,
                "image": str(out_img.relative_to(out_root)).replace("\\", "/"),
                "label": str(out_lbl.relative_to(out_root)).replace("\\", "/"),
            }
        )
    return counts


def classify_real_orient(rows) -> str:
    if not rows:
        return "empty"
    aspects = [bh / bw for _, _, _, bw, bh in rows if bw > 1e-9]
    if not aspects:
        return "empty"
    med = float(np.median(aspects))
    if med < 0.95:
        return "sideways"
    return "upright"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=PKG / "data" / "multi_orient",
    )
    ap.add_argument(
        "--split-dir",
        type=Path,
        default=None,
    )
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--skip-rotate", action="store_true")
    args = ap.parse_args()

    cam = _camera_root()
    split_dir = args.split_dir or (cam / "vial_dataset" / "split_70_10_20")
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest = {
        "camera_root": str(cam),
        "split_source": str(split_dir),
        "subset_rules": {
            "upright": "real stills with GT median h/w>=0.95; synth upright_*; stems without rot suffix",
            "sideways": "synth *sideways*; rotated copies *_r90/*_r270; GT median h/w<0.95",
            "negative": "empty labels / hard_negative / empty synth",
        },
        "samples": [],
    }

    n_copy = n_rot = 0
    for split in ("train", "val", "test"):
        img_dir = split_dir / "images" / split
        lbl_dir = split_dir / "labels" / split
        if not img_dir.exists():
            print(f"WARN missing {img_dir}", file=sys.stderr)
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            dst_img = out / "images" / split / img.name
            # normalize extension keep
            if dst_img.suffix.lower() != img.suffix.lower():
                dst_img = out / "images" / split / img.name
            dst_lbl = out / "labels" / split / f"{img.stem}.txt"
            copy_pair(img, lbl, dst_img, dst_lbl)
            rows = parse_yolo_lines(dst_lbl)
            orient = classify_real_orient(rows)
            manifest["samples"].append(
                {
                    "stem": img.stem,
                    "split": split,
                    "orient": orient,
                    "source": "vial_dataset",
                    "kind": "real",
                    "image": str(dst_img.relative_to(out)).replace("\\", "/"),
                    "label": str(dst_lbl.relative_to(out)).replace("\\", "/"),
                }
            )
            n_copy += 1

            if args.skip_rotate or split == "test":
                continue
            if not rows:
                continue
            for suffix, code, fn in (
                ("_r90", cv2.ROTATE_90_CLOCKWISE, rotate_yolo_90_cw),
                ("_r270", cv2.ROTATE_90_COUNTERCLOCKWISE, rotate_yolo_90_ccw),
            ):
                stem = add_rotated(
                    img,
                    lbl,
                    out / "images" / split,
                    out / "labels" / split,
                    suffix,
                    code,
                    fn,
                )
                if stem is None:
                    continue
                n_rot += 1
                manifest["samples"].append(
                    {
                        "stem": stem,
                        "split": split,
                        "orient": "sideways",
                        "source": "rotated_real",
                        "kind": suffix,
                        "image": f"images/{split}/{stem}.jpg",
                        "label": f"labels/{split}/{stem}.txt",
                    }
                )

    synth_counts = {}
    if not args.skip_synth:
        synth_counts = ingest_synth(
            cam / "sideways_race" / "synth", out, manifest["samples"]
        )

    data_yaml = {
        "path": str(out).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "vial"},
    }
    # Ultralytics wants names as list or dict — write YAML manually
    yaml_text = (
        f"path: {data_yaml['path']}\n"
        f"train: {data_yaml['train']}\n"
        f"val: {data_yaml['val']}\n"
        f"test: {data_yaml['test']}\n"
        "names:\n"
        "  0: vial\n"
    )
    (out / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (out / "orientation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # summary counts
    from collections import Counter

    by = Counter((s["split"], s["orient"]) for s in manifest["samples"])
    print(f"Wrote dataset -> {out}")
    print(f"copied_real={n_copy} rotated_sideways={n_rot} synth={synth_counts}")
    for k, v in sorted(by.items()):
        print(f"  {k}: {v}")
    print(f"data.yaml -> {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
