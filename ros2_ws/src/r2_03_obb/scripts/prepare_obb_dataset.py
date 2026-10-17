#!/usr/bin/env python3
"""Convert AABB YOLO labels + sideways synth → Ultralytics OBB dataset.

Sideways samples are oversampled (default 4×) so orientation is not ignored.
Hard angled synth notes (`angle=…`) recover true rotated length/diameter.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2_03_obb.obb_geometry import (  # noqa: E402
    OBBDet,
    aabb_xywh_to_obb,
    load_yolo_aabb_label,
    obb_to_yolo_corners_line,
    parse_angle_from_note,
    yolo_xywhn_to_obb,
)

CAMERA = ROOT.parents[2]  # .../Camera/ros2_ws/src/r2_03_obb → parents[2]=Camera


def _copy_pair(
    img_src: Path,
    dets: List[OBBDet],
    out_img: Path,
    out_lbl: Path,
) -> None:
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_src, out_img)
    bgr = cv2.imread(str(img_src), cv2.IMREAD_COLOR)
    if bgr is None:
        h = w = 640
    else:
        h, w = bgr.shape[:2]
    lines = [obb_to_yolo_corners_line(d, w, h) for d in dets]
    out_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _from_raw(raw_dir: Path) -> List[Tuple[Path, List[OBBDet], str]]:
    """Return (image, dets, split_hint) for vial_dataset/raw."""
    items = []
    for img in sorted(raw_dir.glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        dets = load_yolo_aabb_label(img.with_suffix(".txt"), w, h)
        # Empty negatives: keep empty label.
        ori = "empty" if not dets else (dets[0].orientation if dets else "upright")
        items.append((img, dets, ori))
    return items


def _from_sideways_synth(synth_dir: Path) -> List[Tuple[Path, List[OBBDet], str]]:
    man_path = synth_dir / "manifest.json"
    if not man_path.exists():
        return []
    man = json.loads(man_path.read_text(encoding="utf-8"))
    samples = man.get("samples") or man.get("entries") or []
    if not samples and isinstance(man, list):
        samples = man
    # Some manifests are {"samples":[...]} — also accept top-level list via keys
    if not samples and "id" not in man:
        # try reading as list of filenames from directory scan fallback
        pass
    items: List[Tuple[Path, List[OBBDet], str]] = []
    # Prefer iterating entries from manifest; if structure is {"samples": [...]}
    if isinstance(man, dict) and "samples" not in man and "entries" not in man:
        # maybe each key is an id — fall back to glob JSON
        samples = []
        for jp in sorted(synth_dir.glob("*.json")):
            if jp.name == "manifest.json":
                continue
            samples.append(json.loads(jp.read_text(encoding="utf-8")))

    for entry in samples:
        if isinstance(entry, str):
            continue
        kind = entry.get("kind", "")
        role = entry.get("role", "")
        color_name = entry.get("color")
        if not color_name:
            continue
        img = synth_dir / color_name
        if not img.exists():
            continue
        boxes = entry.get("boxes") or []
        note = entry.get("note") or ""
        angle = parse_angle_from_note(note)
        dets: List[OBBDet] = []
        for b in boxes:
            if len(b) < 4:
                continue
            x, y, bw, bh = [float(v) for v in b[:4]]
            # Axis-aligned sideways left/right → angle 0; hard uses note angle.
            if angle is None and kind.startswith("sideways"):
                ang = 0.0
            elif angle is None and kind.startswith("hard_sideways"):
                ang = -20.0  # default mild tilt if note missing
            else:
                ang = angle
            # upright / topdown / empty / hard_negative / real_wide: only positives
            if role == "negative" or kind in {
                "upright",
                "topdown",
                "empty",
                "hard_negative",
                "real_wide",
            }:
                # For upright controls with note box — still convert if boxes present
                # and kind is upright with control GT in note — skip (negatives for sideways race)
                if kind == "upright" and boxes:
                    # Include upright GT for OBB train (detect upright too).
                    dets.append(aabb_xywh_to_obb(x, y, bw, bh, angle_deg=None, source="synth_upright"))
                continue
            dets.append(aabb_xywh_to_obb(x, y, bw, bh, angle_deg=ang, source="synth_sideways"))
        # For upright kind: parse control box from note if empty boxes
        if kind == "upright" and not dets:
            import re

            m = re.search(r"box=\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", note)
            if m:
                x, y, bw, bh = map(int, m.groups())
                dets.append(aabb_xywh_to_obb(x, y, bw, bh, angle_deg=None, source="synth_upright"))
        tag = "sideways" if "sideways" in kind else ("upright" if kind == "upright" else kind)
        # Skip pure negatives without boxes for OBB train empties separately
        if not dets and role == "negative" and kind in {"empty", "hard_negative", "topdown", "real_wide"}:
            items.append((img, [], "empty"))
        elif dets:
            items.append((img, dets, tag))
    return items


def _is_upright_tag(tag: str) -> bool:
    if tag in {"upright", "unknown"}:
        return True
    if tag in {"empty", "sideways", "topdown", "hard_negative", "real_wide"}:
        return False
    # raw stems like vial_0001 / 01_held_...
    return True


def build_dataset(
    out_dir: Path,
    *,
    camera_root: Path,
    sideways_mult: int,
    upright_mult: int,
    seed: int,
    val_frac: float,
) -> dict:
    rng = random.Random(seed)
    raw_items = _from_raw(camera_root / "vial_dataset" / "raw")
    # Prefer train split images when available (keeps test clean for holdout reports).
    split_train = camera_root / "vial_dataset" / "split_70_10_20" / "images" / "train"
    if split_train.is_dir():
        # Rebuild raw_items from train+val only for FT; still convert AABB→OBB.
        labeled: List[Tuple[Path, List[OBBDet], str]] = []
        for split_name in ("train", "val"):
            img_dir = camera_root / "vial_dataset" / "split_70_10_20" / "images" / split_name
            lbl_dir = camera_root / "vial_dataset" / "split_70_10_20" / "labels" / split_name
            if not img_dir.is_dir():
                continue
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                h, w = bgr.shape[:2]
                lbl = lbl_dir / f"{img.stem}.txt"
                dets = load_yolo_aabb_label(lbl, w, h) if lbl.exists() else []
                tag = "empty" if not dets else "upright"
                labeled.append((img, dets, tag))
        if labeled:
            raw_items = labeled

    synth_items = _from_sideways_synth(camera_root / "sideways_race" / "synth")

    # Balance upright vs sideways (Round-2 contract: both orientations).
    expanded: List[Tuple[Path, List[OBBDet], str]] = []
    for img, dets, tag in raw_items:
        n = upright_mult if (dets and _is_upright_tag(tag)) else 1
        for _ in range(n):
            expanded.append((img, dets, tag))
    for img, dets, tag in synth_items:
        if tag == "sideways":
            n = sideways_mult
        elif tag == "upright":
            n = max(upright_mult, 6)  # synth upright was starved — oversample hard
        else:
            n = 1
        for _ in range(n):
            expanded.append((img, dets, tag))

    rng.shuffle(expanded)
    n_val = max(1, int(len(expanded) * val_frac))
    val_set = expanded[:n_val]
    train_set = expanded[n_val:]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split, subset in (("train", train_set), ("val", val_set)):
        for i, (img, dets, tag) in enumerate(subset):
            stem = f"{tag}_{img.stem}_{i:04d}"
            out_img = out_dir / "images" / split / f"{stem}{img.suffix.lower()}"
            out_lbl = out_dir / "labels" / split / f"{stem}.txt"
            _copy_pair(img, dets, out_img, out_lbl)

    data_yaml = {
        "path": str(out_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "vial"},
    }
    yaml_path = out_dir / "data_obb.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    def _count(tag: str, subset) -> int:
        return sum(1 for _, _, t in subset if t == tag)

    n_up = sum(1 for _, d, t in train_set if d and _is_upright_tag(t))
    summary = {
        "out_dir": str(out_dir),
        "train": len(train_set),
        "val": len(val_set),
        "train_sideways": _count("sideways", train_set),
        "train_upright_pos": n_up,
        "train_synth_upright": _count("upright", train_set),
        "sideways_mult": sideways_mult,
        "upright_mult": upright_mult,
        "data_yaml": str(yaml_path),
    }
    (out_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "obb_split",
        help="Output OBB dataset root",
    )
    ap.add_argument("--camera-root", type=Path, default=CAMERA)
    ap.add_argument("--sideways-mult", type=int, default=3, help="Oversample factor for sideways")
    ap.add_argument("--upright-mult", type=int, default=2, help="Oversample factor for upright/raw")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()
    build_dataset(
        args.out,
        camera_root=args.camera_root,
        sideways_mult=args.sideways_mult,
        upright_mult=args.upright_mult,
        seed=args.seed,
        val_frac=args.val_frac,
    )


if __name__ == "__main__":
    main()
