#!/usr/bin/env python3
"""Build a compact multi-orient train set for fast CPU fine-tunes.

Keeps all sideways synth + rotated samples + a capped upright real set so
training finishes in a reasonable wall-clock on CPU while still covering both
orientations.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
SRC = PKG / "data" / "multi_orient"
DST = PKG / "data" / "multi_orient_compact"


def main() -> int:
    if not (SRC / "orientation_manifest.json").exists():
        print("Run prepare_multiorient_dataset.py first", file=sys.stderr)
        return 1
    man = json.loads((SRC / "orientation_manifest.json").read_text(encoding="utf-8"))
    samples = man["samples"]
    rng = random.Random(42)

    upright_train = [s for s in samples if s["split"] == "train" and s["orient"] == "upright"]
    side_train = [s for s in samples if s["split"] == "train" and s["orient"] == "sideways"]
    neg_train = [s for s in samples if s["split"] == "train" and s["orient"] == "negative"]
    # Cap upright; keep ALL sideways; keep some negatives
    upright_keep = rng.sample(upright_train, min(100, len(upright_train)))
    neg_keep = rng.sample(neg_train, min(30, len(neg_train))) if neg_train else []
    train_keep = upright_keep + side_train + neg_keep

    val_keep = [s for s in samples if s["split"] == "val"]
    test_keep = [s for s in samples if s["split"] == "test"]

    if DST.exists():
        shutil.rmtree(DST)
    for split in ("train", "val", "test"):
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

    def copy_one(s):
        split = s["split"]
        src_img = SRC / s["image"]
        src_lbl = SRC / s["label"]
        if not src_img.exists():
            return False
        dst_img = DST / "images" / split / src_img.name
        dst_lbl = DST / "labels" / split / (src_img.stem + ".txt")
        shutil.copy2(src_img, dst_img)
        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)
        else:
            dst_lbl.write_text("", encoding="utf-8")
        return True

    n = 0
    for s in train_keep:
        # remap split field stays train
        s = dict(s)
        s["split"] = "train"
        if copy_one(s):
            n += 1
    for s in val_keep:
        copy_one(s)
    for s in test_keep:
        copy_one(s)

    yaml = (
        f"path: {str(DST).replace(chr(92), '/')}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: vial\n"
    )
    (DST / "data.yaml").write_text(yaml, encoding="utf-8")
    out_man = {
        "parent": str(SRC),
        "note": "compact CPU train set",
        "train_counts": {
            "upright": len(upright_keep),
            "sideways": len(side_train),
            "negative": len(neg_keep),
            "total_copied_train": n,
        },
        "samples": train_keep + val_keep + test_keep,
    }
    (DST / "orientation_manifest.json").write_text(json.dumps(out_man, indent=2), encoding="utf-8")
    print(
        f"compact train upright={len(upright_keep)} sideways={len(side_train)} "
        f"neg={len(neg_keep)} total_train_files~{n} -> {DST}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
