#!/usr/bin/env python3
"""Tiny multi-orient set for CPU: upright + synth sideways + sampled rotations."""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
SRC = PKG / "data" / "multi_orient"
DST = PKG / "data" / "multi_orient_fast"


def main() -> int:
    if not (SRC / "orientation_manifest.json").exists():
        print("Run prepare_multiorient_dataset.py first", file=sys.stderr)
        return 1
    man = json.loads((SRC / "orientation_manifest.json").read_text(encoding="utf-8"))
    samples = man["samples"]
    rng = random.Random(42)
    up = [s for s in samples if s["split"] == "train" and s["orient"] == "upright"]
    side_synth = [
        s
        for s in samples
        if s["split"] == "train"
        and s["orient"] == "sideways"
        and s.get("source") == "sideways_race_synth"
    ]
    side_rot = [
        s
        for s in samples
        if s["split"] == "train"
        and s["orient"] == "sideways"
        and s.get("source") == "rotated_real"
    ]
    neg = [s for s in samples if s["split"] == "train" and s["orient"] == "negative"]
    keep = (
        rng.sample(up, min(60, len(up)))
        + side_synth
        + rng.sample(side_rot, min(100, len(side_rot)))
        + (rng.sample(neg, min(20, len(neg))) if neg else [])
    )
    val = [s for s in samples if s["split"] == "val"]
    test = [s for s in samples if s["split"] == "test"]

    if DST.exists():
        shutil.rmtree(DST)
    for split in ("train", "val", "test"):
        (DST / "images" / split).mkdir(parents=True)
        (DST / "labels" / split).mkdir(parents=True)

    def copy(s, split=None):
        split = split or s["split"]
        si = SRC / s["image"]
        sl = SRC / s["label"]
        if not si.exists():
            return
        di = DST / "images" / split / si.name
        dl = DST / "labels" / split / (si.stem + ".txt")
        shutil.copy2(si, di)
        if sl.exists():
            shutil.copy2(sl, dl)
        else:
            dl.write_text("", encoding="utf-8")

    for s in keep:
        copy(s, "train")
    for s in val:
        copy(s)
    for s in test:
        copy(s)

    path = str(DST).replace("\\", "/")
    (DST / "data.yaml").write_text(
        f"path: {path}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: vial\n",
        encoding="utf-8",
    )
    print(
        f"fast train n={len(keep)} side_synth={len(side_synth)} "
        f"side_rot={min(100, len(side_rot))} up={min(60, len(up))} -> {DST}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
