"""Build a fixed 70% train / 10% validate / 20% test split (seed 42)."""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
raw = ROOT / "vial_dataset" / "raw"
out = ROOT / "vial_dataset" / "split_70_10_20"
seed = 42

pairs = []
unlabeled = 0
for img in sorted(raw.glob("*.jpg")):
    txt = img.with_suffix(".txt")
    if not txt.exists():
        unlabeled += 1
        continue
    has_box = bool(txt.read_text(encoding="utf-8").strip())
    pairs.append((img.name, txt.name, has_box))

if unlabeled:
    print(f"WARNING: {unlabeled} jpg(s) in raw/ have no .txt — skipped")
if len(pairs) < 4:
    raise SystemExit(f"Need at least 4 labeled pairs in {raw}; got {len(pairs)}")
positives = sum(1 for _, _, has in pairs if has)
if positives < 1:
    raise SystemExit("Every label is empty — draw vial boxes before splitting")

rng = random.Random(seed)
rng.shuffle(pairs)
n = len(pairs)
n_test = max(1, round(n * 0.20))
n_val = max(1, round(n * 0.10))
n_train = n - n_test - n_val
if n_train < 1:
    raise SystemExit(f"Not enough images: {n}")

test = pairs[:n_test]
val = pairs[n_test : n_test + n_val]
train = pairs[n_test + n_val :]
assert len(train) + len(val) + len(test) == n
if sum(1 for _, _, has in train if has) < 1:
    raise SystemExit(f"Train split has zero positives (seed={seed}); add labels and retry")

if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)

manifest = {
    "seed": seed,
    "counts": {"train": len(train), "validate": len(val), "test": len(test), "total": n},
    "train": [p[0] for p in train],
    "validate": [p[0] for p in val],
    "test": [p[0] for p in test],
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def materialize(split: str, items) -> None:
    idir = out / "images" / split
    ldir = out / "labels" / split
    idir.mkdir(parents=True)
    ldir.mkdir(parents=True)
    for img_name, txt_name, _ in items:
        shutil.copy2(raw / img_name, idir / img_name)
        shutil.copy2(raw / txt_name, ldir / Path(txt_name).name)


materialize("train", train)
materialize("val", val)  # Ultralytics 'val' == our 10% validate
materialize("test", test)

yaml_path = out / "data.yaml"
yaml_path.write_text(
    "\n".join(
        [
            f"path: {out.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: vial",
            "",
        ]
    ),
    encoding="utf-8",
)

pos = sum(1 for _, _, p in pairs if p)
neg = sum(1 for _, _, p in pairs if not p)
print(
    json.dumps(
        {
            "total": n,
            "positives": pos,
            "empty_negatives": neg,
            **manifest["counts"],
            "yaml": str(yaml_path),
        },
        indent=2,
    )
)
