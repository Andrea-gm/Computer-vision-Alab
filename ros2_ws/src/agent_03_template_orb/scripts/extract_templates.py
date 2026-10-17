#!/usr/bin/env python3
"""Extract vial templates from the labeled train split into data/templates/.

Usage (no ROS required):
  cd E:\\Desktop\\Jessica\\Camera\\ros2_ws\\src\\agent_03_template_orb
  python scripts/extract_templates.py
  python scripts/extract_templates.py --max 16 --split train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_03_template_orb.template_bank import (  # noqa: E402
    default_camera_root,
    default_templates_dir,
    extract_templates_from_split,
)


def main() -> int:
    camera = default_camera_root()
    ap = argparse.ArgumentParser(description="Extract vial templates for agent_03")
    ap.add_argument(
        "--split-dir",
        type=Path,
        default=camera / "vial_dataset" / "split_70_10_20",
    )
    ap.add_argument("--split", default="train", choices=("train", "val", "test"))
    ap.add_argument("--out", type=Path, default=default_templates_dir())
    ap.add_argument("--max", type=int, default=14)
    ap.add_argument("--canon-height", type=int, default=80)
    args = ap.parse_args()

    if not args.split_dir.exists():
        print(f"split-dir missing: {args.split_dir}", file=sys.stderr)
        return 1

    tmpls = extract_templates_from_split(
        args.split_dir,
        args.out,
        split=args.split,
        max_templates=args.max,
        canon_height=args.canon_height,
    )
    print(f"wrote {len(tmpls)} templates -> {args.out}")
    for t in tmpls:
        print(f"  {t.name}  src={t.source_image}  area={t.area_frac:.4f}  aspect={t.aspect:.2f}")
    return 0 if tmpls else 2


if __name__ == "__main__":
    raise SystemExit(main())
