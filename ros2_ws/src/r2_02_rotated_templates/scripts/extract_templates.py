#!/usr/bin/env python3
"""Extract upright + sideways vial templates into data/templates/.

Usage (no ROS required):
  cd E:\\Desktop\\Jessica\\Camera\\ros2_ws\\src\\r2_02_rotated_templates
  python scripts/extract_templates.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2_02_rotated_templates.template_bank import (  # noqa: E402
    default_camera_root,
    default_templates_dir,
    extract_templates,
)


def main() -> int:
    camera = default_camera_root()
    ap = argparse.ArgumentParser(description="Extract rotated-template bank for r2_02")
    ap.add_argument("--camera-root", type=Path, default=camera)
    ap.add_argument("--split", default="train", choices=("train", "val", "test"))
    ap.add_argument("--out", type=Path, default=default_templates_dir())
    ap.add_argument("--max-upright", type=int, default=8)
    ap.add_argument("--max-sideways", type=int, default=8)
    ap.add_argument("--canon-long", type=int, default=80)
    args = ap.parse_args()

    tmpls = extract_templates(
        args.out,
        camera_root=args.camera_root,
        split=args.split,
        max_upright=args.max_upright,
        max_sideways=args.max_sideways,
        canon_long=args.canon_long,
    )
    n_up = sum(1 for t in tmpls if t.pose == "upright")
    n_side = sum(1 for t in tmpls if t.pose == "sideways")
    print(f"wrote {len(tmpls)} templates -> {args.out} (upright={n_up} sideways={n_side})")
    for t in tmpls:
        print(
            f"  {t.name}  pose={t.pose}  src={t.source_image}  "
            f"area={t.area_frac:.4f}  aspect={t.aspect:.2f}"
        )
    if n_side == 0:
        print("WARNING: zero sideways templates — check sideways_race/synth", file=sys.stderr)
        return 2
    return 0 if tmpls else 2


if __name__ == "__main__":
    raise SystemExit(main())
