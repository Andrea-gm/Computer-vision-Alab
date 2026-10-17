# Cleanup Report

Cleanup date: 2026-10-12

## Result

- Before: **5.336 GiB** (5,729,283,013 bytes)
- After: **3.809 GiB** (4,089,824,173 bytes)
- Freed: **1.527 GiB** / **1.639 GB** (1,639,458,840 bytes)

## Biggest offenders found

- `Camera/ros2_ws/`: 3.661 GiB before cleanup
  - `.pixi/`: 2.847 GiB before cache cleanup; retained because it is the active ROS environment
  - `src/`: 828.4 MiB; retained as source and package assets
- `Camera/weight_race/`: 949.1 MiB
- `Camera/runs/`: 435.2 MiB
- `Camera/sideways_race/synth/`: 132.2 MiB
- `Camera/vial_dataset/`: 131.7 MiB

## Deleted

- `Camera/weight_race/` — 949.1 MiB of completed old experiment runs. Searches found only self-references inside that folder; live detector and ROS parameters do not use it.
- Python `__pycache__/`, `.pytest_cache/`, and `*.pyc` — 215.3 MiB across 1,489 cache directories. These regenerate automatically.
- `Camera/runs/train-race/{latest-full-split,latest-full-split-2,r3-04-mild-live,r3-weakangle-oversample}/` — 176.7 MiB of stale/non-winning training outputs.
- `Camera/sideways_race/synth/` — 132.2 MiB of deterministic synthetic RGB-D data. Regenerate with `python sideways_race/eval_sideways.py --regen`.
- `Camera/runs/{vial,vial-2,vial-3,vial-4,bon-vid-06,detect}/` — 83.2 MiB of old scratch runs not referenced by live weight resolution.
- `Camera/ros2_ws/log/`, `build/`, and `_smoke_logs/` — 4.6 MiB of regenerable ROS build/log output.
- Temporary `*.log`, `*.tmp`, crash-dump, and remaining bytecode files — 2.5 MiB.

## Kept

- `Camera/depthcamera.py`, all `Camera/ros2_ws/src/`, and sideways-race code/methods.
- `Camera/ros2_ws/.pixi/`, `pixi.toml`, and `pixi.lock`; this is the active ROS environment.
- `Camera/ros2_ws/install/` (1.1 MiB), so the existing installed workspace remains available.
- `Camera/vial_dataset/` and `vial_photos/`; these are training/source data.
- Live and fallback weights:
  - `runs/vial-live/weights/best.pt`
  - `runs/bon-winner/weights/best.pt`
  - `runs/train-race/{r1-winner,r2-winner,r3-winner}/weights/best.pt`
  - `runs/r2-01-multiorient/` (referenced by the R2-01 flow)
  - `runs/vial-ft/` (training/promote staging)
- ROS package model assets and `r2_10_fusion_multiorient/models/error_frames`; they are under source packages and were not large enough to justify risking current flows.
- `.git/` and unrelated top-level project files.

## Rebuild ROS

From `Camera\ros2_ws`:

```bat
build.cmd
```

For only the current R2-10 package:

```bat
build_r2_10.cmd
```

Only `build/` and logs were cleared; `install/` was retained. Either build command recreates build output and refreshes the install as needed.
