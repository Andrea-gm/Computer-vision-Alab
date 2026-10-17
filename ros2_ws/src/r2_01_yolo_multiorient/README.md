# Round-2 Agent #01 — Multi-orientation YOLO vial detector

Package: `E:\Desktop\Jessica\Camera\ros2_ws\src\r2_01_yolo_multiorient\`

## Why Round-1 failed (do not copy)

| Issue | Evidence |
|-------|----------|
| `vial-live` is upright-biased | Raw labels: median GT `h/w ≈ 1.59`; only **1** wide box (`h/w < 0.85`) in 357 positives |
| Sideways ignored | `vial-live` @ conf 0.25 on `sideways_race/synth` sideways positives: **recall 0.0** (0/40), while test upright recall ≈ **0.96** |
| Classical fallback expands **downward only** | Assumes cap-on-top upright pose — brittle for on-side vials |

This package fixes coverage with **(1)** a multi-orient train set (rotated real stills + synth sideways), **(2)** fine-tune with `degrees=180`, and **(3)** optional 90° rotate-TTA at inference.

## Orientation subset split (eval)

Documented in `scripts/evaluate.py` / `data/multi_orient/orientation_manifest.json`:

| Subset | Definition |
|--------|------------|
| **upright** | Manifest `upright`, or filename `upright_*`, or GT median aspect `h/w ≥ 0.95` |
| **sideways** | Manifest `sideways`, or filename contains `sideways`, or stem `*_r90`/`*_r270`, or GT median `h/w < 0.95` |
| **negative** | Empty GT / `empty` / `hard_negative` synth |

Eval always scores `sideways_race/synth` so sideways numbers cannot be hidden.

## Layout

```
r2_01_yolo_multiorient/
  package.xml  setup.py  setup.cfg  requirements.txt  README.md
  config/params.yaml
  launch/vial_detect.launch.py
  models/best.pt                 # fine-tuned multi-orient weights
  models/eval_results.json       # upright vs sideways metrics
  data/multi_orient/             # prepared YOLO dataset + data.yaml
  scripts/
    prepare_multiorient_dataset.py
    train_multiorient.py
    evaluate.py
  r2_01_yolo_multiorient/
    vial_detector_node.py
    detector.py                  # YOLO + rotate-TTA (no upright classical)
    image_codec.py  msgs_compat.py  types.py
```

## Setup

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\r2_01_yolo_multiorient
pip install -r requirements.txt
```

## Train (multi-orient)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\r2_01_yolo_multiorient
python scripts\prepare_multiorient_dataset.py
python scripts\train_multiorient.py --epochs 25 --device cpu --prepare
# GPU if available:
# python scripts\train_multiorient.py --epochs 40 --device 0 --batch 8
```

Weights land in:

- `models/best.pt` (package)
- `E:\Desktop\Jessica\Camera\runs\r2-01-multiorient\weights\best.pt`

## Evaluate (upright vs sideways) — required before claiming success

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\r2_01_yolo_multiorient
python scripts\evaluate.py --model models\best.pt --conf 0.25
```

Prints separate **UPRIGHT** / **SIDEWAYS** / **ALL** blocks; writes `models/eval_results.json`.

## Build & run (ROS2)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws
colcon build --packages-select r2_01_yolo_multiorient --merge-install
# Windows RoboStack / pixi:
call install\setup.bat
ros2 launch r2_01_yolo_multiorient vial_detect.launch.py
# live:
ros2 launch r2_01_yolo_multiorient vial_detect.launch.py run_offline:=false
```

### Key params (`config/params.yaml`)

| Param | Default | Meaning |
|-------|---------|---------|
| `model_path` | `.../models/best.pt` | Multi-orient YOLO |
| `confidence_threshold` | `0.25` | Match Camera holdout gate |
| `dataset_path` | `.../vial_dataset/raw` | Offline folder |
| `run_offline` | `true` | Dataset vs live Image |
| `tta_rotate` | `true` | Extra 90° CW/CCW infer |

### Topics

| Topic | Type |
|-------|------|
| `/vial/detections` | `vision_msgs/Detection2DArray` (if installed) |
| `/vial/detections_json` | `std_msgs/String` JSON (always) |
| `/vial/annotated` | `sensor_msgs/Image` |

## Measured results (`models/eval_results.json`)

Baseline `vial-live` on sideways synth: **recall 0.0**. After multi-orient fine-tune + TTA @ conf 0.25:

| Subset | N | Precision | Recall | mean IoU |
|--------|--:|----------:|-------:|---------:|
| Upright | 109 | 0.863 | **0.927** | 0.759 |
| Sideways | 113 | 0.682 | **0.929** | 0.718 |
| Negative empty_fp | 51 | — | — | 45/51 FP frames |

See `SUPERVISOR_RETURN.md` for full supervisor packet.

## Honest failure modes

1. **Empty / hard-negative FP** — high at conf 0.25 + TTA; raise conf (~0.35) or set `tta_rotate: false` for live.
2. **Sideways precision** — recall is strong; extra boxes on clutter pull precision down (~0.68).
3. **Synth vs real sideways gap** — train/eval sideways lean on `sideways_race/synth` + 90° rotated stills.
4. **TTA cost** — ~3× forward passes (~0.8 s/frame CPU); disable for FPS.
5. **Transparent cup / glare / top-down** — RGB-only; top-down not optimized.
