# Round-2 Agent #03 — YOLOv8-OBB (oriented vials)

Package path: `E:\Desktop\Jessica\Camera\ros2_ws\src\r2_03_obb\`

Detects the clear plastic vial (white screw cap) **upright and sideways** with
**oriented bounding boxes** (rotated rectangles). Round-1 AABB detectors fail on
side views because a horizontal / tilted vial forces a large axis-aligned box
full of background; OBB aligns with the vial long axis so IoU stays high and
the network learns orientation instead of fighting aspect priors.

## Why OBB helps sideways

| Failure mode (AABB) | OBB behavior |
|---------------------|--------------|
| Sideways vial → wide box swallows table clutter | Tight rotated rect along lid→cup |
| Angled (±15–30°) hard samples → AABB IoU collapses | `minAreaRect` / YOLO-OBB tracks θ |
| Upright-trained aspect gate rejects w≫h | Orientation head / sideways OBB path |

Labels: classic YOLO `xywh` → four-corner OBB via dataset geometry;
`sideways_race/synth` hard samples use `angle=` notes to recover length/diameter.
Sideways stills oversampled **4×** in `prepare_obb_dataset.py`.

### Offline metrics (conf 0.25, OBB-IoU ≥ 0.5)

**Ship path:** `models/best.pt` (YOLOv8-OBB) **+ vial-live AABB→OBB upright fuse**
+ sideways OBB bridge. No long retrain. See `eval_contract.json`.

| Split | Before | After | Target |
|-------|-------:|------:|-------:|
| **Upright synth** (`upright_*`) | 0.30 | **1.00** (6/6) | ≥0.70 |
| **Sideways synth** | 0.95 | **0.925** (37/40) | ≥0.85 |
| **vial_dataset test upright** | — | **0.96** (71/74) | ≥0.70 |

Backend: `yolov8_obb+upright_aabb+sideways`. Quick check:

```powershell
python scripts/quick_eval_contract.py
```

## Layout

```
r2_03_obb/
  package.xml  setup.py  setup.cfg  requirements.txt  README.md
  config/params.yaml
  launch/vial_detect.launch.py
  scripts/prepare_obb_dataset.py
  scripts/train_obb.py
  scripts/evaluate.py          # offline upright vs sideways
  models/best.pt               # after train_obb.py
  data/obb_split/              # prepared OBB labels
  r2_03_obb/
    vial_detector_node.py
    obb_detector.py            # YOLOv8-OBB → AABB-refine → classical OBB
    obb_geometry.py
    image_codec.py
    msgs_compat.py             # theta = angle_rad on Detection2D
```

## Dependencies

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\r2_03_obb
pip install -r requirements.txt
```

## Prepare + train OBB (recommended)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\r2_03_obb
python scripts/prepare_obb_dataset.py --sideways-mult 4
python scripts/train_obb.py --prepare --epochs 25 --batch 8
# writes models/best.pt
```

## Offline evaluate (no ROS)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\r2_03_obb
python scripts/evaluate.py --conf 0.25
# optional: --model models\best.pt
```

Reports **upright** vs **sideways** image recall @ OBB-IoU≥0.5, mean OBB IoU,
AABB envelope IoU (Round-1 comparison), and empty FP rate.
Writes `eval_results.json`.

Without trained OBB weights the detector falls back to **AABB YOLO
(vial-live) → minAreaRect refine** plus **classical PCA/minAreaRect OBB** so
eval still runs offline.

## Build & run (ROS2)

```bash
cd E:/Desktop/Jessica/Camera/ros2_ws
colcon build --packages-select r2_03_obb
# Linux: source install/setup.bash
# Windows RoboStack: call install\setup.bat

ros2 launch r2_03_obb vial_detect.launch.py
ros2 launch r2_03_obb vial_detect.launch.py run_offline:=false
```

### Topics

| Topic | Type | Notes |
|-------|------|-------|
| `/vial/detections` | `vision_msgs/Detection2DArray` | `bbox.center.theta` = angle (rad), size = (long, short) |
| `/vial/detections_json` | `std_msgs/String` | Always; includes corners + orientation |
| `/vial/annotated` | `sensor_msgs/Image` | Drawn OBB polylines |

## Backend priority

1. `models/best.pt` (YOLOv8-OBB) **always fused** with vial-live AABB→OBB (upright)
2. `sideways_race` I_hard_robust → OBB (sideways)
3. Classical upright / PCA OBB last resort

## Limitations

- AABB upright fuse can raise empty/hard-neg FP on synth clutter; contract
  metrics prioritize upright+sideways recall.
- Packaged OBB weights are short CPU FT; longer GPU FT optional later.
- Transparent cup / live depth gates not in the ROS node.
- Windows Torch/OpenMP: validate with `scripts/quick_eval_contract.py` first.
