# Round-2 Agent 07 — Depth any-pose vial finder

ROS2 package: **RGB-D cylinder / PCA geometry for vials standing or lying**.
Major axis is estimated from a 3D cluster (vertical **or** horizontal **or**
angled). Orientation-aware RGB fallback when depth is absent; **YOLO RGB**
when `runs/vial-live/weights/best.pt` is available (real upright stills).

Round-1 (`agent_08_depth_geometry`) snapped the body axis to camera X/Y and
was race-tuned for sideways-only. This package keeps free 3D PCA when it
agrees with the clear-cup / dark-body direction, measures box length from
extent (tighter upright IoU), and recovers angled hard cases (including
wrong lid-Z / `hard_sideways_right`).

## Sensors / data

| Asset | Path / topic | Role |
|-------|----------------|------|
| Dual Orbbec Gemini RGB-D | `/orbbec/cam0\|cam1/color/image_raw`, `.../depth/image_raw`, `.../camera_info` | Live metric depth (`orbbec_dual_sync`) |
| Synth RGB-D | `Camera/sideways_race/synth/*_{color.png,depth.npy}` | Offline any-pose / race eval |
| Real upright stills | `Camera/vial_dataset/split_70_10_20` | RGB-only upright IoU eval (no depth) |
| YOLO weights (optional) | `Camera/runs/vial-live/weights/best.pt` | RGB upright / still fallback |
| Fitted vial mm | `Camera/vial_3d_dims.json` (+ `config/vial_3d_dims.json`) | Cylinder prior Ø≈54 × H≈73 mm |

**Suitability: HIGH** for any-pose when depth is present. Transparent cups
create depth holes; the opaque white lid seeds the cluster. Real stills
without depth use YOLO RGB (classical alone is weak on white-wall scenes).

## Pose estimation approach

1. Seed white caps (HSV); optional dim-lid second pass for hard synth.
2. Align / deproject depth → XYZ (mm).
3. 8-direction **clear-depth + dark-RGB** body scan → image body direction
   (cardinal clear-cup wins when decisive).
4. Neighborhood Z-band cluster → **3D PCA major axis** (kept free in 3D when
   elongated and aligned with the body cue — not forced upright).
5. Classify `upright` / `sideways` / `angled` from axis; place cylinder prior;
   project AABB. Box length prefers measured clear/dark extent; **wrong-Z**
   lids (`length_px ≪ cap`) get a cap-ratio length rescue for hard angled
   samples.
6. RGB-only fallback: **YOLO** if weights resolve, else classical
   orientation-aware 2D (still-dataset / wiring). No RGB fallback after a
   real-depth clear-cup reject.

## Package path

```
E:\Desktop\Jessica\Camera\ros2_ws\src\r2_07_depth_anypose\
```

## Build / run (Windows RoboStack)

```bat
cd /d E:\Desktop\Jessica\Camera\ros2_ws
call _ros_env.cmd
colcon build --packages-select r2_07_depth_anypose --merge-install
call install\setup.bat

ros2 launch r2_07_depth_anypose depth_anypose.launch.py
```

Cam1:

```bat
ros2 launch r2_07_depth_anypose depth_anypose.launch.py ^
  color_topic:=/orbbec/cam1/color/image_raw ^
  depth_topic:=/orbbec/cam1/depth/image_raw ^
  camera_info_topic:=/orbbec/cam1/color/camera_info
```

### Topics

| Direction | Topic | Type |
|-----------|-------|------|
| in | `color_topic` | `sensor_msgs/Image` |
| in | `depth_topic` | `sensor_msgs/Image` (`16UC1` mm or `32FC1`) |
| in | `camera_info_topic` | `sensor_msgs/CameraInfo` |
| out | `~/detections2d` | `vision_msgs/Detection2DArray` or `Float32MultiArray` |
| out | `~/detections_flat` | `std_msgs/Float32MultiArray` (always) |
| out | `~/poses` | `geometry_msgs/PoseArray` (if available) |
| out | `~/annotated` | `sensor_msgs/Image` (BGR) |

`detections_flat` layout per detection (14 floats):
`x, y, w, h, score, cx_mm, cy_mm, cz_mm, ax, ay, az, radius_mm, height_mm, orient`
where orient: `0=sideways, 1=upright, 2=angled, 3=topdown`.

### Params

See `config/params.yaml`. Notable:

- `allow_rgb_fallback` (default `true`) — 2D path when depth missing
- `use_yolo_rgb` (default `true`) — prefer YOLO weights for RGB stills
- `require_depth` — skip frames without depth
- `min_score` (default `0.25`)

## Offline evaluate (no ROS)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\r2_07_depth_anypose
python evaluate.py
python evaluate_vial_dataset.py
python evaluate_vial_dataset.py --no-yolo --out results\eval_vial_dataset_upright_classical.json
```

Writes `results/eval_latest.json`, `results/eval_vial_dataset_upright.json`.
See `results/SUMMARY.md` for a compact table.

### Latest metrics — synth (`sideways_race/synth`, 76 samples)

| Split | recall@0.5 | FP | mean IoU |
|-------|------------|----|----------|
| **Race** (sideways positives; sideways/angled preds only) | **0.975 (39/40)** | **0** | **0.675** |
| **Any-pose** | **0.865 (45/52)** | **0** | **0.697** |

Per-kind (any-pose):

| kind | recall | mean IoU |
|------|--------|----------|
| sideways_left | 1.000 (12/12) | 0.727 |
| sideways_right | 0.917 (11/12) | 0.682 |
| hard_sideways_left | 1.000 (8/8) | 0.630 |
| hard_sideways_right | **1.000 (8/8)** | 0.632 |
| upright | **1.000 (6/6)** | 0.843 |
| topdown | 0.000 (0/6) | — |

**hard_sideways_right before → after:** **0.250 → 1.000** (see `results/SUMMARY.md`).

Round-1 reference (`agent_08`): race recall **0.675**, fp_count **6**, mean IoU **0.646**.

### Latest metrics — vial_dataset upright (real stills, RGB only)

| Mode | recall@0.5 | precision@0.5 |
|------|------------|---------------|
| **YOLO RGB** (default) | **0.938** | **0.744** |
| Classical RGB (`--no-yolo`) | 0.365 | 0.106 |

Target recall ≥ 0.70 on upright real stills: **met** via YOLO RGB path.

## Limitations

- Clear plastic bodies break Orbbec depth; ranging is **cap-first**.
- **Top-down** (looking into the mouth) is not emitted by default — solid white
  squares collide with hard-negative lid clutter without a body cue.
- Classical RGB alone fails when white lids merge into white walls; YOLO
  weights are required for usable upright still recall.
- RGB fallback is not metric; prefer real depth for 3D pose.
- Single-view cylinder; dual-cam triangulation needs real stereo extrinsics.
- `vision_msgs` / `geometry_msgs` may be absent from minimal RoboStack; flat
  array + annotated image always work.
