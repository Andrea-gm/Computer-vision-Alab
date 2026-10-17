# Agent 08 — Depth / geometry vial finder

ROS2 package: **point-cloud clustering + support-plane height + cylinder hypothesis** for clear vials with white screw caps on Orbbec RGB-D.

## Sensors / data (inspected)

| Asset | Path / topic | Role |
|-------|----------------|------|
| Dual Orbbec Gemini RGB-D | `/orbbec/cam0\|cam1/color/image_raw`, `.../depth/image_raw`, `.../camera_info` | Live metric depth (via `orbbec_dual_sync`) |
| Synth RGB-D eval set | `Camera/sideways_race/synth/*_{color.png,depth.npy}` | Offline recall/FP |
| Fitted vial mm | `Camera/vial_3d_dims.json` (+ package `config/vial_3d_dims.json`) | Cylinder size prior |
| Stereo extrinsics | `Camera/stereo_extrinsics.json` | **PLACEHOLDER** — not required for this single-view pipeline |

**Suitability: HIGH.** Real depth exists (not RGB-only). Transparent cup bodies create depth holes; the opaque white lid is the reliable metric seed. Pseudo-depth is only a ROS wiring fallback when depth is absent.

## Algorithm

1. Seed white caps (HSV).
2. Align / deproject depth → XYZ (mm) with `camera_info` or HFOV≈90° prior.
3. Fit a coarse support plane; keep seed neighborhood points in a Z-band / height band.
4. 3D PCA → cylinder axis; place Ø≈54 mm × H≈73 mm prior; score with clear-cup holes + size.
5. Publish 2D boxes + 3D poses; draw annotated image.

## Package path

```
E:\Desktop\Jessica\Camera\ros2_ws\src\agent_08_depth_geometry\
```

## Build / run (Windows RoboStack)

```bat
cd /d E:\Desktop\Jessica\Camera\ros2_ws
call _ros_env.cmd
colcon build --packages-select agent_08_depth_geometry
call install\setup.bat

ros2 launch agent_08_depth_geometry depth_geometry.launch.py
```

With live dual Orbbec publishers already up (`launch_dual_sync.cmd`), defaults subscribe to cam0. For cam1:

```bat
ros2 launch agent_08_depth_geometry depth_geometry.launch.py ^
  color_topic:=/orbbec/cam1/color/image_raw ^
  depth_topic:=/orbbec/cam1/depth/image_raw ^
  camera_info_topic:=/orbbec/cam1/color/camera_info
```

### Topics

| Direction | Topic | Type |
|-----------|-------|------|
| in | `color_topic` (param) | `sensor_msgs/Image` |
| in | `depth_topic` (param) | `sensor_msgs/Image` (`16UC1` mm or `32FC1`) |
| in | `camera_info_topic` | `sensor_msgs/CameraInfo` |
| out | `~/detections2d` | `vision_msgs/Detection2DArray` if installed, else `Float32MultiArray` |
| out | `~/detections_flat` | `std_msgs/Float32MultiArray` (always) |
| out | `~/poses` | `geometry_msgs/PoseArray` (if `geometry_msgs` present) |
| out | `~/annotated` | `sensor_msgs/Image` (BGR) |

`detections_flat` layout per detection (14 floats):  
`x, y, w, h, score, cx_mm, cy_mm, cz_mm, ax, ay, az, radius_mm, height_mm, orient(0=sideways,1=upright)`.

### Params

See `config/params.yaml`. Notable:

- `allow_pseudo_depth` (default `true`) — RGB→weak Z when depth missing (**not metric**).
- `require_depth` — if `true`, skip frames without depth (disables pseudo).

## Offline evaluate (no ROS)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\agent_08_depth_geometry
python evaluate.py
python evaluate.py --limit 20
```

Writes `results/eval_latest.json`.

Latest offline run on `sideways_race/synth` (76 samples): **recall@0.5 = 0.675 (27/40)**, **fp_count = 6** (rate 0.167), **mean IoU = 0.646**.

## Limitations

- Clear plastic bodies break Orbbec depth; ranging is **cap-first**, not full-surface ICP.
- Single-view cylinder fit; dual-cam triangulation needs real `stereo_extrinsics.json` (still PLACEHOLDER).
- Pseudo-depth is a wiring stub only — do not trust metric poses without real depth.
- `vision_msgs` / `geometry_msgs` may be absent from the minimal RoboStack pin; flat array + annotated image always work. Install extras if you need native Detection2DArray / PoseArray types.
