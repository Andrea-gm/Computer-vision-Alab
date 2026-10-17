# Agent #05 — Blob detection + geometric priors

Classical vial locator using **OpenCV SimpleBlobDetector**, **MSER**, and
**geometric priors** (area, circularity, inertia, convexity, aspect) tuned to
white vial caps in `E:\Desktop\Jessica\Camera\vial_dataset`.

Package path (competition-corrected):

```
E:\Desktop\Jessica\Camera\ros2_ws\src\agent_05_blob_geometry\
```

## Dataset findings

| Item | Value |
|------|--------|
| Root | `Camera/vial_dataset` (`raw`, `review`, `split_70_10_20*`) |
| Labels | YOLO txt, class `0: vial` |
| Primary split | `split_70_10_20` — 261 / 37 / 75 train/val/test |
| Extra stills | `Camera/vial_photos` (11 labeled JPGs) |
| Appearance | Clear plastic body + **opaque white screw cap**; often hand-held; side and near-top views |

GT box stats (test sample): median area ~7k px, aspect w/h ~0.86. Caps are the
stable high-contrast cue; clear bodies are low-contrast and specular.

## Suitability verdict

**Moderate / conditional fit — not a top-tier detector for this set.**

| Strength | Weakness |
|----------|----------|
| White caps are blob/MSER-friendly | Clear bodies fragment or vanish |
| Fast (~30–40 ms/frame, CPU) | Side-view caps are short wide bands → circularity filter misses some |
| No training required; params tunable | Cluttered benches / bright walls → false positives |
| Cap→body expansion works when cap found | Expansion prior fails under extreme pose / scale |

Measured on `split_70_10_20` **test** (IoU≥0.5, conf≥0.70, defaults):

- Precision ≈ **0.39**, Recall ≈ **0.55**, image-level accuracy ≈ **0.55**
- Mean matched IoU ≈ **0.70**

Honest fire risk: **medium** — solid classical baseline, but YOLO (and similar)
will dominate accuracy on this labeled set. Keep this agent if you need a
training-free / explainable / low-deps path or a proposal generator.

## Algorithm

1. CLAHE + Gaussian blur; HSV white mask (`S < white_s_max`, `V > white_v_min`).
2. `SimpleBlobDetector` (bright) on CLAHE and mask — size / circularity / inertia / convexity.
3. MSER + connected components on white mask; score contours with geometric priors.
4. Expand each accepted **cap** region to a full-vial box (`expand_width_scale`, `expand_height_scale`).
5. Confidence gate + greedy IoU NMS.

## Build & run (ROS2)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws
# Activate your ROS2 / RoboStack env first
colcon build --packages-select agent_05_blob_geometry
.\install\setup.ps1   # or setup.bat

# Offline dataset playback
ros2 launch agent_05_blob_geometry vial_detect.launch.py run_offline:=true

# Live camera topic
ros2 launch agent_05_blob_geometry vial_detect.launch.py run_offline:=false
```

Topics:

- `/vial/detections` — `vision_msgs/Detection2DArray` (if `vision_msgs` installed)
- `/vial/detections_json` — `std_msgs/String` JSON (always)
- `/vial/annotated` — annotated `sensor_msgs/Image`

Params: `config/params.yaml` (blob thresholds, white mask, expansion scales).

## Evaluate (no ROS required)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\agent_05_blob_geometry
pip install -r requirements.txt
python scripts/evaluate.py --split test --conf 0.70
python scripts/evaluate.py --dataset E:\Desktop\Jessica\Camera\vial_photos
```

## Dependencies

- Python 3.10+
- `numpy`, `opencv-python`, `PyYAML`
- ROS2: `rclpy`, `sensor_msgs`, `std_msgs`, `launch`, `launch_ros`
- Optional: `vision_msgs` (falls back to JSON string)

## Limitations / failure modes

- Misses vials when the **cap is occluded**, out of frame, or not white.
- Bright paper / walls / gloves create **FP blobs**.
- Cap→body expansion assumes roughly upright side/top views; sideways vials
  (`sideways_race` style) degrade IoU.
- Not a substitute for a trained detector when labels and GPU are available.

## Layout

```
agent_05_blob_geometry/
  agent_05_blob_geometry/
    blob_detector.py      # core CV
    vial_detector_node.py # ROS2 node
    msgs_compat.py
    image_codec.py
    types.py
  config/params.yaml
  launch/vial_detect.launch.py
  scripts/evaluate.py
  package.xml
  setup.py
  requirements.txt
  README.md
```
