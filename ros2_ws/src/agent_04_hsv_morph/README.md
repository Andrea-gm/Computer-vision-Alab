# Agent #04 — HSV/LAB + morphology vial detector

ROS2 package under `E:\Desktop\Jessica\Camera\ros2_ws\src\agent_04_hsv_morph\`.

Classical pipeline: **HSV + LAB white-cap segmentation → morphological open/close →
connected components / contour filters → optional Canny edge body expand** for the
clear plastic vial (opaque white screw cap) in `E:\Desktop\Jessica\Camera`.

## Dataset summary (inspected)

| Signal | Finding |
|--------|---------|
| Source | `vial_dataset/raw` (~373 stills) + `split_70_10_20` (261 / 37 / 75) |
| Class | Single class `0: vial` (YOLO `cx cy w h` normalized) |
| Cap | Opaque **white** screw top — near-achromatic (train sample: **~97%** of top-bbox pixels have HSV `S < 40`, high `V`) |
| Body | **Clear / transparent** cup — no stable hue; inherits desk / hand / hoodie |
| Background | Cluttered lab desk (gray surface, cables, monitors), white walls, hands, skin |
| Aspect | Mid-field vials ~ upright boxes; median `bh/bw ≈ 1.15` (p10–p90 ≈ 1.06–1.34) |

**Calibrated defaults** (from train crop stats + val grid search):

- HSV white: `S ∈ [0, 40]`, `V ∈ [195, 255]` (hue unused / full range)
- LAB white: `L ≥ 190`, `|a−128| ≤ 12`, `|b−128| ≤ 12`
- Morph: ellipse open/close `k=5`
- Edge assist under cap + aspect / area / circularity filters

## Suitability verdict: **CONDITIONAL — keep with edge/shape priors**

Hue-based color segmentation alone is **unreliable** for this object:

1. Cap is white → no distinctive hue; only low-S / high-V (or high-L) works.
2. Clear cup has **no color identity**; pure color masks miss the body.
3. White walls, paper, watch bands, specular glare create **false positives**.

**Still in-family strengthening (implemented):**

- Dual HSV∪LAB white mask (not hue banding)
- Morphological open/close to clean speckles / fill cap ribs
- Connected components + contour fill / circularity / size gates
- Canny edges under the cap to extend a vial-height box (shape prior)
- NMS + `max_dets` + tunable `min_score`

**Honest risk:** Precision is the weak point (desk clutter / white distractors).
This method is a strong **classical baseline / demo**, not a YOLO replacement for
production vial-live weights.

## Layout

```
ros2_ws/src/agent_04_hsv_morph/
  package.xml
  setup.py
  setup.cfg
  requirements.txt
  README.md
  resource/agent_04_hsv_morph
  config/params.yaml          # HSV/LAB ranges + morph + filters (tunable)
  launch/vial_detect.launch.py
  scripts/evaluate.py
  agent_04_hsv_morph/
    vial_detector_node.py     # ROS2 node
    detector.py               # HSV/LAB + morph + CC + edges
    image_codec.py            # sensor_msgs Image ↔ numpy (no cv_bridge)
    msgs_compat.py            # vision_msgs or JSON String fallback
```

## Dependencies

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\agent_04_hsv_morph
pip install -r requirements.txt
```

**ROS2 assumptions**

- **Linux:** ROS 2 Humble/Jazzy, `rclpy`, optional `ros-$ROS_DISTRO-vision-msgs`.
- **Windows:** RoboStack / conda ROS env used by `Camera/ros2_ws` (same as `orbbec_dual_sync`).
- **vision_msgs:** Preferred `Detection2DArray` on `/vial/detections`. If missing,
  detections still publish as `std_msgs/String` JSON on `/vial/detections_json`.

## Build & run (ROS2)

```bash
cd E:/Desktop/Jessica/Camera/ros2_ws
colcon build --packages-select agent_04_hsv_morph
# Linux:
source install/setup.bash
# Windows RoboStack:
# call install\setup.bat

ros2 launch agent_04_hsv_morph vial_detect.launch.py
# live camera:
ros2 launch agent_04_hsv_morph vial_detect.launch.py run_offline:=false
```

### Topics

| Topic | Type | Notes |
|-------|------|-------|
| `/vial/detections` | `vision_msgs/Detection2DArray` | if `vision_msgs` installed |
| `/vial/detections_json` | `std_msgs/String` | always |
| `/vial/annotated` | `sensor_msgs/Image` | when `visualize:=true` |
| `/camera/color/image_raw` | `sensor_msgs/Image` | live subscribe (default) |

Tune HSV ranges live via params, e.g. `hsv_s_max:=35 hsv_v_min:=200`.

## Offline evaluate (no ROS)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\agent_04_hsv_morph
python scripts/evaluate.py --split val
python scripts/evaluate.py --split test
```

Smoke metrics at IoU≥0.5 with default `params.yaml` (classical baseline):

| Split | Precision | Recall | mean IoU (matched) | ~ms/frame |
|-------|-----------|--------|--------------------|-----------|
| val   | ~0.28     | ~0.58  | ~0.72              | ~16       |
| test  | ~0.27     | ~0.62  | ~0.74              | ~15       |

Precision is the main failure mode (white distractors). Raise `min_score` / tighten
`hsv_v_min` for fewer FPs; lower them for recall.

## Limitations

- Clear-cup IoU depends on edge expansion under the cap; heavy occlusion / motion blur hurts.
- White distractors (walls, paper, silicone watch band) raise FP unless score/aspect gates are tight — which then drops recall.
- Lighting / white-balance shifts move the V/L thresholds; retune `params.yaml` per camera.
- Cap-down or extreme close-ups change aspect heuristics.
- No depth / stereo — RGB classical only.
