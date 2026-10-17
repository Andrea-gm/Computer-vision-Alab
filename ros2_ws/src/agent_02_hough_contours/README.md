# Agent #02 — HoughCircles / Contour-Ellipse vial detector

ROS2 package under `E:\Desktop\Jessica\Camera\ros2_ws\src\agent_02_hough_contours\`.

Competition package: **pure classical OpenCV** geometry for the clear plastic vial
(opaque white screw cap) in `E:\Desktop\Jessica\Camera`. No deep learning.

## Dataset summary

| Signal | Finding |
|--------|---------|
| Path | `vial_dataset/raw` (+ `split_70_10_20`) |
| Scale | **373** labeled stills (**357** positives, **16** empty negatives) |
| Labels | YOLO format `class cx cy w h` (normalized), class `0: vial` |
| Object | Clear cup + **opaque white screw cap**; GT box = lid through cup bottom |
| Views | Desk-top (near-top / ~45°) and hand-held side angles; cluttered office/lab |
| Lighting | Indoor overhead; specular glare on cap and clear body |
| Cap geometry | Circular when near top-down; **elliptical** when off-axis |
| Box stats | Median norm w/h ≈ **0.12 × 0.19**, mean aspect (w/h) ≈ **0.64** |

Hard cases for classical CV: white-on-white surfaces, low-res phone crops, hand
occlusion of the cup, clear-body edges that vanish into the desk.

## Suitability verdict: **MODERATE — keep classical, adapt beyond pure Hough**

| Signal | Finding |
|--------|---------|
| Cap cue | Strong: bright, low-saturation blob with closed contour |
| Pure `HoughCircles` | **Weak alone** — many lids are ellipses; clutter produces circle FPs |
| Contours + circularity | **Good** on desk-top Orbbec frames when white-prefiltered |
| Ellipse fit | **Required adaptation** (`cv2.fitEllipse`) for off-axis views |
| Full-vial box | Cap-only geometry → expand downward to match GT (cup body) |

**Why not fire / replace with DL:** The assigned algorithm family is viable as a
cap-first classical pipeline. We document honest limits vs YOLO rather than
pretending Hough alone solves transparent cups.

**Adaptation chosen (still classical):** HSV white prefilter → external contours →
circularity + ellipse-aspect filters → optional HoughCircles with white-disk
support → NMS → downward expansion. Watershed was considered for touching
clutter but is unnecessary for the common single-vial desk scenes.

## Layout

```
ros2_ws/src/agent_02_hough_contours/
  package.xml
  setup.py
  setup.cfg
  requirements.txt
  README.md
  resource/agent_02_hough_contours
  config/params.yaml
  launch/vial_detect.launch.py
  scripts/evaluate.py
  agent_02_hough_contours/
    vial_detector_node.py   # ROS2 node (offline + Image sub)
    detector.py             # Hough + contour/ellipse core
    image_codec.py          # sensor_msgs Image ↔ numpy (no cv_bridge)
    msgs_compat.py          # vision_msgs or JSON String fallback
```

## Dependencies

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\agent_02_hough_contours
pip install -r requirements.txt
```

**ROS2 assumptions**

- **Linux:** ROS 2 Humble/Jazzy, `rclpy`, optional `ros-$ROS_DISTRO-vision-msgs`.
- **Windows:** RoboStack / Pixi env used by `Camera/ros2_ws` (same as `orbbec_dual_sync`).
- **vision_msgs:** Preferred `Detection2DArray` on `/vial/detections`. If missing,
  detections still publish as `std_msgs/String` JSON on `/vial/detections_json`.

## Build & run (ROS2)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws
.\build.ps1
# or: colcon build --packages-select agent_02_hough_contours
# then activate install overlay (build.ps1 / source_install.ps1)

ros2 launch agent_02_hough_contours vial_detect.launch.py
# live camera:
ros2 launch agent_02_hough_contours vial_detect.launch.py run_offline:=false
```

Standalone after overlay / `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "E:\Desktop\Jessica\Camera\ros2_ws\src\agent_02_hough_contours;$env:PYTHONPATH"
ros2 run agent_02_hough_contours vial_detector_node --ros-args --params-file `
  E:\Desktop\Jessica\Camera\ros2_ws\src\agent_02_hough_contours\config\params.yaml
```

### Node parameters

| Param | Default | Meaning |
|-------|---------|---------|
| `image_topic` | `/camera/color/image_raw` | Live `sensor_msgs/Image` |
| `dataset_path` | `.../vial_dataset/raw` | Offline folder |
| `run_offline` | `true` | Dataset vs subscribe |
| `visualize` | `true` | Publish annotated image |
| `dp` | `1.2` | Hough inverse accumulator ratio |
| `min_dist_frac` | `0.08` | Hough minDist as fraction of min(H,W) |
| `canny` | `110` | Hough `param1` (Canny high) |
| `accum` | `28` | Hough `param2` (accumulator threshold) |
| `min_radius_frac` / `max_radius_frac` | `0.02` / `0.12` | Circle radius bounds |
| `min_circularity` | `0.45` | Contour circularity floor |
| `min_ellipse_ar` | `0.45` | Ellipse axis ratio floor |
| `v_min` / `s_max` | `170` / `70` | White HSV prefilter |
| `score_thresh` | `0.40` | Drop weak candidates |
| `max_dets` | `2` | Cap after NMS |

### Topics

| Topic | Type | Notes |
|-------|------|-------|
| `/vial/detections` | `vision_msgs/Detection2DArray` | If `vision_msgs` installed |
| `/vial/detections_json` | `std_msgs/String` | Always (JSON) |
| `/vial/annotated` | `sensor_msgs/Image` | BGR annotated |

Logs rolling **FPS** and **detection count** every `log_every_n` frames.

## Evaluate (no ROS required)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\agent_02_hough_contours
python scripts/evaluate.py --split test
python scripts/evaluate.py --split raw --max-images 50
python scripts/evaluate.py --dataset E:\Desktop\Jessica\Camera\vial_dataset\raw
```

Reports precision / recall / mean IoU / image accuracy when YOLO `.txt` GT exists;
writes `eval_results.json`.

## Holdout snapshot (classical baseline)

`python scripts/evaluate.py --split test` (IoU ≥ 0.5, defaults in `params.yaml`):

| Metric | Approx. |
|--------|---------|
| Precision | ~0.25–0.35 (white clutter FPs) |
| Recall | ~0.35–0.45 |
| Mean IoU (matched) | ~0.65–0.70 |
| Infer | ~30 ms / frame (~33 FPS CPU) |

Numbers vary with `score_thresh` / `max_dets`. This is a geometry baseline, not YOLO.

## Limitations

- **Transparent cup** is not segmented; the box is a **cap-anchored expansion** and
  can miss tall/sideways cups or over-cover hands.
- **Off-axis / elliptical lids** weaken pure Hough; ellipse-contour branch helps but
  still fails on extreme side views and tiny phone crops.
- **White clutter** (tissues, paper, monitor stands, glare) produces false caps →
  empty-negative FP risk.
- **No depth / clear-cup gates** (unlike live `depthcamera.py` BoN gates).
- Expect **lower precision/recall than fine-tuned YOLO** on this cluttered desk set;
  classical is a geometry baseline, not the production detector.
