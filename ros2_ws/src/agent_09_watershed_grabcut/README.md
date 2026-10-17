# Agent #09 — Watershed / GrabCut vial segmentation

Automated (no-click) **watershed + GrabCut** instance segmentation for clear plastic vials with white screw caps. ROS2 `ament_python` package under `ros2_ws/src`.

## Dataset findings

| Item | Value |
|------|--------|
| Location | `E:/Desktop/Jessica/Camera/vial_dataset/` |
| Labels | YOLO txt, class `0` = vial |
| `raw` | ~373 images (357 nonempty labels, 16 empty) |
| Split used | `split_70_10_20_latest2` (train/val/test) |
| Instances | Almost always **one vial per frame** (no labeled racks) |
| Appearance | Transparent body + opaque white lid; desk clutter, hands, plain/white BG |

## Suitability verdict

**Marginal / high fire-risk for this dataset.**

- GrabCut/watershed need reliable FG/BG seeds. The **white lid** is the only stable seed; the clear body inherits desk color and leaks through graph-cut color models.
- Cluttered desk scenes produce many bright low-saturation false lids (paper, monitors, bags).
- Close-ups on white/plain backgrounds blow out the cap mask → center-prior fallback is crude.
- **Better families for this data:** fine-tuned YOLO (agent 01) or supervised segmentation. This package is the honest classical graph-cut baseline.

Still implemented end-to-end with automatic markers (distance-transform peaks + center prior), not interactive clicks.

## Algorithm

1. Adaptive white-cap mask (high V, low S; threshold auto-picked for ~lid fill fraction).
2. Filter compact lid-like connected components.
3. Distance-transform local maxima → watershed markers (instance seeds / rack-ready).
4. Grow clear-cup extent from each seed (edge + low-sat row scan).
5. GrabCut refine (rect init, then mask init with sure-FG on the lid).
6. Score (lid-top white fraction, aspect, center prior, border penalty) → NMS → boxes.

## Paths

```
E:\Desktop\Jessica\Camera\ros2_ws\src\agent_09_watershed_grabcut\
  agent_09_watershed_grabcut\
    segmenter.py           # core CV
    vial_detector_node.py  # ROS2 node
    msgs_compat.py         # vision_msgs or JSON fallback
    image_codec.py
  config/params.yaml
  launch/vial_detect.launch.py
  scripts/evaluate.py
  package.xml, setup.py, requirements.txt, README.md
```

## Build & run (ROS2)

From the workspace (Windows RoboStack / pixi as used by this repo):

```bat
cd E:\Desktop\Jessica\Camera\ros2_ws
call activate_ros2.cmd
colcon build --packages-select agent_09_watershed_grabcut
call install\setup.bat
ros2 launch agent_09_watershed_grabcut vial_detect.launch.py
```

Live camera (subscribe mode):

```bat
ros2 launch agent_09_watershed_grabcut vial_detect.launch.py run_offline:=false
```

Topics:

| Topic | Type |
|-------|------|
| `/vial/detections` | `vision_msgs/Detection2DArray` (if installed) |
| `/vial/detections_json` | `std_msgs/String` JSON (always) |
| `/vial/annotated` | `sensor_msgs/Image` |

Tune markers / GrabCut in `config/params.yaml`: `confidence_threshold`, `grabcut_iters`, `use_center_prior`, `max_side`.

## Evaluate (no ROS)

```bat
cd E:\Desktop\Jessica\Camera\ros2_ws\src\agent_09_watershed_grabcut
python scripts/evaluate.py --split val
python scripts/evaluate.py --split test --iou 0.3
python scripts/evaluate.py --dataset E:\Desktop\Jessica\Camera\vial_dataset\raw
```

## Dependencies

- Python: `numpy`, `opencv-python`, `PyYAML` (`requirements.txt`)
- ROS2: `rclpy`, `sensor_msgs`, `std_msgs`, `launch`, `launch_ros`
- Optional: `vision_msgs` (else JSON-only detections)

## Measured metrics (honest)

On `split_70_10_20_latest2` val with defaults (`conf=0.40`, `max_dets=1`, IoU match 0.3):

| Metric | Val |
|--------|-----|
| Precision | ~0.21 |
| Recall | ~0.21 |
| Mean matched IoU | ~0.40 |
| IoU≥0.5 recall | ~0 |
| Inference | ~1–2 s/frame (GrabCut-dominated) |

These numbers confirm the suitability verdict: shippable classical baseline, not competitive with YOLO on this data.

## Limitations / fire risk

- Unsupervised graph-cut **cannot match** a fine-tuned detector on transparent vials in clutter.
- Weak IoU≥0.5; false positives on bright clutter remain even with `max_dets=1`.
- Empty frames / all-white backgrounds: few or no seeds.
- Multi-vial racks are supported structurally (DT peaks) but **absent from current GT** — raise `max_dets` if racks appear.
- GrabCut is slow (~1 s/frame at `grabcut_iters=2`).
