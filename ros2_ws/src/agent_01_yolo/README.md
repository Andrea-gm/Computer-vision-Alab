# Agent #01 — YOLO (Ultralytics) vial detector

ROS2 package under `E:\Desktop\Jessica\Camera\ros2_ws\src\agent_01_yolo\`.

Competition package: one-stage **YOLOv8/v11** detector for the clear plastic vial
(opaque white screw cap) in `E:\Desktop\Jessica\Camera`.

## Suitability verdict: **STRONG FIT — keep YOLO**

| Signal | Finding |
|--------|---------|
| Labels | **YOLO format** already (`class cx cy w h` normalized), class `0: vial` |
| Scale | **373** labeled stills in `vial_dataset/raw` (357 positives, 16 empty negatives) |
| Split | `split_70_10_20` — 261 / 37 / 75 (seed 42), `data.yaml` ready for Ultralytics |
| Weights | Fine-tuned **YOLOv8s** live at `runs/vial-live/weights/best.pt` (train-race lineage) |
| Domain | Transparent cup + white cap — COCO-pretrained alone is weak; **domain FT is required** and already exists |

**Why not pivot:** This repo’s production path (`depthcamera.py`, train races, BoN gates)
is already YOLO. Segmentation (Mask R-CNN/SAM) was explicitly deferred in `plan.md`.
Classical white-cap is only a **demo fallback** when weights fail to load.

**Risks (honest):** Clear cups still confuse RGB-only boxes under glare / weak angles;
live system adds depth “clear-cup” gates that this ROS node does **not** reimplement.
Small empty-negative count → empty FP risk if conf is too low. Weak-camera angle gap
documented in Camera `plan.md` remains an open data issue, not an algorithm-family issue.

## Layout

```
ros2_ws/src/agent_01_yolo/
  package.xml
  setup.py
  setup.cfg
  requirements.txt
  README.md
  resource/agent_01_yolo
  config/params.yaml
  launch/vial_detect.launch.py
  scripts/evaluate.py
  scripts/train_finetune.py
  agent_01_yolo/
    vial_detector_node.py   # ROS2 node
    detector.py             # Ultralytics + weight resolve
    classical_fallback.py   # white-cap demo path
    image_codec.py          # sensor_msgs Image ↔ numpy (no cv_bridge)
    msgs_compat.py          # vision_msgs or JSON String fallback
```

## Dependencies

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\agent_01_yolo
pip install -r requirements.txt
```

**ROS2 assumptions**

- **Linux (primary):** ROS 2 Humble/Jazzy, `rclpy`, optional `ros-$ROS_DISTRO-vision-msgs`.
- **Windows:** RoboStack / conda ROS env used by `Camera/ros2_ws` (same as `orbbec_dual_sync`).
  Torch + OpenMP can conflict with OpenCV on Windows (see existing `detect_bridge` notes);
  if the node aborts on first inference, run `scripts/evaluate.py` first to validate weights,
  or set `allow_classical_fallback: true` (default).
- **vision_msgs:** Preferred publisher type is `vision_msgs/Detection2DArray` on
  `/vial/detections`. If unavailable, detections still publish as `std_msgs/String` JSON
  on `/vial/detections_json` (same schema always).

## Build & run (ROS2)

```bash
cd E:/Desktop/Jessica/Camera/ros2_ws
colcon build --packages-select agent_01_yolo
# Linux:
source install/setup.bash
# Windows RoboStack:
# call install\setup.bat

ros2 launch agent_01_yolo vial_detect.launch.py
# live camera:
ros2 launch agent_01_yolo vial_detect.launch.py run_offline:=false
```

Standalone (after `pip install -e .` inside a sourced ROS env):

```powershell
$env:PYTHONPATH = "E:\Desktop\Jessica\Camera\ros2_ws\src\agent_01_yolo;$env:PYTHONPATH"
ros2 run agent_01_yolo vial_detector_node --ros-args --params-file `
  E:\Desktop\Jessica\Camera\ros2_ws\src\agent_01_yolo\config\params.yaml
```

### Node parameters

| Param | Default | Meaning |
|-------|---------|---------|
| `image_topic` | `/camera/color/image_raw` | `sensor_msgs/Image` |
| `dataset_path` | `.../vial_dataset/raw` | Offline folder |
| `run_offline` | `true` | Dataset vs subscribe |
| `model_path` | `.../runs/vial-live/weights/best.pt` | YOLO `.pt` |
| `confidence_threshold` | `0.25` | Match plan.md holdout gate |
| `visualize` | `true` | Publish annotated image |
| `allow_classical_fallback` | `true` | Cap heuristic if YOLO load fails |

### Topics

| Topic | Type | Notes |
|-------|------|-------|
| `/vial/detections` | `vision_msgs/Detection2DArray` | If `vision_msgs` installed |
| `/vial/detections_json` | `std_msgs/String` | Always (JSON) |
| `/vial/annotated` | `sensor_msgs/Image` | BGR annotated |

Logs rolling **FPS** and **detection count** every `log_every_n` frames.

## Evaluate (no ROS required)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\agent_01_yolo
python scripts/evaluate.py --split test --conf 0.25
python scripts/evaluate.py --split raw --max-images 50
```

Reports precision / recall / mean IoU / image accuracy when YOLO `.txt` GT exists;
otherwise qualitative detection ratio + timing. Writes `eval_results.json`.

## Train / fine-tune path

Preferred (Camera promote discipline — do not overwrite live on a tie):

```powershell
cd E:\Desktop\Jessica\Camera
python train_vial_model.py split
python train_vial_model.py train
python eval_asymmetric_split.py --weights runs\vial-ft\weights\best.pt
# only after clear holdout beat:
python train_vial_model.py train --export-live
```

Package-local Ultralytics wrapper:

```powershell
python scripts/train_finetune.py --epochs 40
python scripts/evaluate.py --model runs\agent01-ft\exp\weights\best.pt
```

## Failure modes

1. **Missing weights** → classical white-cap fallback (lower precision; demo only).
2. **vision_msgs missing** → JSON-only detections (documented).
3. **Transparent vial / glare / sideways** → RGB YOLO may miss or box lid-only; live
   `depthcamera.py` clear-cup depth gates are out of scope for this node.
4. **Windows Torch/OpenMP abort** → known RoboStack issue; validate with `evaluate.py`.
5. **Empty negatives scarce** → lowering conf below ~0.15 risks empty FP (see `camera_gates.json`).

## Match to Camera conventions

- Weight resolve order mirrors `depthcamera.find_weights`: vial-live → r1 → r3 → bon → yolov8s.
- Image codec avoids `cv_bridge` like `orbbec_dual_sync.image_codec`.
- Holdout metrics use IoU ≥ 0.5 and conf 0.25 like `plan.md`.
