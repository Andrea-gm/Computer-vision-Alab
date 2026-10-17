# Agent #10 — Multi-stage vial ensemble

**Package path (canonical):** `E:\Desktop\Jessica\Camera\ros2_ws\src\agent_10_ensemble`

Real cascade detector for **this** Camera dataset: clear plastic vial + opaque white screw cap, YOLO-format single class `vial`.

## Why ensemble wins on THIS data

| Fact about the dataset | Why a cascade helps |
|---|---|
| Cap is opaque white; cup is nearly invisible in RGB | Classical HSV/edge proposals recover boxes YOLO under-scores |
| GT boxes are full lid→cup (aspect ~0.47–1.20) | Stage-2 shape filter kills wallpaper / hand blobs |
| Hard negatives: empty benches + bright tables | Stage-3 verifier (trained logistic on crops) kills FP from color |
| Existing BoN race: YOLO∪cap NMS tied 100% test | Union of proposal sources is already proven here |
| Live dual-cam still needs recall at low conf | Optional YOLO branch at conf 0.10 → verifier → NMS |

A **pure YOLO** fine-tune is strong when weights match the angle/lighting, but misses weak-cap / weird pose frames that classical proposals still fire on. A **pure classical** detector over-fires on white surfaces. The ensemble keeps both recall sources and lets the verifier + NMS decide.

## Pipeline

1. **Proposals (union):** white-cap HSV expand → edge/contour boxes → optional YOLO (`runs/vial-live` / race winners)
2. **Shape filter:** aspect / area gates from GT stats
3. **Verifier:** sklearn `StandardScaler + LogisticRegression` on 50-D classical features (white bands, edges, soft-HOG); heuristic fallback if no pickle
4. **Soft-NMS:** fuse overlapping cap/edge/YOLO boxes

## Build / run (ROS2)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws
# activate your ROS / pixi env first
colcon build --packages-select agent_10_ensemble
.\install\setup.ps1
ros2 launch agent_10_ensemble vial_detector.launch.py
```

Topics:

- sub: `/camera/color/image_raw` (`sensor_msgs/Image`)
- pub: `/vial/detections` (`vision_msgs/Detection2DArray`, or JSON `std_msgs/String` if `vision_msgs` missing)
- pub: `/vial/annotated` (`sensor_msgs/Image`)
- pub: `/vial/detections_json` (always JSON)

## Offline train + evaluate (no ROS required)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\agent_10_ensemble
python scripts\train_verifier.py --split E:\Desktop\Jessica\Camera\vial_dataset\split_70_10_20_latest2
python scripts\evaluate.py --split E:\Desktop\Jessica\Camera\vial_dataset\split_70_10_20_latest2 --subset test
```

## Layout

```
agent_10_ensemble/
  agent_10_ensemble/   # proposals, features, ensemble, ROS node
  config/ensemble_params.yaml
  launch/vial_detector.launch.py
  models/verifier.pkl
  scripts/train_verifier.py
  scripts/evaluate.py
  SUPERVISOR_NOTES.md
  requirements.txt
```

## Measured on `split_70_10_20_latest2` test (56 images)

| Method | Acc @ IoU≥0.5 | Recall | empty FP | mAP@0.5 | FPS* |
|---|---:|---:|---:|---:|---:|
| Pure YOLO (`vial-live`, conf 0.25) | 0.893 | 0.907 | 1 | — | ~6.3 |
| Classical only (cap+edge+verifier) | 0.179 | 0.167 | 1 | 0.12 | ~27 |
| **Ensemble (YOLO∪cap∪edge → verifier → NMS)** | **0.893** | **0.907** | **1** | **0.872** | ~5.3 |

\*Same machine; YOLO dominates runtime. Ensemble matches YOLO on this stills split; classical branch is the live safety net when YOLO under-fires (prior BoN #6 behavior). See `models/eval_test_*.json`.

See `SUPERVISOR_NOTES.md` for winner criteria.
