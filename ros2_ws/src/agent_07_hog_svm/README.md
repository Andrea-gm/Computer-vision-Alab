# Agent #07 — HOG + Linear SVM vial detector

Classical ML detector for clear plastic vials with white screw caps using **OpenCV HOG** features and a **Linear SVM** (scikit-learn `LinearSVC`), with multi-scale sliding windows and hard-negative mining.

## Dataset findings

| Item | Value |
|------|--------|
| Root | `E:\Desktop\Jessica\Camera\vial_dataset` |
| Labels | YOLO txt (`class xc yc w h`), class `0 = vial` |
| Raw pairs | **373** jpg+txt (357 positive, **16** empty negatives) |
| Split `split_70_10_20` | **261** train / **37** val / **75** test |
| Appearance | Rigid vial + opaque white cap; held / table / close-up; scale ~67–964 px |

### Labeling assumptions (training)

1. Positives = YOLO class-0 boxes covering **lid through cup** (not lid-only).
2. Empty `.txt` = background-only hard negatives.
3. Easy negatives = random crops with IoU &lt; 0.1 vs GT.
4. Hard negatives = sliding-window hits with score ≥ 0 and IoU &lt; 0.3 vs GT (one retrain pass).

## Suitability verdict

**Usable fit with caveats.** Vials are mostly rigid and edge-rich (cap + cylinder silhouette), so HOG+SVM is a legitimate classical baseline. Weaknesses on *this* data:

- Clear cup body leaks background texture into the window.
- Pose mix (cap-top, cap-bottom, held close) stretches a single `winSize`.
- Sliding windows are slow vs one-stage CNNs; we cap `max_image_side` for ROS rates.

**Pivot within classical ML (kept):** Linear SVM on HOG (not RBF — too slow at detect time); hard-negative mining instead of selective-search (SS is heavy/unavailable in many OpenCV builds); single upright-biased window `(64, 80)` matching median bbox aspect ≈ 0.84.

Fire risk: lower recall/precision than YOLO on the same split; treat as classical baseline, not production winner.

## Package path (mandatory)

```
E:\Desktop\Jessica\Camera\ros2_ws\src\agent_07_hog_svm\
```

## Train

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\agent_07_hog_svm
python scripts\train_hog_svm.py `
  --dataset E:\Desktop\Jessica\Camera\vial_dataset\split_70_10_20 `
  --out models\hog_svm_vial.json
```

Outputs:

- `models/hog_svm_vial.json` — weights + bias + meta  
- `models/hog_svm_vial.yml` — OpenCV FileStorage sidecar  
- `models/train_metrics.json`

### Holdout (test, conf=0.5, IoU≥0.5)

| Metric | Value |
|--------|--------|
| Precision | **0.79** |
| Recall | **0.81** |
| Mean matched IoU | **0.71** |
| Empty FP | 1 |
| Mean infer | ~387 ms @ `max_image_side=480` |

## Evaluate (no ROS required)

```powershell
python scripts\evaluate.py `
  --dataset E:\Desktop\Jessica\Camera\vial_dataset\split_70_10_20 `
  --split test `
  --model models\hog_svm_vial.json `
  --out models\eval_metrics.json
```

## Build & run (ROS2)

Requires a sourced ROS2 environment (`rclpy`). On this Windows host, system Python often lacks `rclpy` — use the same env that builds `orbbec_dual_sync`, or WSL2.

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws
colcon build --packages-select agent_07_hog_svm
.\install\setup.ps1

# Offline over dataset folder
ros2 launch agent_07_hog_svm vial_detect.launch.py `
  dataset_path:=E:/Desktop/Jessica/Camera/vial_dataset/split_70_10_20/images/test `
  model_path:=E:/Desktop/Jessica/Camera/ros2_ws/src/agent_07_hog_svm/models/hog_svm_vial.json

# Live camera topic
ros2 launch agent_07_hog_svm vial_detect.launch.py `
  image_topic:=/camera/color/image_raw
```

### Topics

| Topic | Type | Notes |
|-------|------|--------|
| `vial/annotated` | `sensor_msgs/Image` | Boxes drawn |
| `vial/detections` | `vision_msgs/Detection2DArray` | If `vision_msgs` installed |
| `vial/detections_json` | `std_msgs/String` | Always published (fallback) |

### Params

See `config/params.yaml`: `image_topic`, `dataset_path`, `model_path`, `confidence_threshold`, `visualize`, `scale_factor`, `step_size`, `max_image_side`, `nms_iou`.

## Dependencies

- Python: `numpy`, `opencv-python`, `scikit-learn` (`requirements.txt`)
- ROS2: `rclpy`, `sensor_msgs`, `std_msgs`, optional `vision_msgs`

## Limitations

- Multi-scale sliding window ≈ tens–hundreds of ms/frame at `max_image_side=480`.
- Single window aspect; strong pose/scale outliers miss.
- Transparent vial + cluttered desk → HOG FP without mining / higher threshold.
- Not a replacement for the repo’s YOLO live weights (`runs/vial-live`).
