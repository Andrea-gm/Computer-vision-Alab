# r2_05_aug_retrain — Round-2 Agent #05 (data-centric)

Heavy **rotation / perspective / scale** offline augmentation + **sideways_race/synth**
injection, then YOLO retrain from **COCO `yolov8n.pt`** (not vial-live).

Training was **stopped early at epoch 17/20** (user speed demand). Shipped weights are
the Ultralytics `best.pt` from that run.

## Holdout metrics (conf=0.25, IoU≥0.5, imgsz=320)

| Split | N | Precision | Recall | mean IoU |
|-------|---|-----------|--------|----------|
| **Upright** (test sample, max 40) | 40 | **0.946** | **0.875** | 0.804 |
| **Sideways** (synth holdout) | 12 | **1.000** | **0.917** | 0.750 |

- Recall gap (upright − sideways) = **−0.042** (sideways does **not** collapse)
- `sideways_recall_ok = true`
- Full JSON: `results/eval_upright_sideways.json`

## Label audit (raw)

| Signal | Value |
|--------|-------|
| Raw stills | 373 (357 pos / 16 empty) |
| Upright-like (h/w > 1.0) | **353 / 357** |
| Sideways-like (h/w < 0.85) | **1 / 357** |
| After aug (train boxes) | sideways fraction **~0.36** |

## Aug pipeline

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\r2_05_aug_retrain
python scripts/build_aug_dataset.py --augs-per-image 2 --seed 42
```

1. Seed train/val from `vial_dataset/split_70_10_20`
2. Offline: ±180° + sideways-biased (~±90°) rotate, perspective, scale, photo jitter
3. Inject `sideways_race/synth` (hold out ~30% sideways)
4. Holdouts: `holdout_upright` = official test; `holdout_sideways` = synth holdout

## Train / eval

```powershell
# Already run (stopped epoch 17). Do NOT start multi-hour retrain unless asked.
# python scripts/train.py --epochs 20 --imgsz 320 --batch 8 --device cpu

python scripts/evaluate.py --conf 0.25 --iou 0.5 --imgsz 320 --max-images 40
```

Weights: **`models/weights/best.pt`** (also `runs/r2_05/weights/best.pt`)

## ROS2

```bash
cd E:/Desktop/Jessica/Camera/ros2_ws
colcon build --packages-select r2_05_aug_retrain
source install/setup.bash
ros2 launch r2_05_aug_retrain vial_detect.launch.py
ros2 launch r2_05_aug_retrain vial_detect.launch.py run_offline:=false
```

## Paths

| Artifact | Path |
|----------|------|
| Package | `E:\Desktop\Jessica\Camera\ros2_ws\src\r2_05_aug_retrain\` |
| Weights | `...\models\weights\best.pt` |
| Aug data | `...\data\aug_split\` |
| Eval | `...\results\eval_upright_sideways.json` |
| Audit | `...\results\aug_audit.json` |
