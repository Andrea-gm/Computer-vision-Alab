# r2_06_two_stage_orient — detect + orientation classify

Round-2 Agent #06. Two-stage vial detector that **fixes Round-1 upright misses** and **adds sideways**.

**Package:** `E:\Desktop\Jessica\Camera\ros2_ws\src\r2_06_two_stage_orient\`

## Why Round-1 missed easy upright

From `agent_06_segmentation` + classical peers (`agent_02/04/05`):

| Failure | Effect |
|---------|--------|
| Stage-A weight resolution preferred COCO `yolov8n-seg` before vial-live **det** | Wrong classes / empty preds on easy stills |
| Cap→body expand **downward only** | Misses when lid is not on top; no sideways |
| No rotation TTA | Upright-trained YOLO never proposes lying vials |
| Tight classical aspect / score gates | Dropped mid-field upright lids under desk clutter |

Offline `agent_06` reported test `bbox_recall@0.5 = 0.9` when forced onto `vial-live`, but the live/seg resolution path and upright-only geometry still failed user tests and ignored sideways.

## Fix (this package)

**Stage A — proposals (recall-oriented)**

1. Load **vial det** weights only (`vial-live` → `r1-winner` → `bon-winner`); never auto-pick COCO-seg.
2. Low YOLO conf (`0.12`) + **90/180/270° rotation TTA** with correct inverse box map.
3. White-cap seeds expanded on **four axes** (down/up/left/right), not down-only.
4. Optional `sideways_race` Method-I when depth is present.

**Stage B — orientation + refine**

- Class ∈ `{upright, sideways, other}` from cap position, aspect, depth-hole axis.
- Refine AABB: swap/expand along major axis so lying vials cover lid→cup.

## Layout

```
r2_06_two_stage_orient/
  r2_06_two_stage_orient/
    stage_a.py stage_b.py pipeline.py
    vial_detector_node.py image_codec.py msgs_compat.py types.py
  config/params.yaml
  launch/vial_detect.launch.py
  scripts/evaluate.py
  results/eval_latest.json
  package.xml setup.py requirements.txt README.md
```

## Evaluate (no ROS)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\r2_06_two_stage_orient
python scripts\evaluate.py
# classical-only smoke (no Torch):
python scripts\evaluate.py --no-yolo --synth-only
```

Reports **per-orientation** precision / recall / mean IoU (upright from `vial_dataset` test + synth upright notes; sideways from `sideways_race/synth`).

### Latest metrics (`results/eval_latest.json`)

| Subset | Recall@0.5 | Precision | Img recall | Mean IoU |
|--------|------------|-----------|------------|----------|
| **upright** | **0.975** (78/80) | 0.565 | 0.963 | 0.729 |
| **sideways** | **0.975** (39/40) | 0.406 | 0.975 | 0.831 |
| overall (pos) | 0.929 | 0.413 | 0.921 | 0.763 |

vs Round-1 `agent_06_segmentation` test: bbox recall 0.90, **no sideways**, and weight resolution could prefer COCO-seg over vial-live.

Run: `python scripts/evaluate.py` (151 samples, ~160 s with YOLO on CPU).

## Build & run (ROS2)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws
colcon build --packages-select r2_06_two_stage_orient
.\install\setup.ps1
ros2 launch r2_06_two_stage_orient vial_detect.launch.py
# live:
ros2 launch r2_06_two_stage_orient vial_detect.launch.py run_offline:=false
```

### Topics

| Topic | Type | Role |
|-------|------|------|
| `/vial/detections_json` | `std_msgs/String` | JSON boxes + `orientation` |
| `/vial/detections` | `vision_msgs/Detection2DArray` | If `vision_msgs` installed |
| `/vial/annotated` | `sensor_msgs/Image` | Color-coded by orientation |

## Limitations

- Orientation is classical (cap/aspect/depth), not a trained pose head.
- Rotation TTA ×4 increases latency when YOLO is enabled.
- Synth upright GT comes from control notes (sideways race stores upright as negatives).
