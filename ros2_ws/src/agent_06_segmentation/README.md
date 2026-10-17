# agent_06_segmentation — Instance segmentation vial detector

ROS2 (ament_python) package for **YOLOv8-seg** instance segmentation of clear plastic vials, with train scaffolding and a **mask-from-bbox** demo fallback.

**Package path:** `E:\Desktop\Jessica\Camera\ros2_ws\src\agent_06_segmentation\`

## Dataset findings

| Item | Result |
|------|--------|
| Primary data | `Camera/vial_dataset/` (raw + `split_70_10_20`) |
| Label format | YOLO **detection** only: `class cx cy w h` (5 tokens) |
| Masks / polygons | **None** in `raw/`, splits, or review JSON |
| Sidecars | `raw/*.json` have pixel `box [x,y,w,h]` + depth distance — no segmentation |
| Synth race set | `sideways_race/synth/*.json` — boxes only |
| Counts (approx.) | ~373 raw pairs; 357 non-empty 5-token labels sampled as all-bbox |

## Suitability verdict

**Prefer bbox detection (YOLOv8 det) for this dataset.** There is no instance-mask GT, so true Mask R-CNN / YOLOv8-seg training cannot be supervised honestly.

Segmentation is still useful when you need a **silhouette overlay**, lid/body ROI for depth, or a path toward real polygon labels. This package:

1. Runs **YOLOv8-seg** if `runs/vial-seg` or `yolov8n-seg.pt` is available  
2. Else loads existing **vial det** weights (`vial-live` → r1 → bon) and synthesizes masks from boxes (ellipse by default)  
3. Provides **prepare + train** scaffolding that converts boxes → pseudo YOLO-seg polygons  

## Layout

```
ros2_ws/src/agent_06_segmentation/
  agent_06_segmentation/
    vial_detector_node.py   # ROS2 node
    segmenter.py            # Ultralytics + mask fallback
    image_codec.py
  config/params.yaml
  launch/vial_segment.launch.py
  scripts/
    prepare_seg_dataset.py  # bbox → YOLO-seg pseudo polygons
    train_seg.py            # train → runs/vial-seg/
    evaluate.py             # bbox IoU + mask IoU (real or pseudo)
  package.xml  setup.py  requirements.txt  README.md
```

## Build & run (ROS2)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws
# optional: pip install -r src\agent_06_segmentation\requirements.txt
colcon build --packages-select agent_06_segmentation
.\install\setup.ps1

# Offline over vial_dataset/raw
ros2 launch agent_06_segmentation vial_segment.launch.py

# Live topic
ros2 launch agent_06_segmentation vial_segment.launch.py run_offline:=false image_topic:=/orbbec/cam0/color/image_raw
```

### Topics

| Topic | Type | Role |
|-------|------|------|
| `/vial/detections_json` | `std_msgs/String` | Always — JSON boxes + mask metadata |
| `/vial/detections` | `vision_msgs/Detection2DArray` | If `vision_msgs` installed |
| `/vial/annotated` | `sensor_msgs/Image` | Boxes + tinted masks |
| `/vial/mask_overlay` | `sensor_msgs/Image` | Mask overlay only |

Node executable name: **`vial_detector_node`**.

## Train scaffolding (pseudo masks)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\agent_06_segmentation
python scripts\prepare_seg_dataset.py --shape ellipse --points 16
python scripts\train_seg.py --epochs 30 --model yolov8n-seg.pt
# → E:\Desktop\Jessica\Camera\runs\vial-seg\weights\best.pt
```

The ROS node auto-picks `runs/vial-seg/weights/best.pt` when present.

## Evaluate

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\agent_06_segmentation
python scripts\evaluate.py --split test
# Optional: trained seg weights
python scripts\evaluate.py --split test --weights E:\Desktop\Jessica\Camera\runs\vial-seg\weights\best.pt
```

- **`bbox_recall_at_0.5` / `bbox_mean_iou`** — vs real GT boxes  
- **`mask_mean_iou_real_gt`** — only if label files have polygons  
- **`pseudo_mask_miou`** — pred mask vs ellipse/rect from GT box (demo metric)

## Weight resolution order

1. `model_path` param  
2. `Camera/runs/vial-seg/weights/best.pt`  
3. `yolov8n-seg.pt` / `yolov8s-seg.pt` under Camera root  
4. Det weights: `vial-live` → `r1-winner` → `bon-winner` (+ mask-from-bbox)  
5. Ultralytics download of `yolov8n-seg.pt`

## Limitations

- No real mask GT → trained seg heads learn **box-shaped / elliptical** pseudo silhouettes, not true clear-vial boundaries.  
- COCO-pretrained `yolov8n-seg` will not know the `vial` class until fine-tuned; without `vial-seg` weights the node falls back to **det + ellipse**.  
- Clear plastic + white lid scenes are hard for generic COCO segmenters.  
- Mask overlay is for demo / ROI; production ranging should keep using lid-depth + bbox pipelines unless real polygons are labeled.  
- `vision_msgs` is optional; JSON always works on Windows RoboStack.
