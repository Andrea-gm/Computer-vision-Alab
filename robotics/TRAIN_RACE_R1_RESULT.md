# TRAIN RACE — Round 1 RESULT

**Applied to MAIN:** `E:\Desktop\Jessica\Camera`  
**Rank key (user priority):** highest recall / fewest FN (IoU≥0.5 on positives) → test accuracy → lowest empty FP → higher val → simpler.

## Winner

| Field | Value |
| --- | --- |
| **WORKTREE_ID** | `train-r1-07-cf168274` |
| **Strategy** | MVAug-inspired mild perspective + freeze/unfreeze (phase1 selected) |
| **Test acc** | **100.0%** (11/11) |
| **Val acc** | **100.0%** (6/6) |
| **Recall (positives)** | **1.0** (TP 10 / FN 0) |
| **Empty FP** | **0** |
| **Weights (source)** | `…\train-r1-07-cf168274\runs\train-r1-07-mvaug-freeze-phase1\weights\best.pt` |
| **Weights (MAIN)** | `runs/train-race/r1-winner/weights/best.pt` **and** `runs/vial-live/weights/best.pt` |

Split: seed **42**, train/val/test **40/6/11** from `vial_dataset/raw` (review folder not used). Dataset not wiped.

## Full ranking

| Rank | ID | Test% | Val% | Recall | FN | Empty FP | Strategy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | train-r1-07-cf168274 | 100.0 | 100.0 | 1.0 | 0 | 0 | MVAug mild perspective + freeze/unfreeze (phase1) |
| 2 | train-r1-05-3b4c8d1b | 90.9 | 100.0 | 1.0 | 0 | 1 | longer cos-LR for low-conf recall |
| 3 | train-r1-04-efb37cb8 | 90.9 | 100.0 | 0.9 | 1 | 0 | copy-paste / multi-angle fine-tune |
| 4 | train-r1-10-29c3a586 | 90.9 | 100.0 | 0.9 | 1 | 0 | ensemble-ready yolov8n single-best |
| 5 | train-r1-02-8ab04926 | 90.9 | 83.3 | 0.9 | 1 | 0 | stronger mosaic, mixup off, higher imgsz |
| 6 | train-r1-09-d93541be | 81.8 | 100.0 | 0.8 | 2 | 0 | short high-LR then low-LR fine-tune |
| 7 | train-r1-01-f27c408d | 81.8 | 83.3 | 0.8 | 2 | 0 | mild HSV-safe aug fine-tune |
| 8 | train-r1-03-a92325a8 | 72.7 | 83.3 | 0.7 | 3 | 0 | YOLOv8s careful anti-overfit |
| 9 | train-r1-06-01954e2b | 72.7 | 83.3 | 0.7 | 3 | 0 | hard-negative mining |
| 10 | train-r1-08-2d4a57e3 | 72.7 | 83.3 | 0.7 | 3 | 0 | depth-informed / strict box QA |

## MAIN apply notes

- Copied winner → `runs/train-race/r1-winner/weights/best.pt` and `runs/vial-live/weights/best.pt`.
- `depthcamera.find_weights` prefers `vial-live` → `r1-winner` → `bon-winner`.
- Live defaults eased for FN (strong YOLO conf **0.15**, mild structure relax in `camera_gates.json`); independent dual keeps **no peer soft-locate inventing boxes**.
- Dual-independent corroboration-only IPC unchanged.
- Round 2 / Round 3 not applied here.
