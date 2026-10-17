# R2-06 supervisor notes

## Round-1 upright miss (root cause)

1. `agent_06_segmentation.resolve_model_path(prefer_seg=True)` could select COCO `yolov8n-seg.pt` before vial-live **det** weights → empty / wrong-class preds on easy stills.
2. Classical peers expand cap→body **downward only** → brittle on real mounts; no sideways path.
3. No rotation TTA; sideways ignored.
4. During R2 debug: ungated Method-I on zero-depth RGB stills flooded upright frames and displaced YOLO (fixed by requiring valid depth).

## Fix shipped

- Stage A: vial-det weights only + recall conf + conditional rot TTA + 4-axis cap proposals + Method-I **iff** depth valid.
- Stage B: `{upright, sideways, other}` + swap/expand refine; trusted YOLO/Method-I boxes lightly padded.

## Metrics (pass vs R1)

| | R1 agent_06 | R2-06 |
|--|-------------|-------|
| upright recall@0.5 | 0.90 (test only) | **0.975** |
| sideways recall@0.5 | n/a (0) | **0.975** |

**PASS** on upright+sideways recall improvement.

## Paths / run

- Package: `E:\Desktop\Jessica\Camera\ros2_ws\src\r2_06_two_stage_orient\`
- Eval JSON: `...\results\eval_latest.json`
- `python scripts\evaluate.py`
