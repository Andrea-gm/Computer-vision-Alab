# r2_04_cap_body_cascade

CV Expert ROUND-2 Agent #04 — **orientation-agnostic** white-cap → transparent-body geometric cascade.

## Diagnosis (why upright was 0.01)

On `vial_dataset` test, **cap seeds hit GT in ~97% of images**. Failure was ranking/expansion on RGB-only stills: rim-noise dual-blob links outscored correct geometric probes, and there was **no depth** (synth holes) to anchor the body. Round-1-style “classical only” repeated that miss.

## Fix

1. **RGB-only path**: re-score with white-end + edge support; **penalize** dual-blob rim links; boost cardinal geometric probes.
2. **YOLO RGB proposal stage** (when `depth` is absent): fuse `runs/train-race/r2-winner` boxes via NMS so upright stills recover.
3. **Depth/sideways path**: unchanged classical dual-blob + PCA axis expansion (YOLO skipped when depth present).

Axis expansion remains orientation-agnostic (not down-only).

## Metrics @ IoU≥0.5, conf=0.35

| Bucket | Recall | Precision | mean IoU |
|--------|--------|-----------|----------|
| **upright dataset** test (n=75) | **0.78** | 0.39 | 0.75 |
| **sideways** synth (n=40) | **0.85** | 0.58 | 0.67 |
| upright synth (n=6) | 0.50 | 1.00 | 0.73 |

Gate: upright dataset ≥0.50 (target ≥0.70) **and** sideways ≥0.70 → **pass** (`results/metrics.json`).

Full JSON: `results/metrics.json`.

## Evaluate

```bat
cd E:\Desktop\Jessica\Camera\ros2_ws\src\r2_04_cap_body_cascade
python scripts\evaluate.py --json-out results\metrics.json
python scripts\evaluate.py --no-yolo   REM classical-only ablation
```

## ROS2

```bat
cd E:\Desktop\Jessica\Camera\ros2_ws
colcon build --packages-select r2_04_cap_body_cascade
ros2 launch r2_04_cap_body_cascade vial_detect.launch.py
```

## Risks

- Upright precision still modest (~0.28) from residual classical glare FPs alongside YOLO.
- YOLO weights required for upright rescue (`ultralytics` + `r2-winner/best.pt`); without them RGB upright recall collapses.
- Sideways hard-angle FN ~6/40 remains.
