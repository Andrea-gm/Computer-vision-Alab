# r2_07_depth_anypose — evaluation summary

Generated from offline evals (no ROS). IoU threshold **0.5**.

## vial_dataset upright (real RGB stills)

Source: `Camera/vial_dataset/split_70_10_20` val+test, upright GT = boxes with h/w ≥ 1.05.  
**No depth** on this dataset — RGB path only.

| Mode | recall@0.5 | precision@0.5 | tp / pos | notes |
|------|------------|---------------|---------|-------|
| **rgb_yolo** (default when `runs/vial-live/weights/best.pt` present) | **0.938** | **0.744** | 90 / 96 | val 0.914 / test 0.951 |
| rgb_classical (blob/MSER/CC, `--no-yolo`) | 0.365 | 0.106 | 35 / 96 | lids merge into white walls |

Artifacts: `eval_vial_dataset_upright.json`, `eval_vial_dataset_upright_classical.json`.

## Synth race / any-pose (`sideways_race/synth`, 76 samples)

Artifact: `eval_latest.json`.

| Split | recall@0.5 | FP | mean IoU (matched) |
|-------|------------|----|--------------------|
| **Race** (sideways positives; sideways/angled preds) | **0.975 (39/40)** | **0** | 0.675 |
| **Any-pose** | **0.865 (45/52)** | **0** | 0.697 |

### Per-kind (any-pose)

| kind | recall | mean IoU |
|------|--------|----------|
| sideways_left | 1.000 (12/12) | 0.727 |
| sideways_right | 0.917 (11/12) | 0.682 |
| hard_sideways_left | 1.000 (8/8) | 0.630 |
| **hard_sideways_right** | **1.000 (8/8)** | 0.632 |
| upright | 1.000 (6/6) | 0.843 |
| topdown | 0.000 (0/6) | — |

### hard_sideways_right before → after (synth)

| Stage | recall@0.5 | notes |
|-------|------------|-------|
| Package README baseline / early Round-2 | **0.250 (2/8)** | chronic undersize + misses |
| Mid Round-2 (pre wrong-Z rescue) | **0.500 (4/8)** | sizing helpers; 4 near-miss / miss |
| **Current** | **1.000 (8/8)** | wrong-Z cap-ratio rescue + extent-collapse boost |

Race recall stayed **≥ 0.70** throughout the final gate (now **0.975**).

## What changed (supervisor follow-up)

1. **RGB upright real stills:** hybrid RGB path — YOLO (`vial-live/best.pt`) when weights exist, else classical. Param `use_yolo_rgb` (default true).
2. **Depth / synth hard_right:** detect wrong lid Z (`length_px < 1.55× cap_span`); restore race-like length/diam; rescue early extent collapse when clear+dark stay strong.
3. Depth geometry path unchanged in spirit (PCA / clear-cup); no RGB fallback after a real-depth clear-cup reject.
