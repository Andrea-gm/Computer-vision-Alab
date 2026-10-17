# R2 Agent #02 — Rotated multi-scale templates

**Package:** `E:\Desktop\Jessica\Camera\ros2_ws\src\r2_02_rotated_templates\`

Classical CV vial detector (no YOLO). Multi-scale + **multi-angle** NCC on gray+edge templates. Template bank from **upright train GT** + **sideways synth GT** (even indices held-in; odds/hard held-out).

## Orientation strategy

1. Extract 8 upright + 8 sideways base templates (`data/templates/`).
2. Expand each at angles `0/45/90/135/180/225/270/315°`.
3. Multi-scale `TM_CCOEFF_NORMED` (gray 0.65 + Canny 0.35).
4. Pose-consistent filters + cap-polarity + cup-rim gates (sideways); softer top-cap gate (upright).
5. Aspect range `0.35–3.0` (Round-1 agent_03 used `0.75–2.20` and **rejected sideways**).

## Metrics (IoU≥0.5)

### Quick smoke (`--max-images 20 --max-sideways 24`, conf 0.52)

| Set | Recall | Precision | Img-acc | Notes |
|-----|-------:|----------:|--------:|-------|
| **Upright** (test[:20]) | **0.20** | 0.24 | 0.20 | tp/fp/fn=4/13/16 |
| **Sideways** (synth[:24]) | **0.36** | 0.20 | 0.36 | easy left 2/3, right 2/2; hard 0 |

### Full prior run (conf 0.52, all test + held-out synth)

| Set | Recall | Precision | Notes |
|-----|-------:|----------:|-------|
| Upright (test n=75) | 0.095 | 0.15 | weak vs agent_03 upright-only |
| Sideways overall | 0.47 | 0.41 | includes hard_* |
| **sideways_left** (held-out) | **0.80** | — | 8/10 |
| **sideways_right** (held-out) | **0.90** | — | 9/10 |
| hard_sideways_* | 0.00 | — | angled/occluded not covered |

**Not upright-only:** easy sideways does not fail silently (0.8 / 0.9 on held-out left/right).

## Run

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\r2_02_rotated_templates
pip install -r requirements.txt
python scripts/extract_templates.py
python scripts/evaluate.py --split test --conf 0.52 --max-images 20 --max-sideways 24
# full:
python scripts/evaluate.py --split test --conf 0.52
```

### ROS2

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws
colcon build --packages-select r2_02_rotated_templates
ros2 launch r2_02_rotated_templates vial_detect.launch.py
ros2 launch r2_02_rotated_templates vial_detect.launch.py run_offline:=false
```

Topics: `/vial/detections` (Detection2DArray if available), `/vial/detections_json`, `/vial/annotated`.

## Risks

- **Slow:** ~0.3–0.5 FPS (templates × angles × scales).
- **Upright weak** vs Round-1 upright-only template agent (~10–30% recall).
- **hard_sideways / ±15–30°** recall ≈ 0.
- Synth clutter / topdown / real_wide still produce some FP.
- No depth used.
