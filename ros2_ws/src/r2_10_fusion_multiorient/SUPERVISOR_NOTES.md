# SUPERVISOR_NOTES — Round-2 Agent #10 fusion multi-orient

**Deliverable only:** `E:\Desktop\Jessica\Camera\ros2_ws\src\r2_10_fusion_multiorient\`  
(Does not depend on other `r2_*` packages.)

## Round-1 root cause (ruthless)

User report: Round-1 was terrible even when orientation looked “correct.” Diagnosis from `agent_*` READMEs + eval artifacts + sideways race:

1. **Dataset / training skew** — `vial_dataset` labels are overwhelmingly upright. Fine-tuned YOLO (`vial-live`) reaches ~0.90 upright test recall but **0.00 sideways recall** on `sideways_race/synth` (confirmed via `--yolo-only` ablation in this package).
2. **Classical agents encode upright physics** — downward cap expand (`agent_04/05/09`), single upright HOG window (`agent_07`), upright templates (`agent_03`), aspect gates that reject fallen AABB h/w ~0.45–0.85.
3. **Ensemble did not fix orientation** — `agent_10` matched YOLO on upright stills; classical recall 0.17; **no sideways metrics**, no rotation TTA.
4. **Naive fusion is worse than YOLO** — early Round-2 attempts that always ran Method-I without depth **destroyed upright** (Method-I fires on 47/56 upright test images when `depth=None`).
5. **Sideways race winner needs depth** — `I_hard_robust` = 0.95 recall / 0 FP on synth **with** RGB-D; must be depth-gated in a multi-orient stack.

## Fix shipped

| Stage | Behavior |
|---|---|
| YOLO | conf≥0.25, smart TTA (rotations only if native upright weak) |
| Sideways classical | Method-I when depth present; RGB fallback only if YOLO empty |
| Fusion rule | If sideways fires → it owns overlaps; keep non-overlap upright YOLO |
| Empty veto | depth ∧ Method-I∅ → drop TTA/classical upright; native YOLO only if conf≥0.50 & upright aspect |
| Refine | Cap-axis refine on YOLO only |

## Metrics (ran `scripts/evaluate.py`)

| | Precision | Recall | imgR@0.5 | mean IoU | empty FP |
|---|---:|---:|---:|---:|---:|
| Upright test | 0.942 | 0.907 | 0.907 | 0.662 | 2 |
| Sideways synth | 0.623 | 0.950 | 0.950 | 0.845 | 0 |

**Pass/fail vs Round-1:** upright **PASS** · sideways **PASS** · overall **PASS**

| Round-1 reference | Value | This package |
|---|---|---|
| agent_01 YOLO upright recall | ~0.90 | **0.907** (tied) |
| agent_01 / YOLO sideways | unsupported | **0.950** imgR |
| sideways baseline_depthcamera | 0.375 R | **0.950** R, **0** empty FP |
| agent_10 ensemble upright | 0.907 R | **0.907** R |

Artifact: `models/eval_multiorient.json`.

## Paths

- Package: `E:\Desktop\Jessica\Camera\ros2_ws\src\r2_10_fusion_multiorient\`
- Weights: `E:\Desktop\Jessica\Camera\runs\vial-live\weights\best.pt`
- Upright split: `E:\Desktop\Jessica\Camera\vial_dataset\split_70_10_20_latest2`
- Sideways synth: `E:\Desktop\Jessica\Camera\sideways_race\synth`

## Build / run

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws
.\build_r2_10.cmd
.\launch_r2_10.cmd run_offline:=false
```

Offline: `python scripts/evaluate.py` from the package root.

## Confidence for user retest

**High (~0.85)** on the frozen upright split + sideways synth harness (both PASS).  
**Medium (~0.65)** on live dual-cam without depth on the color topic — sideways branch degrades to YOLO-TTA / RGB fallback; publish depth on `/camera/depth/image_raw` for production sideways.  
Upright empty FP (2/2 empties in test) is the remaining soft spot vs ideal 0.
