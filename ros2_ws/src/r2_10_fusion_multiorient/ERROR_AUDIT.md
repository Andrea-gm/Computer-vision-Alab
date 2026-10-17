# r2_10 frame-level error audit

Audit date: 2026-10-09  
Production weights: `E:\Desktop\Jessica\Camera\runs\vial-live\weights\best.pt`  
Match rule: one-to-one boxes at IoU >= 0.50.

## Executive finding

The published `eval_multiorient.json` is reproducible, but its PASS hides two
important gaps:

1. `topdown_*` is counted as **negative**, although each frame contains a vial.
   The production stack misses all 6/6 top-down controls.
2. Synthetic `upright_*` controls are excluded from the score. The production
   stack misses all 6/6.

The frozen real upright holdout remains good but not perfect: 49 TP, 3 FP,
5 FN (precision 94.2%, recall 90.7%). Across its 56 frames, 7 frames are
wrong. Normal synthetic sideways is 24/24. Hard angled/occluded sideways is
14/16, but fusion adds 22 unrelated YOLO boxes; this is the largest actionable
fusion defect.

Do **not** simply repoint production to `r2_01` weights. In an A/B run through
the same fusion code, the five real-holdout misses stayed missed while FP rose
from 3 to 10 (precision 94.2% -> 83.1%). Its apparent synthetic improvement is
not a clean holdout result because those synthetic samples were ingested by
the r2_01 training pipeline.

## What was run

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws
.\build_r2_10.cmd
python .\src\r2_10_fusion_multiorient\scripts\audit_errors.py
```

The build passed. The audit calls the same `FusionVialDetector` and production
weight path as ROS, but runs deterministic batches with GT. Live
`run_offline:=false` was not used to declare correctness: it is an infinite
stream and there is no synchronized live GT. The ROS offline replay points at
`vial_dataset/raw`; it visualizes predictions but does not score its labels,
and much of it is training data.

Detailed machine-readable output:

- `models/error_audit.json`
- annotated evidence: `models/error_frames/`
- reproducible audit: `scripts/audit_errors.py`
- r2_01 A/B: `models/error_audit_r2_01_model.json`

The broad audit visited 132 frames and initially reports 43 wrong frames,
87 TP, 34 FP, and 27 FN. Eight `real_wide_*` samples have incomplete GT (see
below), so these broad totals must not be presented as a clean accuracy metric.

## All production error frames

### Real upright holdout: 7 wrong frames

| Frame | Error | Observation |
|---|---|---|
| `05_sitting_on_paper` | FN + FP | GT vial missed; RGB classical fires on unrelated edge, IoU 0 |
| `06_held_tight_cap_on_top` | FN | clipped/close weak-angle vial, no prediction |
| `07_plain_background` | FN | large upright vial, no prediction |
| `10_cap_on_top_close` | FN | close upright vial, no prediction |
| `vial_0021` | FN | weak-angle vial, no prediction |
| `vial_0035` | FP | native YOLO, confidence 0.288, fires on white circular clutter |
| `vial_0191` | FP | RGB-only `classical_sideways`, confidence 0.88 |

Pose totals on this holdout: upright has 2 FN; weak-angle has 3 FN plus the
unrelated classical FP in `05_sitting_on_paper`; empty/clutter has 2 FP.

### Top-down: 6/6 missed

`topdown_00`, `topdown_01`, `topdown_02`, `topdown_03`, `topdown_04`,
`topdown_05`.

With depth present, Method-I returns empty, then the fusion empty veto retains
only confident native upright YOLO. That intentionally suppresses these
square/circular top views. The old evaluator incorrectly rewards the six
misses as six correct negatives.

### Synthetic upright controls: 6/6 missed

`upright_00`, `upright_01`, `upright_02`, `upright_03`, `upright_04`,
`upright_05`.

These tiny synthetic controls are not representative enough to set a
production threshold, but they expose the same depth-veto/model-coverage gap.
They were excluded from `eval_multiorient.json`.

### Normal sideways: no errors

All `sideways_left_00..11` and `sideways_right_00..11` pass: 24/24 TP,
0 FP, 0 FN.

### Hard angled/occluded sideways: all 16 frames contain an error

- Actual vial FN / bad box: `hard_sideways_left_00` (best IoU 0.490) and
  `hard_sideways_left_01` (complete miss).
- Vial matched but unrelated extra boxes present:
  `hard_sideways_left_02..07` and `hard_sideways_right_00..07`.

Method-I supplies 14 correct depth-backed boxes. The fusion then keeps
non-overlapping YOLO detections from clutter. Of 23 FP boxes in this bucket,
22 are YOLO/native/TTA and all have confidence below 0.60; one is the
Method-I box at IoU 0.490. This explains the reported sideways precision of
62.3% despite 95% recall.

### `real_wide_*`: GT defect, not eight trustworthy CV failures

`real_wide_00..07` are all flagged by strict matching, but their frames retain
an obvious source-image upright vial while GT describes only a separate tiny
wide vial. Some are duplicate source scenes. For example, `real_wide_02`
labels the tiny object near `(260,147)` while YOLO correctly detects the
unlabelled upright vial near `(402,214)`. Repair these annotations before using
these eight frames in precision/recall.

### True negative controls

All 8 `hard_negative_*` and all 8 synthetic `empty_*` frames pass with zero
detections.

## Source attribution

Across the broad strict audit, FP boxes are attributed as:

- YOLO/native/refined: 26
- YOLO rotation TTA: 5
- classical sideways: 3

The dominant failure is therefore not Method-I itself; it is permissive
retention of unrelated, non-overlapping YOLO boxes after Method-I has already
found the depth-supported sideways vial. Camera attribution cannot be scored:
the offline GT has no dual-camera identity, and both live subscriptions use
the same configured output `frame_id`.

## Ranked root causes

1. **Training coverage and evaluation leakage:** the main production YOLO is
   overwhelmingly upright-trained. Top-down is unsupported; real sideways
   GT is essentially absent. Existing synth is reused across train/eval in
   r2_01, so it cannot prove generalization.
2. **Fusion gate is too permissive after a depth match:** `fusion.py` keeps
   every non-overlapping upright YOLO box at confidence >= 0.35. This creates
   22 hard-sideways FP boxes.
3. **Depth empty veto encodes “not sideways” as “not a vial”:** this suppresses
   all top-down controls and the tiny synthetic upright controls.
4. **RGB classical fallback is unsafe without depth:** it causes
   `05_sitting_on_paper` and `vial_0191` behavior.
5. **GT quality:** `real_wide_*` has incomplete labels, and the current headline
   evaluator treats vials in `topdown_*` as negatives.

## Best solution

The primary path is a **clean, group-disjoint real multi-pose dataset followed
by one retrained multi-orientation YOLO model, while retaining depth Method-I
as a high-precision sideways branch**.

Do this instead of threshold-only tuning or directly pinning the current
r2_01 model:

1. Capture synchronized RGB-D from both production cameras: upright,
   15-45 degree weak angles, fully sideways left/right, top-down/cap-facing,
   clipped/close, glare, hand occlusion, and empty/cap-only clutter.
2. Label every vial in each image. Repair `real_wide_*`. Keep sequences,
   source captures, rotations, and near-duplicates in the same split group.
3. Freeze a real test set before training. Synthetic images may augment train,
   but must not be the claimed holdout.
4. Start from the r2_01 multi-orientation training recipe, retrain with real
   sideways/top-down examples and explicit empty/cap-only negatives.
5. Keep `I_hard_robust`/Method-I when aligned depth is valid. When it finds a
   vial, require extra YOLO boxes to have depth/body support or, as an
   immediate candidate gate, confidence >= 0.60. On this audit that candidate
   removes all 22 unrelated hard-sideways YOLO FP boxes; validate it on
   genuine multi-vial scenes before production.
6. Split the depth veto into pose-aware evidence: Method-I-empty should veto
   only weak **sideways** hypotheses, not confident top-down/upright
   hypotheses from a retrained detector.
7. Disable RGB-only classical sideways in production unless temporal
   consistency confirms it. Use two-of-three frames and preserve camera ID in
   output.
8. Re-run this frame catalog plus a new real pose holdout. Gate release on
   per-pose recall, empty/cap-only FP rate, IoU, and per-camera results—not one
   aggregate PASS.

Short-term, test the `>=0.60` extra-YOLO gate and raise native empty-scene YOLO
confidence to at least 0.35. These are mitigations, not substitutes for the
real multi-pose retrain.

## Limits

- No claim is made about every possible live frame; live streams have no GT.
- The real holdout is mostly upright and has only two empty images.
- Sideways and top-down conclusions are currently synthetic.
- Axis-aligned IoU penalizes angled vials; OBB/segmentation is an alternative
  if grasp geometry, not only vial presence, is required.
