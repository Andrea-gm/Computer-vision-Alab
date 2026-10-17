# SUPERVISOR_NOTES — Agent #10 ensemble

## Package location

`E:\Desktop\Jessica\Camera\ros2_ws\src\agent_10_ensemble\`  
(Do **not** treat `ros2_vial_competition/` as the deliverable.)

## Recommended winner criteria

Rank agents with this lexicographic key (frozen before looking at scores):

1. **Test accuracy @ IoU ≥ 0.5** on `vial_dataset/split_70_10_20_latest2` (or the supervisor-frozen split) — positives must match GT; empties must produce **zero** boxes.
2. **Empty-negative FP count** (lower is better) — break ties against color-only vaporware.
3. **mAP@0.5** on the same split (ranking quality when multiple boxes appear).
4. **mean best-IoU** on positives (localization quality).
5. **FPS** on the same machine / image size (penalize GPU-only monsters if CPU FPS &lt; 5).
6. **Robustness extras (secondary):** held-vial / white-surface / plain-background subsets; dual-cam stills if available.

## What is needed to crown a winner after testing

- [ ] Freeze the eval split path + seed (prefer `split_70_10_20_latest2` or rebuild from `raw/` with documented seed).
- [ ] Run each agent’s `evaluate.py` (or equivalent) with identical IoU / conf policy.
- [ ] Publish a single CSV/JSON table: acc, empty_fp, mAP50, mean_iou, fps, notes.
- [ ] Sanity-check 10 hard frames by eye (annotated images): FN on clear vial, FP on empty, lid-only boxes.
- [ ] Confirm ROS2 packages build under `ros2_ws` and publish `Detection2DArray` + annotated `Image`.
- [ ] Disqualify README-only / “call other agent” entries.

## Ensemble-specific claims to verify

- Ablation: YOLO-only vs classical-only vs full ensemble on the same split.
- Verifier pickle must exist (`models/verifier.pkl`) or heuristic scores are weaker.
- `vision_msgs` optional at runtime; JSON fallback must not hide missing detections in race scoring.

## Self-eval snapshot (Agent #10, 2026-10-09)

Split: `vial_dataset/split_70_10_20_latest2` **test** (54 pos / 2 empty).

| Variant | Acc | Recall | empty FP | mAP50 | FPS |
|---|---:|---:|---:|---:|---:|
| Full ensemble | 0.893 | 0.907 | 1 | 0.872 | 5.3 |
| Classical-only | 0.179 | 0.167 | 1 | 0.12 | 27 |
| YOLO-only (ref) | 0.893 | 0.907 | 1 | — | 6.3 |

Honest take: on labeled stills, ensemble **ties** fine-tuned YOLO (does not beat it). Classical-only is weak for full lid→cup boxes. Crown a winner only after live/dual-cam + empty-bench stress where cascade recall matters.
