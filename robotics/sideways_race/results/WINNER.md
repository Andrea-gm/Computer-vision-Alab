# Sideways vial race winner

Full evaluation run: `2026-10-09T21:24:09Z`  
Dataset: 40 sideways positives and 36 negatives (76 samples total)  
Primary ranking: recall at IoU ≥ 0.5, then fewer false positives, then mean IoU.

| Rank | Method | Recall@0.5 | FP / 36 | FP rate | Mean IoU |
|---:|---|---:|---:|---:|---:|
| 1 | `I_hard_robust` | **0.9500** (38/40) | **0** | **0.0000** | **0.8448** |
| 2 | `J_research` | 0.9250 (37/40) | 0 | 0.0000 | 0.7545 |
| 3 | `G_ensemble` | 0.9000 (36/40) | 0 | 0.0000 | 0.7014 |
| 4 | `E_fuse_cap_edge` | 0.7500 (30/40) | 0 | 0.0000 | 0.6139 |
| 5 | `C_depth_pca` | 0.7500 (30/40) | 0 | 0.0000 | 0.5774 |
| 6 | `H_b2f_merge` | 0.7250 (29/40) | 0 | 0.0000 | 0.6869 |
| 7 | `B2_improved` | 0.7250 (29/40) | 0 | 0.0000 | 0.6316 |
| 8 | `D_edge_template` | 0.6750 (27/40) | 36 | 1.0000 | 0.5494 |
| 9 | `B_rotated_white_cap` | 0.6000 (24/40) | 0 | 0.0000 | 0.5117 |
| 10 | `F_cap_depth_hybrid` | 0.5750 (23/40) | 0 | 0.0000 | 0.5732 |
| 11 | `A_cap_horizontal_grow` | 0.4000 (16/40) | 4 | 0.1111 | 0.3509 |
| 12 | `baseline_depthcamera` | 0.3750 (15/40) | 8 | 0.2222 | 0.3379 |

## Winner

`I_hard_robust` wins outright under the declared lexicographic ranking with
38/40 positives found and no false positives. `H_b2f_merge`'s reported old-set
result was reproduced (1.0000 recall, 0 FP, 0.9887 mean IoU on 52 samples), but
it collapses to 0.7250 recall on the current hard-expanded set. The winner also
reproduces 1.0000 recall and 0 FP on the old set.

The live promotion remains an upright-first fallback after accepted YOLO and
cap-first proposals miss. ROS pins the same `I_hard_robust` module while
retaining the isolated subprocess worker, respawn behavior, YOLO default-off,
and the existing crash hardening.

Re-run from `Camera/`:

```powershell
python sideways_race/eval_sideways.py --regen
```
