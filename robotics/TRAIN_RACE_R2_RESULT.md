# TRAIN RACE Round 2 — RESULT (LEAN — 3 trainers)

**Status:** Round 2 complete (all 3 `RESULT.md`). Supervisor also ran interim holdout eval during train (same gate as R1).

## Winner (R2 race)

- **Agent / id:** **#2** `train-r2-02-c13f659b`
- **Strategy:** ≤20-epoch fine-tune from r1-winner; AdamW + cosine low LR; freeze 5 layers; weak-angle geometry aug
- **Test accuracy:** **100.0%** (conf 0.25, IoU≥0.5 on MAIN `split_70_10_20` 40/6/11 seed 42)
- **Recall (positives):** 1.0 | **Empty FP:** 0
- **Val accuracy:** 100.0%
- **Source weights:** `C:\Users\Ceder-Alab\.cursor\worktrees\train-r2-02-c13f659b\runs\train-r2-02\weights\best.pt`
- **Archived to MAIN:** `runs/train-race/r2-winner/weights/best.pt`

## Live weights decision

- **R1 baseline (live):** 100% test, recall 1.0, empty FP 0 (`runs/vial-live` = `runs/train-race/r1-winner`)
- **R2 best:** ties R1 on primary gate (test IoU≥0.5 → recall → empty FP) — **does not clearly beat**
- **`runs/vial-live/weights/best.pt` left unchanged** (still R1 SHA256 `B4338780…`)
- `depthcamera.find_weights` still prefers `runs/vial-live` then `r1-winner` then `bon-winner`

## Ranking (accuracy → recall → fewer empty FP → val)

| Rank | Agent | Worktree | Test % | Recall | FP | Val % | Incomplete? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | `train-r2-02-c13f659b` | 100.0 | 1.0 | 0 | 100.0 | no |
| 2 | 1 | `train-r2-01-c0d5873d` | 90.9 | 0.9 | 0 | 100.0 | no |
| 3 | 3 | `train-r2-03-b9f67015` | 81.8 | 0.8 | 0 | 100.0 | no |

## Notes

- No dataset wipe; trainers used MAIN `vial_dataset` / seed-42 split copies only.
- FN soft-clear + `--dual-independent` behavior untouched.
- Interim accelerator eval (`runs/train-race/r2-interim-eval.json`) already showed #2 tied with R1 before RESULT.md landed; final RESULT.md confirms.

## Dual-independent (live)

```powershell
cd E:\Desktop\Jessica\Camera
python depthcamera.py --dual-independent
```

## Round 3 design (next)

- Keep live at R1/vial-live (100%) unless a candidate **clearly** beats test IoU≥0.5, then recall, then lower empty FP.
- Lean trainers again; fine-tune from `r1-winner` / `vial-live` (or `r2-winner` only as an alternate seed — same ceiling today).
