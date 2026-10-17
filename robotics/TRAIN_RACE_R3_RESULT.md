# TRAIN RACE Round 3 — RESULT (FINAL LEAN — 3 trainers) + all-rounds summary

**Status:** Round 3 complete (all 3 `RESULT.md`). Supervisor verified canonical `best.pt` on MAIN `split_70_10_20` (seed 42, 40/6/11) at conf 0.25, IoU≥0.5.

## Winner (R3 race)

- **Agent / id:** **#2** `train-r3-02-e05d17b9`
- **Strategy:** ≤15-epoch dual-seed FT (vial-live vs r2-winner); kept **vial-live** base; AdamW + cosine low LR; freeze 5; weak-angle geometry aug; conf-robust eval 0.25 + 0.18
- **Test accuracy:** **100.0%** (11/11)
- **Recall (positives):** 1.0 | **Empty FP:** 0
- **Val accuracy:** 100.0%
- **Model size:** 22,512,938 bytes (smallest among 100% R3 ties)
- **Source weights:** `C:\Users\Ceder-Alab\.cursor\worktrees\train-r3-02-e05d17b9\runs\train-r3-02\weights\best.pt`
- **Archived to MAIN:** `runs/train-race/r3-winner/weights/best.pt`
- **SHA256:** `320DC562F62D568C2BC8009D9F98D45BAF4F0582037C73D131A9178CED424511`

## Live weights decision

- **Prior live (R1):** 100% test / recall 1.0 / FP 0 — SHA256 `B4338780…`
- **R3 best:** pure primary tie with live; **better secondary = smaller model** vs R3-01 (also 100%); #2 also documents test@0.18 = 100%
- **`runs/vial-live/weights/best.pt` updated** to R3-02 (final polish promote allowed on pure tie)
- `depthcamera.find_weights` still prefers `vial-live` → `r1-winner` → `bon-winner`

## R3 ranking (test → recall → fewer empty FP → val → smaller size)

| Rank | Agent | Worktree | Test % | Recall | FP | Val % | Size | Incomplete? |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2 | `train-r3-02-e05d17b9` | 100.0 | 1.0 | 0 | 100.0 | 22,512,938 | no |
| 2 | 1 | `train-r3-01-dfa42b67` | 100.0 | 1.0 | 0 | 100.0 | 22,513,834 | no |
| 3 | 3 | `train-r3-03-7638e594` | 90.9 | 0.9 | 0 | 100.0 | 22,512,938 | no |

## All three rounds (holdout protocol: conf 0.25, IoU≥0.5, seed 42, 40/6/11)

| Round | Winner | Test % | Recall | Empty FP | Val % | Became vial-live? |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **R1** | `train-r1-07-cf168274` (MVAug mild + freeze/unfreeze phase1) | 100.0 | 1.0 | 0 | 100.0 | **yes** (first live) |
| **R2** | `train-r2-02-c13f659b` (AdamW+cos freeze-5 weak-angle) | 100.0 | 1.0 | 0 | 100.0 | no (tie, no clear beat) |
| **R3** | `train-r3-02-e05d17b9` (dual-seed lean; kept vial-live base) | 100.0 | 1.0 | 0 | 100.0 | **yes** (tie + smaller model / final polish) |

### Archives on MAIN

| Artifact | Path |
| --- | --- |
| Live | `runs/vial-live/weights/best.pt` (= R3 winner) |
| R1 archive | `runs/train-race/r1-winner/weights/best.pt` |
| R2 archive | `runs/train-race/r2-winner/weights/best.pt` |
| R3 archive | `runs/train-race/r3-winner/weights/best.pt` |

Detail write-ups: `TRAIN_RACE_R1_RESULT.md`, `TRAIN_RACE_R2_RESULT.md`.

## Notes

- No dataset wipe; trainers used MAIN `vial_dataset` / seed-42 split copies only.
- FN soft-clear + `--dual-independent` behavior untouched.
- Interim accelerator (`runs/train-race/r3-interim-eval.json`) saw #2 hit 100% before RESULT.md; final ranking uses each agent’s **canonical** `runs/train-r3-0N/weights/best.pt` (newest nested `best.pt` can be a discarded seed/phase).

## Dual-independent (live)

```powershell
cd E:\Desktop\Jessica\Camera
python depthcamera.py --dual-independent
```
