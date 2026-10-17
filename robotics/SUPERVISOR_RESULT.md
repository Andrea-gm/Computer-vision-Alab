# SUPERVISOR_RESULT — BoN vial ID race (10 strategies)

**Repo:** `E:\Desktop\Jessica\Camera`  
**Date:** 2026-10-02  
**Winner:** **#3 — Asymmetric per-camera thresholds** (`bon-vid-03-ba9c8430`)

## Ranking (test IoU≥0.5 accuracy, then FP, then simplicity, then `locate_pair`)

| Rank | # | Strategy | Test acc % | FP (empties) | Notes |
|-----:|--:|----------|-----------:|-------------:|-------|
| **1** | **3** | **Asymmetric per-camera thresholds** | **100.0** | **0** | Weak conf 0.05 primary; strong 0.25 still 83.3%. Simplest 100% winner. |
| 2 | 6 | YOLO∪cap ensemble NMS | 100.0 | 0 | Official needs winner→trained **fallback** weights; alone 83.3%. More complex. |
| 3 | 1 | YOLO retrain baseline | 83.3 | 0 | Fine-tune prior bon-winner; scratch nano 16.7%. |
| 3 | 4 | Depth-hole clear-body scorer | 83.3 | 0 | Stills lack depth; synthetic clear/opaque/wall OK. |
| 3 | 5 | Aspect + geometry hard gates | 83.3 | 0 | Gates don’t hurt test; miss is YOLO-empty `vial_0005`. |
| 3 | 7 | Soft-clear dual / strict single | 83.3 | 0 | Dual soft helps weak-depth sim; RGB score unchanged. |
| 3 | 9 | Two-stage verify | 83.3 | 0 | Cap+clear re-check; miss `vial_0005`. |
| 8 | 8 | Background/wall FP rejection | 66.7 | 0 | Retrain+gates; flatness inactive on RGB stills. |
| 9 | 2 | Cap-first classifier | 50.0 | 0 | Color-only (no weights at eval); 0 FP. |
| 9 | 10 | Hybrid retrain + soft peer gates | 50.0 | 0 | Primary conf 0.25; soft conf 0.13 → 66.7%. |

Split for all agents: seed **42**, **20 / 3 / 6** (29 raw pairs). No `vial_dataset` wipe.

## Why #3 won

1. **Highest fair test accuracy** for the dual-cam failure mode (100% under the weak profile that live cam1 actually uses).
2. **FP control:** empty negatives stay 0 detections at conf 0.05.
3. **Simpler than #6** (no second YOLO weights cascade / union NMS stack).
4. **Preserves `locate_pair`** same-frame sticky peer soft + per-camera distances + `vial` overlay.
5. Directly targets “both cams see the vial; one almost always fails ID” via `CameraGateProfile` / `--weak-camera`.

## Applied into main

| File | Action |
|------|--------|
| `depthcamera.py` | Replaced with #3 (asymmetric gates + CLI) |
| `camera_gates.json` | Added (strong/weak profile snapshot) |
| `eval_asymmetric_split.py` | Added |
| `test_asymmetric_gates.py` | Added |
| `bon_vid_03_metrics.json` | Added (race metrics copy) |
| `plan.md` | Updated with winner section |
| `vial_dataset/` | **Untouched** (34 raw jpgs before/after) |
| Sibling worktrees | **Not deleted** |

## Verification on main

```
python -m py_compile depthcamera.py eval_asymmetric_split.py test_asymmetric_gates.py  # OK
python test_asymmetric_gates.py   # 7 OK
python test_locate_pair.py        # 5 OK
python -m unittest test_dual_review  # 26 OK
```

## Restart

```powershell
cd E:\Desktop\Jessica\Camera
python depthcamera.py --dual
```

Optional: `--weak-camera 1` (default), `--cam0-conf 0.25`, `--cam1-conf 0.05`.
