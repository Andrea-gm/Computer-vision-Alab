# FN supervisor — false “not in view”

**Date:** 2026-10-02  
**Winner:** **Agent A** (`FN_FIX_A.md`) — `soft_clear` on independent / standalone

## Ranking

| Rank | Agent | Approach | FN relief | Empty/wall reject | Notes |
|-----:|:-----:|----------|-----------|-------------------|-------|
| **1** | **A** | Independent `soft_clear` + conf **0.18** | Strong (unlocks soft path without peer) | Unit-tested | Primary root cause for `--dual-independent` |
| 2 | B | Cap-first `detect_vial_by_cap` fallback | Strong when YOLO misses | Unit-tested | Complementary; no peer invent |
| 3 | C | Weak JSON gates + message honesty | Helps weak cam + UI honesty | Opaque wall test kept | Alone does not accept weak-depth vials |

## Applied on MAIN

All three landed cleanly in `depthcamera.py` (no conflict markers). **A** is the ranking winner; **B+C** retained as compatible layers.

Preserved:
- Independent dual: `locate_standalone` / no peer-invented boxes
- Weights: `find_weights` prefers `runs/vial-live` then `runs/bon-winner`
- `plan.md` section: **False not-in-view fix**

## Verification

```
python -m unittest test_cap_fallback test_independent_dual test_asymmetric_gates test_locate_pair test_dual_review test_find_weights
→ Ran 69 tests — OK
```

## Live

```powershell
cd E:\Desktop\Jessica\Camera
python depthcamera.py --dual-independent
```
