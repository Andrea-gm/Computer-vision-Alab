# FN Fix A — false “No vial in view” (Agent A)

**Status:** Patched on **MAIN** (`E:\Desktop\Jessica\Camera`). No worktree (`fn-fix-a-*` / `.cursor/worktrees` unavailable). Dataset not wiped. Distances still per-camera (never averaged). Overlay label remains **`vial`**.

## Root cause

1. Overlay text **`No vial in view`** is drawn in `annotate()` when `fix is None` (`depthcamera.py`).
2. Preferred live path is **`--dual-independent`** → each process calls **`locate_standalone`** → `locate(..., peer_confirmed=False)`.
3. Classic **`locate_pair`** can unlock a soft white-cap path via sticky **`peer_confirmed=True`** when clear-cup depth holes fail (glare / Orbbec fill-in).
4. Independent / single-cam had **no** soft path: accept required full **`looks_like_capped_clear_vial`** (white lid + structure + clear-cup depth holes). When the vial was clearly in RGB but cup depth holes were weak, every proposal was rejected → false negative.
5. Strong YOLO floor **0.25** also dropped borderline real vials before gates ran.

## Fix (live independent / single-cam)

| Change | Detail |
|--------|--------|
| `DEFAULT_YOLO_CONF` | **0.25 → 0.18** (CLI `--conf` default; `camera_gates.json` `cam0_strong.yolo_conf`) |
| `soft_clear` | New accept flag on `accepts_vial_box` / `choose_box` / `locate` |
| `locate_standalone` | Calls `locate(..., peer_confirmed=False, soft_clear=True)` |
| Soft accept rule | Same as peer soft assist: **white cap + capped look + interior structure** when strict clear-cup fails |
| Walls / empty | Still fail `has_white_cap` / structure (unit-tested) |
| `locate_pair` | Unchanged: sticky peer soft only; no soft_clear on strict pass |

## Tests / metrics

### Unit (authoritative for this FN)

```text
python -m unittest test_independent_dual test_asymmetric_gates test_dual_review test_locate_pair test_find_weights
→ Ran 64 tests — OK
```

Key cases:

- `SoftClearIndependentTests.test_soft_clear_recovers_weak_depth_holes` — filled cup depth → strict fail, `soft_clear=True` accept.
- `SoftClearIndependentTests.test_soft_clear_still_rejects_wall_and_empty_slab` — walls/slabs still rejected.
- `LocateStandaloneTests` — `peer_confirmed=False`, `soft_clear=True`; never calls `locate_pair`.
- Sticky peer soft tests still pass (classic `--dual`).

### Holdout YOLO eval (secondary; not soft_clear)

`eval_asymmetric_split.py` on `runs/bon-winner/weights/best.pt` → `fn_fix_a_metrics.json`:

- Weak conf **0.05**: **72.7%** test (8/11)
- Baseline conf **0.25**: **63.6%** test
- Empty negatives remain clean at weak conf in the eval harness

This eval scores YOLO boxes vs GT; it does **not** exercise live soft_clear RGB-D gates. Live FN relief is covered by the synthetic soft_clear tests above.

## Files touched (MAIN)

- `depthcamera.py` — conf default, `soft_clear`, `locate_standalone`
- `camera_gates.json` — strong `yolo_conf: 0.18`
- `test_independent_dual.py` — SoftClear + locate_standalone asserts
- `test_asymmetric_gates.py` — default conf map
- `test_dual_review.py` — soft_clear parity with peer soft
- `fn_fix_a_metrics.json` — holdout snapshot
- This note: `FN_FIX_A.md`

## How to verify live

```powershell
cd E:\Desktop\Jessica\Camera
python depthcamera.py --dual-independent
# or: python depthcamera.py --camera 0
```

Expect a **`vial`** box when the white-capped clear vial is in frame even if cup depth looks filled-in; blank walls should stay “No vial in view”.
