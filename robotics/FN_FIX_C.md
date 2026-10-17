# FN_FIX_C — Asymmetric weak-cam + message honesty

**Bug:** False `"No vial in view"` when the weak camera (or gated proposals) still had vial candidates.

## Approach

1. **Load `camera_gates.json` at runtime** so cam1 weak YOLO conf **0.05** and relaxed structure floors are not dead documentation.
2. **Thread `gates` through `has_white_cap` → `usable_detection`** so weak aspect/edge floors are not silently replaced by strong defaults (this zeroed real borderline vials).
3. **Message honesty:** if YOLO/cap proposals exist but fail clear-vial gates → overlay **`Low confidence`**; only say **`No vial in view`** when there are truly no candidates.

## Code changes (`depthcamera.py`)

| Area | Change |
|------|--------|
| `load_camera_gates_file` / `weak_gate_profile` / `strong_gate_profile` | Read `camera_gates.json`; weak defaults from `cam1_weak` |
| `has_white_cap(..., gates=)` | Passes gates into `usable_detection` |
| `OrbbecDepthCamera.miss_reason` | `"no_vial"` \| `"low_confidence"` |
| `empty_view_message` / `annotate(..., miss_reason=)` | Honest empty-scene banner |
| `locate` | Detects at `gates.yolo_conf`; tracks raw candidates for miss_reason |

## Tests

```powershell
cd E:\Desktop\Jessica\Camera
python -m unittest test_asymmetric_gates.py -v
```

New cases: JSON weak conf, gates threaded into `has_white_cap`, empty-view copy, locate miss_reason.

## Constraints

- No dataset wipe.
- Independent dual unchanged (no peer invent boxes).
- Empty/wall rejection still required (opaque wall unit test kept).

## Run live

```powershell
python depthcamera.py --dual-independent
```
