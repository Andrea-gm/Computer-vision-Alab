# FN_FIX_B — Cap-first fallback (Agent B)

## Root cause

Live overlay shows **"No vial in view"** while the clear plastic vial + white screw cap is visible because:

1. **YOLO misses** the vial (glare / weak angle / conf just under the cam gate), **or**
2. YOLO proposes a box but **strict clear-cup depth gates reject** it (Orbbec fills cup depth / angle kills hole margin),
3. The existing `detect_vial_by_cap` recovery path still required the **same** strict `accepts_vial_box` / clear-depth gate, so lid+body proposals were dropped,
4. Independent dual / standalone **cannot** unlock peer sticky soft assist, and must not invent a box from the sibling camera.

Net: local RGB already had a white-lid + cup span, but depth-hole gating zeroed the fix → false not-in-view.

## Fix (cap-first)

When YOLO + strict gates leave no box:

1. Run `detect_vial_by_cap` (local color only).
2. Prefer strict clear-vial acceptance on those boxes.
3. If still empty, **accept** via `accepts_cap_fallback_box` / `cap_fallback=True`: white lid + capped look + interior structure + usable aspect — **no clear-depth required**.
4. Thread `cap_fallback` through fit / `choose_box` / final validation / brief hold so the soft accept is not stripped later.

Does **not** invent boxes from peer IPC or sticky peer memory.

## Files

- `depthcamera.py` — `accepts_cap_fallback_box`, `accepts_vial_box(..., cap_fallback=)`, `choose_box(..., cap_fallback=)`, `OrbbecDepthCamera.locate` cap-first recovery.
- `test_cap_fallback.py` — unit coverage for gates + locate without peer/soft_clear.

## Tests

```powershell
cd E:\Desktop\Jessica\Camera
python -m unittest test_cap_fallback.py test_independent_dual.py test_asymmetric_gates.py test_locate_pair.py test_dual_review.py -v
```

## Live check

```powershell
python depthcamera.py --dual-independent
```

No dataset wipe; weights path unchanged.
