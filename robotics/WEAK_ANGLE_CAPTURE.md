# Weak-angle labels (`--weak-camera`)

## Which camera is weak?

Live default is **`--weak-camera 0`** (permissive gates / YOLO conf **0.05** on cam0; cam1 stays strong/tighter). Startup prints:

`Asymmetric gates: weak=cam0 (permissive), strong=cam1 (tighter). Override: --weak-camera N`

If the physically hard view is the other Orbbec, swap:

```powershell
python depthcamera.py --dual-independent --weak-camera 1
```

Profile keys in `camera_gates.json` (`cam0_strong` / `cam1_weak`) are **template names**, not index binding — `--weak-camera` chooses which index gets the weak profile.

## Current raw balance (2026-10-05 audit)

- `raw/`: **341** jpg (330 pos / 11 empty). Cam tags: **cam0≈110**, **cam1≈98**, **unknown≈133**.
- Near-even among tagged cams; weak-angle gap is mostly **hard poses / glare**, not sheer cam count.
- Waiting on user review (cannot auto-R without you):
  - `review/pending/` — **~33** stills (prefer **R** on weak-cam full-vial boxes)
  - `review/incorrect/` — **~258** wrong/partial boxes (**I** → drag lid-through-cup)

## What to capture (you must hold / aim the vial)

Focus on the **failing** camera (whichever index is weak):

1. Run live dual, keep vial in **both** FOVs, let auto-stills save.
2. After **Q**, review:
   - **R** — box is full clear vial + white cap (lid top through cup bottom)
   - **W** — not a vial (empty-bench hard negative)
   - **I** — vial present but box wrong → draw full vial
3. Bias R-accepts toward weak-cam stems / hard angle (glare, foreshortened cup, farther).
4. Do **not** rely on classic `--dual` soft peer invent; keep `--dual-independent`.
5. Sideways/fallen vials: also **R**/label a few end-cap stills (white lid left or right of clear body) so YOLO sees that pose.

```powershell
cd E:\Desktop\Jessica\Camera
python depthcamera.py --dual-independent
# If the wrong cam is tagged weak:
python depthcamera.py --dual-independent --weak-camera 1
python depthcamera.py --review-only
# After new labels:
python train_vial_model.py split --seed 42
python train_vial_model.py train --seed 42
python eval_asymmetric_split.py --weights runs\vial-ft\weights\best.pt
# Promote only on clear holdout beat (see TRAIN_PROMOTE.md)
```
