# Parametric 3D vial model

> **Live size-score / lid@Z prior stays off.** Opt-in `--3d-reproject-assist` works when `stereo_extrinsics.json` is real (not PLACEHOLDER). Offline fit / score / view / mesh tools still work.

Geometry prior for the clear vial + white screw cap. Fitted millimeter size lives in `vial_3d_dims.json`. Offline identify helpers remain in `vial_3d_identify.py` for scoring stills and demos.

Library helpers such as `propose_boxes_from_3d_model` (full upright lid→cup invent) remain in `vial_3d_model.py` for offline/API use and are **not** called from live locate.

Fitted size today (**R3D race winner Agent E**, distance-stratified): **Ø54.22 × H72.69 mm** from **n_fit=216** / holdout **92** (seed 42; raw+review metric sidecars). Holdout label correctness (IoU≥0.5 of proposed box vs GT): **0.6087**. Full race: `runs/train-race/r3d-race/R3D_RACE_RESULT.md`.

Live: `python depthcamera.py --dual-independent` — YOLO + cap-first + edge assist + sideways-assist (default on).  
Weak cam defaults to index 0 (`--weak-camera 0|1` to swap).  
Both-agreement + geometric assist (needs real extrinsics):  
`python depthcamera.py --dual-independent --3d-reproject-assist`  
Physical/USB: see `DUAL_PHYSICAL_CHECKLIST.md`.

---

## Exact commands (copy-paste)

```powershell
cd E:\Desktop\Jessica\Camera

# 0) Best-of-5 3D prior race (70% fit / 30% holdout) — promotes winner dims
python runs\train-race\r3d-race\run_r3d_race.py

# 1) Refit lid+cup mm from ALL labeled stills + distance_mm sidecars (full pool; not the race)
python vial_3d_model.py fit

# 2) Print current dims + extrinsics status
python vial_3d_model.py show

# 3) Interactive 3D (drag to rotate)
python vial_3d_model.py view

# 4) Static PNG preview + Isaac Sim / Omniverse mesh
python vial_3d_model.py preview
python vial_3d_model.py export-mesh
# → assets\vial\vial.usda  (open in Isaac Sim) and vial.obj (MeshConverter)

# 5) Score a candidate box vs the 3D prior (offline; no live locate)
python vial_3d_identify.py score --box 100,120,48,70 --distance-mm 400

# 6) Score GT boxes from raw/review stills that have distance_mm
python vial_3d_identify.py score-stills --limit 20

# 7) Live dual (size-score OFF; edge assist ON; reproject opt-in)
python depthcamera.py --dual-independent
python depthcamera.py --dual-independent --3d-reproject-assist
python depthcamera.py --dual-independent --no-edge-assist

# 8) Lift one box → reproject into the other camera (demo; needs real extrinsics)
python vial_3d_model.py reproject-demo --box 100,120,48,70 --distance-mm 400
python vial_3d_identify.py reproject-still --meta path\to\sidecar.json

# 9) Stereo calib (hands-on board required)
python stereo_calib_stub.py status
python stereo_calib_stub.py howto
python stereo_calib_stub.py capture     # SPACE=save pair, Q=quit
python stereo_calib_stub.py calibrate   # writes stereo_extrinsics.json
```

---

## How live CV uses the 3D model

- **Size-score / lid@Z:** still off (`LIVE_3D_VIAL_ENABLED = False`).
- **`--3d-reproject-assist`:** on when extrinsics are real; proposes a reprojected box on the misser; local clear-vial / soft-clear / scale gates must accept. Independent dual uses peer IPC box+cm (not classic sticky soft invent).

Offline score path (unchanged): project expected lid diameter / total height at Z, soft score `0..1` via `score_box_vs_model` / `score_detection_vs_3d`.

---

## Live assist instead: edge / gradient

After YOLO and cap-first miss, `detect_vial_by_edges` may propose ROIs around bright low-sat lids with strong Sobel/Canny rim gradients. Proposals still must pass clear-vial / soft_clear / scale / within_reach. Dual-independent remains local-only. Disable with `--no-edge-assist`.

---

## A / B / C modes

| Mode | Command | What it uses |
|------|---------|----------------|
| **A (recommended dual)** | `python depthcamera.py --dual-independent` | YOLO + clear-vial gates + cap-first + **edge assist** (default). Peer IPC corroborates cm. Size-score off. |
| **A′ (no edge assist)** | `python depthcamera.py --dual-independent --no-edge-assist` | Cap/YOLO path only. |
| **B (reproject assist)** | `--dual-independent --3d-reproject-assist` | Geometric reproject when extrinsics real; local gates still apply. PLACEHOLDER extrinsics → refused. |

---

## Isaac Sim / Omniverse

Isaac Sim is **not required** for live CV. To inspect the same geometry in Sim:

```powershell
python vial_3d_model.py export-mesh
# Open assets\vial\vial.usda in Isaac Sim, or convert vial.obj with MeshConverter
```

---

## Files

| Path | Role |
|------|------|
| `vial_3d_model.py` | Fit / show / view / preview / export-mesh / reproject-demo + library API |
| `vial_3d_identify.py` | Offline score boxes / stills; filter + rank + lid@Z helpers (not wired live) |
| `vial_3d_dims.json` | Fitted Ø/H (race winner or `fit`) |
| `assets/vial/` | `vial.obj` / `vial.mtl` / `vial.usda` for Isaac Sim |
| `runs/train-race/r3d-race/` | Best-of-5 fit race + `R3D_RACE_RESULT.md` |
| `stereo_extrinsics.json` | Cam0→cam1 `R`,`t` (for offline reproject demos) |
| `stereo_calib_stub.py` | Status + ChArUco capture notes |
| `depthcamera.py` | Live locate: 3D unlinked; edge assist via `detect_vial_by_edges` |
| `plan_research/agent_3d_vial_model.md` | Design write-up |

Does **not** promote YOLO weights, wipe the dataset, or touch an in-progress train.
