# Agent — Parametric 3D vial model → identify from any camera

**Theme:** Build a lightweight geometric prior of the clear vial + white cap from labeled stills, then use that 3D hypothesis to help the weak Orbbec without more YOLO races.  
**Status (2026-10-02):** MVP ready for offline try — `vial_3d_model.py` + companion `vial_3d_identify.py` + `VIAL_3D_README.md` + `stereo_calib_stub.py`. Fitted dims ~Ø56×H75 mm (`vial_3d_dims.json`, 162 sidecars). Live assist remains **opt-in** (`--3d-reproject-assist`); dual-independent default stays peer-soft-free. **`stereo_extrinsics.json` is still PLACEHOLDER** — replace before trusting dual reproject.

---

## Chosen approach (practical)

**Parametric RGB-D vial** = opaque **white lid cylinder** stacked on a **clear cup cylinder**, sized in millimeters from labeled boxes + white-cap depth (when available).

Not chosen for MVP:

| Alternative | Why deferred |
|-------------|--------------|
| NeRF / 3D Gaussian | Heavy; needs many orbit views; overkill for one rigid vial class |
| Open3D mesh + ICP | Open3D not installed; clear cup depth holes make ICP brittle |
| Full MVTrans / MVDeTr | Needs extrinsics + train data; research Agents 03/08/10 already defer |

This matches research Agents **02** (epipolar), **04** (extrinsics + triangulate), and **01** (lid-3D associate) without replacing Attempt-E YOLO or BoN #3 gates.

---

## Algorithm steps

### A. Offline: fit the model from pictures

1. Load positives from `vial_dataset/raw/*.txt` (and optionally review stills with `distance_mm` sidecars).
2. For each sample with **box + distance_mm + image size**:
   - Approximate Gemini intrinsics if not stored (`fx,fy` from FOV prior, or live `CAP_PROP_OBSENSOR_*`).
   - Metric width/height: `w_mm = box_w * Z / fx`, `h_mm = box_h * Z / fy` (same idea as `vial_scale`).
   - Split height: lid ≈ top ~22% of full-vial box (white band); cup = remainder.
3. Robust median → `VialDimensions` (`lid_diameter_mm`, `lid_height_mm`, `cup_diameter_mm`, `cup_height_mm`).
4. Write `vial_3d_dims.json` (never touches YOLO weights / dataset wipe).

### B. Online: lift strong-cam detection → 3D hypothesis

1. Strong cam `locate` succeeds → white-cap depth `Z` via `measure_cap_mm` (already in stack).
2. Cap center pixel → camera-frame XYZ via `position_mm` / `pixel_to_cam`.
3. Attach `VialDimensions` → upright cylinder stack along camera −Y (image up) or gravity-up when known.

### C. Online: reproject into weak cam (gated assist)

1. Requires `stereo_extrinsics.json`: `R` (3×3), `t` (mm) mapping **cam0 → cam1** (or vice versa), plus optional `K0`/`K1`.
2. Transform hypothesis into weak-cam frame; project lid+cup AABB → pixel ROI / box.
3. **Opt-in only** (`--3d-reproject-assist`):
   - Use projected box as a **search ROI** / candidate proposal on the misser.
   - Candidate must still pass **local** clear-vial / soft-clear / cap gates (`accepts_vial_box`) — geometry proposes, gates accept.
   - Distances stay **per-camera** (measure lid depth on the accepting cam; never average).
4. Default `--dual-independent`: **off** — no inventing peer boxes (plan policy unchanged).

### D. Single-cam use (no extrinsics)

Even without stereo calib, the fitted mm template:

- Tightens `vial_scale` / aspect expectations toward this vial’s real size.
- Scores YOLO/cap proposals by 3D-consistency (width/height vs template at measured `Z`).

---

## How existing stills are used

| Asset | Use |
|-------|-----|
| `raw/*.jpg` + `*.txt` | Box shape / aspect; camera id from `*.json` when present |
| Review sidecars with `distance_mm` + `box` | Metric size fit (primary signal today — raw accept currently strips distance; MVP preserves it going forward) |
| Live RGB-D | Lift + clear-cup gates; not training NeRF |
| Paired same-tick dual stills (future) | Refine extrinsics / validate reprojection error |

Depth PNGs/NPYs are **not** required for the parametric path; white-cap `distance_mm` is enough.

---

## Calibration needed

| Level | Needed for | Capture |
|-------|------------|---------|
| **None** | Size template + single-cam 3D score | Already have labeled stills + distances in review |
| **Intrinsics** | Accurate mm size / project | Auto from Orbbec OpenCV props at live time; cache optional |
| **Extrinsics cam0↔cam1** | Reproject assist / epipolar band | ChArUco or chessboard held in **both** FOVs; rigid mounts; see Agent 04 |
| **Synced dual ticks** | Validate reproject live | Same-moment stills from both cams (classic `--dual` or dual capture script) |

Without extrinsics, ship size prior only; turn on reproject after one calib session.

---

## Pros / cons vs more YOLO fine-tune

| | 3D parametric prior | More YOLO FT (esp. weak-angle) |
|--|---------------------|--------------------------------|
| **Pros** | No holdout risk; helps weak cam via geometry; CPU-cheap; aligns with lid-depth truth | Can learn glare/angle appearance; no stereo mount assumption |
| **Cons** | Needs rigid dual mount + extrinsics for cross-cam ROI; bad calib → wrong ROI | Aggressive oversample already **hurt** test (81.8% vs 100%); data bias |
| **Policy fit** | Opt-in assist respects dual-independent default | Orthogonal; leave mild R3-04 alone |

**Verdict:** Prefer 3D assist as the weak-cam bridge; keep mild FT only if it clearly beats holdout. Do not restart failed oversample.

---

## MVP implementation plan

| Piece | Path / API |
|-------|------------|
| Core module | `vial_3d_model.py` |
| Fit CLI | `python vial_3d_model.py fit` → `vial_3d_dims.json` |
| Reproject API | `lift_detection_to_3d(...)`, `reproject_vial_to_camera(...)`, `assist_box_from_peer(...)` |
| Extrinsics stub | `stereo_extrinsics.json` (user fills after ChArUco) |
| Live flag | `depthcamera.py --3d-reproject-assist` (classic dual + optional independent IPC) |
| Sidecar enrich | On R-accept, keep `distance_mm` / `box` / size in raw `*.json` for future fits |
| Not in MVP | Open3D ICP, NeRF, promoting weak YOLO weights, soft peer invent without geometry |

### Live wiring (gated)

1. Load dims (+ extrinsics if present) when flag set.
2. After strong cam fix: lift → reproject → if weak miss, propose ROI box.
3. Weak `locate` may accept proposal only via existing `accepts_vial_box` / soft-clear — **not** classic sticky invent.
4. Overlay may tag `3d-roi` for debug; distances unchanged policy.

---

## Capture next (user)

1. **Orbit / multi-angle stills** of the same vial (both cams): R only on full lid→cup boxes.
2. **Same-tick dual pairs** (vial visible in both) for reproject QA.
3. **ChArUco or chessboard** session for `R,t` (and refresh if mounts move).
4. Optional: keep mounts rigid after calib; note which index is weak.

---

## Citations / stack links

- Epipolar peer gating: `plan_research/agent_02.md`
- ChArUco extrinsics: `plan_research/agent_04.md`
- WBF / lid-3D associate: `plan_research/agent_01.md`
- Live policy: `plan.md` Independent Dual Processes + BoN #3
