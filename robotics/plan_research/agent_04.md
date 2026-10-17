# Research Agent 04 — ChArUco Stereo Extrinsic Calibration + Hartley–Sturm Optimal Triangulation

**Theme:** Extrinsic calibration + 3D projection / triangulation of detections into a shared world frame  
**Repo fit:** Dual Orbbec Gemini 335L (`depthcamera.py --dual` / `locate_pair`); today distances stay per-camera with no world-frame fusion.

---

## Overview

**Method name:** ChArUco-based stereo extrinsic calibration + Hartley–Sturm (L2-optimal) triangulation into a shared world frame (camera-0).

The pipeline has two stages:

1. **Offline extrinsic calibration** — Place a ChArUco board in the overlapping FOV of both Gemini 335L RGB streams. Detect unique corner IDs independently per view (partial board visibility is OK). Intersect common IDs per synchronized pair, then run `cv2.stereoCalibrate` with `CALIB_FIX_INTRINSIC` to estimate only the relative pose `(R, T)` from camera 0 → camera 1. Fix camera 0 as the world origin: `R0 = I`, `T0 = 0`.

2. **Online 3D localization** — For each dual tick where both cameras report a vial (or peer-assisted fix), take a corresponding 2D point (e.g. white-cap centroid / box center), undistort with each camera’s `K, D`, build projection matrices `P0 = K0 [I|0]` and `P1 = K1 [R|T]`, then triangulate. Prefer Hartley–Sturm optimal triangulation (minimize squared reprojection error subject to the epipolar constraint) over plain DLT mid-point; OpenCV’s `triangulatePoints` is the practical linear baseline.

Metric scale comes from the measured ChArUco square size (and, optionally, cross-checks against Orbbec depth on the opaque white lid).

---

## Why this fits multi-angle / dual-Orbbec

| Need | How this method addresses it |
|------|------------------------------|
| Two fixed cameras, shared metric frame | Stereo extrinsics `(R,T)` put both detections in camera-0 coordinates |
| Limited FOV overlap | ChArUco unique IDs allow **partial** board visibility (OpenCV multiview guidance) |
| Sync for calibration & triangulation | Gemini 335L 8-pin multi-device sync; time difference ≤ ~3 ms with AE off |
| Vial is a sparse detection, not dense stereo | Triangulate **one** correspondence (lid/box center), not full disparity maps |
| Clear cup fails depth | World XYZ from RGB triangulation; depth remains a lid confidence cue |
| Already have dual same-tick boxing | `locate_pair` supplies paired fixes; this adds geometry after detection |

---

## Algorithm sketch (docs-level)

### A. Intrinsics (once per camera)

- Prefer Orbbec SDK device calibration / factory `K, D` for RGB, **or** single-camera `calibrateCamera` on a dense ChArUco capture.
- Keep depth→color (D2C) hardware alignment as-is for per-cam depth checks; world fusion uses RGB extrinsics.

### B. Extrinsics (offline, synchronized pairs)

1. Capture N synchronized RGB pairs with ChArUco visible to both (move board through the workspace volume).
2. Detect ChArUco corners + IDs per image; keep only **matched IDs** for that pair.
3. `stereoCalibrate(..., flags=CALIB_FIX_INTRINSIC)` → `R, T, E, F`, report RMS.
4. Persist `{K0,D0,K1,D1,R,T}` (and optional `F` for epipolar gating).

OpenCV 5+ also offers `calibrateMultiview` (MST pairwise init + global refine) if more than two cameras are added later; for two Gemini units, classic `stereoCalibrate` is sufficient.

### C. Online projection / triangulation

1. From each paired vial box, pick `u0`, `u1` (cap center preferred — matches current distance policy).
2. Undistort → normalized or pixel coords consistent with `P`.
3. Optional: reject pairs with large Sampson distance to `F` (bad correspondence / wrong peer).
4. Triangulate → homogeneous `X`; dehomogenize to metric `(X,Y,Z)` in camera-0 frame.
5. Optionally refine with Hartley–Sturm (6th-degree polynomial / L2 optimal) when noise is non-trivial.
6. Report one shared range (e.g. `‖X‖` or Z along cam-0 axis) instead of two independent depth readings.

### D. Depth fusion (optional, Orbbec-specific)

- Project world `X` into each color frame; sample lid depth where valid.
- Use triangulation as primary world estimate; use per-cam depth to gate clear-cup / reject mason jars (existing live logic).

---

## Pros

- **Metric shared world frame** — Ends dual-cam “two distances, no average” limitation; enables one vial pose for robotics / reach checks.
- **Works with sparse YOLO boxes** — No need for dense stereo matching between the two Orbbecs.
- **Partial-overlap friendly** — ChArUco IDs beat plain checkerboards when cams are widely angled.
- **Mature stack** — OpenCV `calib3d` + Hartley–Zisserman theory; well documented samples.
- **Hardware sync available** — Gemini 335L multi-device sync reduces motion-induced extrinsics error.
- **Scalable path** — Same world-frame idea extends to `calibrateMultiview` if a third camera is added.
- **Complements existing peer assist** — Geometry sits *after* `locate_pair`; detection logic stays intact.

## Cons

- **Requires a calibration session** — Board, sync, and careful capture; extrinsics drift if mounts move.
- **Correspondence quality** — Wrong peer box → wrong 3D; needs epipolar / Sampson gating and sticky-peer discipline.
- **Baseline / angle sensitivity** — Nearly parallel rays (vial far, cams close) inflate depth uncertainty along the ray.
- **`triangulatePoints` is linear-DLT** — Suboptimal vs Hartley–Sturm under asymmetric noise; may need a small optimal-refinement step.
- **Sync still matters live** — Soft USB timing without HW sync degrades triangulation of moving scenes.
- **Intrinsics mismatch** — If factory `K` differs from the stream resolution/crop used in live code, world scale biases appear.
- **Does not replace clear-vial depth cues** — Triangulation alone won’t detect opaque jars; keep lid+depth gating.

---

## Orbbec Gemini 335L apply notes

1. **RGB as calibration/triangulation modality** — Run ChArUco and YOLO on color; use Orbbec D2C depth only for lid confidence and existing clear-cup checks.
2. **World origin = camera 0** — Match OpenCV conventions; document which physical unit is index 0 vs 1 (`--list-cameras`).
3. **Enable multi-device sync** — Primary/secondary via 8-pin; prefer AE off during calibration captures; datasheet cites ≤3 ms RGB sync when AE off.
4. **Separate USB 3 ports** — Already required by dual mode; bandwidth affects frame pairing reliability.
5. **Point feature = white cap** — Aligns with plan.md (“distance measured on the white cap”); triangulate cap centers, not cup bottoms.
6. **Live integration point** — After `locate_pair` returns both fixes, if both have valid boxes, triangulate once and overlay a shared world distance (keep per-cam depth as secondary debug).
7. **Validation** — Place vial on a known bench mark; compare triangulated XYZ vs tape measure and vs each cam’s lid depth.
8. **Recalibrate trigger** — Any tripod/clamp move, or RMS / cross-depth disagreement above a threshold.

---

## Citations

1. OpenCV — *Multi-view Camera Calibration Tutorial* (ChArUco partial observation, sync requirement, projection matrices → triangulation).  
   https://docs.opencv.org/5.0/tutorials/calib3d/camera_multiview_calibration/multiview_calibration.html

2. OpenCV — *Camera Calibration and 3D Reconstruction* (`stereoCalibrate`, `CALIB_FIX_INTRINSIC`, `triangulatePoints`, `projectPoints`).  
   https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html

3. OpenCV sample — `multiview_calibration_sample.cpp` (`calibrateMultiview` usage).  
   https://fossies.org/linux/opencv/samples/cpp/multiview_calibration_sample.cpp

4. Hartley & Sturm — *Triangulation* (CVIU 1997): L2-optimal non-iterative triangulation via 6th-degree polynomial.  
   https://perception.inrialpes.fr/Publications/1997/HS97/HartleySturm-cviu97.pdf

5. Hartley & Zisserman — *Multiple View Geometry in Computer Vision* (2nd ed.), esp. Ch. 12 Structure computation / optimal triangulation.  
   https://www.cambridge.org/us/universitypress/subjects/computer-science/computer-graphics-image-processing-and-robotics/multiple-view-geometry-computer-vision-2nd-edition

6. Practical walkthrough — Stereo calibration + triangulation with OpenCV/Python (world = cam1 identity).  
   https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html

7. Orbbec Gemini 335L / 330 series — Multi-camera sync (8-pin), HW timestamps, D2C alignment, device calibration data.  
   - https://store.orbbec.com/products/gemini-335l  
   - https://www.orbbec.com/docs/g330-about-orbbec-gemini-330-series/  
   - Datasheet (multi-camera sync section): https://d1cd332k3pgc17.cloudfront.net/wp-content/uploads/2024/06/Orbbec-Gemini-330-Series-Datasheet-v1.1-2.pdf

8. Stewenius et al. — *How Hard is 3-view Triangulation Really?* (closed-form 3-view L2; relevant if a third camera is added).  
   https://www.robots.ox.ac.uk/~vgg/publications/2005/Stewenius05b/stewenius05b.pdf

---

## Verdict for Jessica Camera

**Strong default** for the dual-Orbbec vial tracker when the goal is one metric 3D position: calibrate once with ChArUco + `stereoCalibrate(CALIB_FIX_INTRINSIC)`, then triangulate paired lid points each dual tick. Keep current YOLO / clear-vial / sticky-peer stack; add world-frame geometry as a post-`locate_pair` stage rather than replacing detection.
