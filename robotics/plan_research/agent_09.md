# Research Agent 09 — UKF Multi-View 3D State Fusion

**Theme:** Kalman / Bayesian fusion of dual-camera measurements (filter on 3D position or box state)  
**Primary method:** **Unscented Kalman Filter (UKF) multi-view fusion** of per-camera 2D boxes (and optional depth) into a shared 3D object state  
**Secondary / production pattern:** NVIDIA DeepStream **MV3DT** Kalman collaborative fusion (confidence-weighted peer measurements)

---

## Overview

Dual Orbbec views today produce independent vial boxes and per-camera distances (`locate_pair`); there is no shared 3D state. A strong next step is a **single Bayesian filter** whose state is the vial’s 3D position (and optionally size / velocity), updated **sequentially** by each camera’s measurement.

The **Unscented Kalman Filter (UKF)** is the preferred Kalman variant when the measurement model is nonlinear (perspective projection of a 3D ellipsoid/box onto each image). Recent multi-camera work:

1. Predict the 3D state with a constant-velocity (or random-walk) motion model.
2. For each camera that has a detection, run a **UKF update** using that view’s 2D box (and/or 3D point from RGB-D).
3. After both cameras update, the posterior mean/covariance is the fused estimate.

This is Bayesian fusion in the classical Kalman sense: each camera contributes a likelihood through its measurement noise covariance; the filter optimally blends them under Gaussian assumptions.

**NVIDIA MV3DT** implements the same idea in production: each camera publishes 3D measurements in a common world frame; peers fuse with Kalman updates and **visibility-weighted** measurement covariances.

---

## Multi-angle / multi-view angle

| Aspect | How UKF / Kalman multi-view fusion helps |
|--------|------------------------------------------|
| **Occlusion / blink** | If cam0 loses the vial for a few frames, cam1 still updates; prediction holds the track until cam0 returns (better than sticky peer memory alone). |
| **Depth noise** | Clear cups give unreliable depth; fuse **box center + white-cap depth** with larger \(R\) on depth than on lateral position, and weight by peer visibility / confidence. |
| **Viewpoint disagreement** | Two YOLO boxes disagree in image space; the filter resolves them in **world 3D** via calibrated extrinsics, not naive averaging of distances. |
| **Box shape** | State can include ellipsoid axes (height/width/length) so full-vial aspect (1.0–1.65) is enforced in 3D, not only per-image heuristics. |
| **Temporal smoothness** | Process noise \(Q\) damps jitter from frame-to-frame YOLO flicker while still tracking hand-held motion. |
| **Gating** | Mahalanobis gating rejects wall/desk false boxes that are inconsistent with the current 3D posterior (complements clear-vial checks). |

**State sketch (minimal for Jessica):**

\[
\mathbf{x} = [x, y, z, \dot{x}, \dot{y}, \dot{z}]^\top
\]

Optional extensions: box height/width, or a 3D ellipsoid as in Ma et al. (2025).

**Measurement options per Orbbec:**

- **A (RGB-only):** 2D box → UKF with nonlinear projection \(h(\mathbf{x})\) (needs calibration).
- **B (RGB-D, simpler):** deproject white-cap pixel + depth → 3D point measurement; linear/near-linear KF/EKF update in world frame (needs relative extrinsics between cam0 and cam1).
- **C (hybrid):** 3D point from depth when valid; fall back to bearing/box-only update when depth is a hole/far (matches clear-cup behavior).

---

## Pros

- Principled fusion of dual Gemini 335L measurements; replaces ad-hoc “no averaging” with uncertainty-aware fusion.
- Handles one-camera miss without dropping the shared track (predict + single-view update).
- UKF handles nonlinear camera projection without Jacobian coding (vs EKF).
- Covariance gives a live **confidence** for distance display / reach-limit gating (1.5 m).
- Scales from 2 cameras to N with the same sequential-update loop.
- Compatible with existing YOLO + clear-vial gating: filter sits **after** `locate_pair` detections.
- Production reference exists (DeepStream MV3DT); research reference exists (UKF multi-cam ellipsoid fusion).

## Cons

- Needs **extrinsic calibration** (or hand–eye / chessboard) between the two Orbbecs in a shared world frame; without it, fusion is wrong.
- Process/measurement noise (\(Q\), \(R\)) must be tuned; bad \(R\) can trust a lying depth hole.
- UKF is heavier than sticky peer memory (still real-time for one vial).
- Assumes roughly Gaussian noise; multimodal YOLO failures need gating + existing clear-vial reject.
- Full ellipsoid-from-boxes pipeline is overkill if only cap distance matters—prefer 3D point state first.
- Clock skew between USB cameras: treat updates as asynchronous (update on arrival) like RGB-D camera-network UKFs.

---

## Orbbec apply (Jessica / dual Gemini 335L)

1. **Calibrate once:** relative pose between OpenCV indexes 0 and 1 (checkerboard visible to both, or manufacturer extrinsics if fixed mount).
2. **Keep** YOLO + clear-vial + `fit_vial_shape` per camera as measurement generators.
3. **Per dual tick:**  
   - Predict shared vial filter.  
   - For each camera with a valid fix: build measurement \(z\) = white-cap 3D point in world (deproject depth at lid) **or** 2D box; set \(R\) larger when depth is invalid/hole.  
   - UKF/EKF update; skip camera if no detection (or only soft-retry peer assist as today).
4. **Display:** one fused distance (and optional per-cam overlay boxes still from YOLO). Do **not** average raw cm blindly—use filter mean.
5. **Gating:** reject updates whose Mahalanobis distance exceeds threshold (wall logos, mason jars that slip past YOLO).
6. **Start simple:** constant-velocity KF on 3D cap position with linear depth measurements (option B); upgrade to UKF+box if calibration and box-only frames matter.
7. **Fits plan.md:** dual same-time boxing stays; fusion layer sits on top of `locate_pair` outputs. Sticky peer memory can remain for **detection assist**; Kalman owns **3D state**.

---

## Algorithm sketch (dual tick)

```
x, P = predict(x, P, dt, Q)
for cam in (0, 1):
    det = locate_result[cam]   # after clear-vial / peer soft-retry
    if det is None: continue
    z, R = measurement_from_orbbec(det, depth, K_cam, T_world_cam)
    if mahalanobis(z, x, P, R) > gate: continue
    x, P = ukf_update(x, P, z, R, h)   # or kf_update if z is 3D
fused_distance = ||x[0:3] - cam_origin|| or height of white cap
```

---

## Citations

1. **Ma, L. V., Fatima, U., Chriv, T. S., Imran, H., & Jeon, M.** (2025). *3D Ground Truth Reconstruction from Multi-Camera Annotations Using UKF*. arXiv:2511.17609.  
   https://arxiv.org/abs/2511.17609 · https://arxiv.org/pdf/2511.17609  
   — Sequential UKF updates per camera; 3D ellipsoid state from 2D boxes; occlusion-aware multi-view fusion.

2. **Julier, S. J., & Uhlmann, J. K.** (2004). Unscented filtering and nonlinear estimation. *Proceedings of the IEEE*, 92(3), 401–422.  
   https://doi.org/10.1109/jproc.2003.823141  
   — Foundational UKF reference for nonlinear measurement models (camera projection).

3. **NVIDIA.** Multi-View 3D Tracking (MV3DT) — DeepStream documentation.  
   https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_MV3DT.html  
   — Production Kalman fusion of peer 3D measurements; visibility-weighted \(R_p = (2 - \mathrm{peerVisibility}) \cdot R_d\); multi-update then predict.

4. **Carraro, M., Munaro, M., Burke, J., & Menegatti, E.** (2019). *Real-time Tracking-by-Detection of Human Motion in RGB-D Camera Networks*. arXiv:1907.12112.  
   https://arxiv.org/abs/1907.12112 · https://ar5iv.labs.arxiv.org/html/1907.12112  
   — Multi RGB-D sensor UKF/Kalman fusion; asynchronous camera updates; outlier / confidence adaptation (relevant to dual Orbbec USB timing).

5. **Bredereke, L., Hartmann, Y., & Schultz, T.** (2025). *A Modular Pipeline for 3D Object Tracking Using RGB Cameras*. arXiv:2503.04322.  
   https://arxiv.org/abs/2503.04322 · https://arxiv.org/html/2503.04322  
   — YOLO detections → multi-camera **EKF** 3D trajectories with covariance as confidence; practical modular pattern.

6. **Nguyen, L. V., et al.** (2024). Track Initialization and Re-Identification for 3D Multi-View Multi-Object Tracking. *Information Fusion*. arXiv:2405.18606.  
   https://arxiv.org/abs/2405.18606  
   — Full Bayesian MV-MOT (GLMB); heavier than needed for one vial, but shows Bayesian multi-view track management.

7. Related implementation notes:  
   - https://github.com/LarsBredereke/object_tracking (EKF multi-cam + YOLO)  
   - https://github.com/linh-gist/3D-Visual-MOT (multi-view Bayesian MOT)

---

## Verdict for plan.md

**Recommend:** **UKF / Kalman multi-view fusion of dual Orbbec measurements** into one 3D vial (cap) state—start with linear 3D-point KF after depth deprojection; use UKF if fusing raw 2D boxes through projection. Keep current `locate_pair` for detection; add filter for fused distance and temporal stability.
