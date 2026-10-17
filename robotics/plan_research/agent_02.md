# Research Agent 02 — Epipolar Multi-View Detection Confidence (Isaac-Medina et al.)

**Theme:** Stereo / epipolar geometry + cross-view consistency  
**Primary method:** Multi-view Object Detection Using Epipolar Constraints (Isaac-Medina, Willcocks, Breckon — ICPR 2020)  
**Repo fit:** Dual Orbbec Gemini 335L RGB-D vial tracker (`depthcamera.py --dual`); keep Attempt-E YOLO weights; post-process only.

---

## Overview

Isaac-Medina et al. treat epipolar geometry as a **post-detector consistency score**, not a replacement detector. Pipeline:

1. Run a standard detector independently on each view (they used YOLOv3; any score-producing detector works).
2. Recover the **fundamental matrix** \(F\) between the two views — either from calibration (\(K, R, t\)) or by treating paired ground-truth box **centroids** as noisy correspondences (normalized 8-point + RANSAC).
3. For a detection centroid \(x\) in view A, form the epipolar line \(\ell' = Fx\) in view B. Measure the point-to-line distance of candidate centroids in B.
4. Model that distance as approximately Gaussian; convert it to a **multi-view epipolar confidence** (tail probability / p-value via `erfc`).
5. Multiply (or otherwise fuse) that geometric probability with the detector’s objectness×class score, then run NMS / accept-reject with the fused score.

Net effect: boxes that agree across views along epipolar lines are boosted; geometrically inconsistent false positives (wall logos, desk clutter, wrong-object boxes) are down-weighted or dropped.

Public reimplementation: [KostadinovShalon/yolo_multiview](https://github.com/KostadinovShalon/yolo_multiview).

Related practical metric (optional scoring upgrade once \(F\) is known): **normalized epipolar cross-distance** \(d'_\mathrm{cross}\) (Yang et al.), used in multi-camera YOLO matching — distance to each other’s epipolar lines, divided by box \(|w+h|\), with optional temporal match bonus ([labvisio multi-object triangulation](https://github.com/labvisio/Multi-Object-Triangulation-and-3D-Footprint-Tracking)).

---

## Multi-angle benefit

| Benefit | Why it helps dual-Orbbec vials |
|--------|--------------------------------|
| **Cross-view FP rejection** | A white-cap lookalike visible in only one cam fails the epipolar match → lower fused confidence. |
| **Peer validation without averaging depth** | Confirms “same physical vial” geometrically; distances stay per-camera (matches current plan). |
| **Works with clear / transparent objects** | Original paper targets X-ray (feature matching fails); same failure mode as clear plastic cups — box centroids still define \(F\) and match scores. |
| **Detector-agnostic** | Sits after YOLO + clear-vial gates; no retrain of `bon-winner`. |
| **Calibrated or uncalibrated path** | Prefer \(F\) from Orbbec intrinsics + a one-time stereo extrinsic; fallback: dual-labeled stills’ lid/box centers. |

---

## Pros / cons for clear vial + depth

### Pros
- Aligns with existing dual path: `locate_pair` already wants peer agreement; epipolar score is a principled upgrade over sticky peer memory alone.
- Cap/box center is a stable correspondence cue even when the clear cup returns bad depth.
- Cheap at runtime (3×3 matvecs + line distances) once \(F\) is cached.
- Does not require ground-plane assumptions (unlike some multi-view people detectors).

### Cons
- Needs a usable \(F\): cameras must stay rigid (or re-estimate after bumping mounts). Wide baseline + small vial → large pixel epipolar error if \(F\) is sloppy.
- Centroid-to-true-object-center bias (paper’s \(\Psi\)): lid-only vs full-vial boxes skew the match — mitigate by using **white-cap center** (or top-third of box) as the correspondence point, consistent with depth measurement.
- If one cam has no detection, geometry cannot “invent” a box (only softens/hardens peers); still combine with current soft-retry / sticky memory.
- Epipolar alone cannot distinguish two coplanar objects on the same line; keep class/clear-cup depth gates.

---

## Citations (URLs)

1. **Primary — Isaac-Medina et al., ICPR 2020**  
   PDF: https://breckon.org/toby/publications/papers/isaac20multiview.pdf  
   DOI: https://doi.org/10.1109/ICPR48806.2021.9413007  
   Durham record: https://durham-repository.worktribe.com/output/1141503/multi-view-object-detection-using-epipolar-constraints-within-cluttered-x-ray-security-imagery  
   Code: https://github.com/KostadinovShalon/yolo_multiview

2. **Hartley & Zisserman — Multiple View Geometry** (foundational \(x'^T F x = 0\))  
   https://www.robots.ox.ac.uk/~vgg/hzbook/

3. **LearnOpenCV — Epipolar geometry intro**  
   https://learnopencv.com/introduction-to-epipolar-geometry-and-stereo-vision/

4. **Yang et al. — normalized epipolar cross-distance** (unified multi-view multi-person tracking)  
   https://link.springer.com/article/10.1007/s41095-021-0241-9  
   (Cited as the \(d'_\mathrm{cross}\) normalization used in later YOLO multi-cam matchers.)

5. **Practical YOLO + epipolar matching (SIBGRAPI / labvisio)**  
   Paper PDF: https://sol.sbc.org.br/index.php/sibgrapi_estendido/article/download/38305/38079/  
   GitHub: https://github.com/labvisio/Multi-Object-Triangulation-and-3D-Footprint-Tracking

6. **MVDet — epipolar re-ID without ground plane** (heavier alternative; not primary)  
   https://link.springer.com/article/10.1007/s10044-023-01168-6

---

## Apply-to-Orbbec sketch (no detector rewrite)

1. **Calibrate once:** With both Gemini 335L fixed, collect a chessboard (or dual-labeled vial stills). Build \(F_{01}\) from OpenCV stereo calibrate, or from Orbbec \(K\) + relative \(R,t\): \(F = K_2^{-T}[t]_× R K_1^{-1}\). Persist to a small JSON next to the app.
2. **Per dual tick (inside / beside `locate_pair`):** After strict clear-vial gating on each cam, take correspondence points = **cap centers** (prefer depth sample point already used for range).
3. **Score pairs:** For each candidate pair \((B_0, B_1)\), compute Sampson or point-line distance both ways; optionally normalize by \(|w+h|\) (Yang). Convert to Isaac-style epipolar probability or hard-threshold \(d'_\mathrm{cross}\).
4. **Fuse with peer memory:** Accept / boost sticky peer assist only if epipolar score passes; reject peer-triggered soft boxes that lie far from \(\ell'\). Do **not** average the two depth readings.
5. **UI debug (optional):** Draw the peer epipolar line in the other window when a vial is locked — quick visual check that \(F\) is sane.
6. **Failure modes:** If only one cam sees the vial, keep current single-view path; if \(F\) missing, fall back to today’s sticky-peer behavior.

**Out of scope for this note:** Retraining YOLO, changing Attempt E, Mask R-CNN/SAM.
