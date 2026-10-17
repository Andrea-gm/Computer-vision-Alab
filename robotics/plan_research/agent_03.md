# Research Agent 03 — MVTrans (Multi-View Perception of Transparent Objects)

**Theme:** RGB-D / depth-aware object detection for transparent objects (clear plastic, depth holes)  
**Primary method:** **MVTrans** (ICRA 2023)  
**Repo fit:** Dual Orbbec Gemini 335L vial tracker (`depthcamera.py --dual` / `locate_pair`)

---

## Overview

MVTrans is an end-to-end **multi-view** architecture for transparent-object perception. Instead of trusting the broken RGB-D depth that depth cameras return on glass/clear plastic (holes, far readings through the body, refraction shifts), it:

1. Takes **multiple RGB views** of the same scene (2 / 3 / 5-view pretrained models exist).
2. Back-projects 2D features into a **3D matching / cost volume**.
3. Predicts a fused depth map from multi-view geometry.
4. Runs multi-task heads for **segmentation**, **3D bounding boxes**, **pose**, and refined depth.

That matches this project's core physics: the clear vial cup does not return steady depth; the opaque white cap does. MVTrans is built for exactly that failure mode—unreliable sensor depth on transparent surfaces—while exploiting **two calibrated viewpoints**, which the dual-Orbbec setup already provides.

Companion dataset: **Syn-TODD** (synthetic Toronto Transparent Object Depth Dataset)—large-scale, photo-realistic, supports RGB-D / stereo / multi-view training.

Related single-view RGB-D baselines (same problem family, weaker multi-angle fit):

| Method | Role | Speed / notes |
|--------|------|----------------|
| **ClearGrasp** (ICRA 2020) | Mask transparent pixels → normals + edges → global depth optimization | Accurate but slow (~2 s); classic “depth holes = transparent” signal |
| **TransCG / DFNet** (RA-L 2022) | End-to-end RGB + raw depth → completed depth | ~16.6 ms, 5.2 MB model; strong real-world completion |
| **TranspareNet** (CoRL 2021) | Point-cloud + depth completion using distorted depth as a clue | Heavier than DFNet |

**Recommendation for this repo:** evaluate **MVTrans 2-view** as the multi-angle detection/fusion path; keep YOLO + clear-cup depth gating as the live detector unless/until MVTrans is proven on Orbbec stills. Optionally use **DFNet** as a lightweight per-camera depth-fix prior if multi-view calibration is deferred.

---

## Multi-angle fit

- Dual Gemini 335L already delivers **two simultaneous RGB-D streams**—MVTrans ships **2-view** weights, so the view count matches out of the box.
- Current `locate_pair` + peer memory (15 frames) is a **late fusion** of independent YOLO detections. MVTrans is **early geometric fusion** (cost volume) before box prediction—stronger when one cam sees glare/refraction and the other sees a clean silhouette.
- Multi-view depth from RGB correspondence can **replace or corroborate** the hand-crafted clear-cup cues (`CLEAR_FAR_DELTA_MM`, `CLEAR_HOLE_MARGIN`) without requiring cup depth to be valid.
- 3D boxes / pose heads give a path to **cross-camera consistency** (same vial in world frame) beyond sticky peer assist.
- Syn-TODD includes clear vessels; domain gap to white-capped plastic vials is real but smaller than opaque-only detectors.

**Caveats for multi-angle:** MVTrans expects known relative camera poses (extrinsics). Dual Orbbec today runs as two independent OpenCV indexes with no stereo calibration. A one-time hand-eye / chessboard extrinsic between cam0 and cam1 is required before full MVTrans fusion; without it, fall back to per-view YOLO + optional DFNet depth fill.

---

## Pros

- Purpose-built for **transparent** objects; does not assume opaque Lambertian depth.
- Native **multi-view** design aligns with dual-Orbbec hardware.
- Joint outputs: depth + segmentation + 3D box + pose (more than YOLO alone).
- Public **2-view pretrained weights** and Syn-TODD dataset.
- Avoids treating depth holes as “missing data to inpaint blindly”—uses geometry across views instead.

## Cons

- Needs **extrinsic calibration** between the two Orbbecs; current live code has none.
- Heavier and more complex than YOLOv8s + depth gating; likely not drop-in for the existing ~real-time overlay loop without profiling.
- Syn-TODD is synthetic; vial with **opaque white screw cap** may need fine-tune or adapter labels.
- Segmentation/pose focus differs from the kept decision: **box detector only for now** (Mask R-CNN / SAM deferred).
- Full stack rewrite risk—must stay **docs / optional side experiment**, not a wipe of `runs/bon-winner` or the dataset.

---

## Citations / links

1. Wang, Y. R., Zhao, Y., Xu, H., Eppel, S., Aspuru-Guzik, A., Shkurti, F., & Garg, A. (2023). **MVTrans: Multi-View Perception of Transparent Objects.** ICRA 2023.  
   - arXiv: https://arxiv.org/abs/2302.11683  
   - HTML: https://ar5iv.labs.arxiv.org/html/2302.11683  
   - Project: https://ac-rad.github.io/MVTrans/  
   - Code: https://github.com/ac-rad/MVTrans  
   - IEEE: https://doi.org/10.1109/ICRA48891.2023.10161089  

2. Syn-TODD dataset: https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/LQKTXE  

3. Sajjan, S., et al. (2020). **ClearGrasp: 3D Shape Estimation of Transparent Objects for Manipulation.** ICRA 2020.  
   - https://arxiv.org/abs/1910.02550  
   - https://sites.google.com/view/cleargrasp  
   - https://github.com/Shreeyak/cleargrasp  
   - Google blog: https://research.google/blog/learning-to-see-transparent-objects/  

4. Fang, H., et al. (2022). **TransCG: A Large-Scale Real-World Dataset for Transparent Object Depth Completion and A Grasping Baseline.** RA-L / IROS.  
   - https://arxiv.org/abs/2202.08471  
   - Project: https://www.graspnet.net/transcg  
   - DFNet docs: https://github.com/Galaxies99/TransCG/blob/main/assets/docs/DFNet.md  

5. Xu, H., et al. (2021). **Seeing Glass: Joint Point Cloud and Depth Completion for Transparent Objects (TranspareNet).** CoRL.  
   - https://proceedings.mlr.press/v164/xu22b/xu22b.pdf  

---

## Orbbec apply sketch (docs only — no detector rewrite)

**Goal:** Improve dual-cam clear-vial boxing when one view has glare / depth holes, without abandoning Attempt E YOLO weights.

1. **Keep live path:** `yolov8s` (`runs/bon-winner`) + white-cap distance + clear-cup depth gating + `locate_pair` peer memory.
2. **Calibrate once:** Chessboard / ChArUco between Gemini 335L index 0 and 1; save `extrinsics_01.json` (R, t) + each camera’s color intrinsics from Orbbec SDK / OpenCV.
3. **Offline prototype (side script, not `depthcamera.py` rewrite):**
   - Grab synchronized RGB pair (`--dual` tick).
   - Run **MVTrans 2-view** pretrained model → transparent mask + 3D box in reference frame.
   - Project box to each camera; compare IoU vs YOLO boxes on the same stills in `vial_dataset/raw/` / review pending.
4. **Optional lightweight prior:** Run **DFNet** per camera on raw Orbbec depth to fill clear-cup holes; use completed depth only for distance / clear-vs-opaque checks (cap still preferred for range).
5. **Acceptance gates (same as plan):** conf-style score; IoU ≥ 0.5 with GT full-vial box; empty negatives → zero dets; reject mason jars / lid-only short aspect.
6. **Do not:** wipe dataset, replace bon-winner until a re-race on rebuilt `split_70_10_20` beats Attempt E on the same test protocol.

**Minimal integration idea (if later approved):** `locate_pair` soft-assist trigger could OR with “MVTrans both-views agree on transparent instance” instead of only peer YOLO memory—still append-only experiment until metrics win.

---

## Bullets for plan.md

- Method: **MVTrans** (multi-view transparent perception; ICRA 2023).
- Why: Dual Orbbec ≈ natural 2-view; ignores bad clear-plastic depth; outputs mask + depth + 3D box.
- Blocker: needs cam0↔cam1 extrinsics (not in live tracker yet).
- Fallback: TransCG **DFNet** per-cam depth completion (~17 ms) under YOLO boxes.
- Action: docs + offline eval only; keep YOLO Attempt E live.
