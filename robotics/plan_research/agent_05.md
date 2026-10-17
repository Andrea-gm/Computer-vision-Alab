# Research Agent 05 — ReST (Reconfigurable Spatial-Temporal Graph for MCMOT)

**Theme:** Multi-camera multi-object tracking (MCMOT) / tracklet fusion across views  
**Method:** **ReST** — Reconfigurable Spatial-Temporal Graph Model  
**Venue:** ICCV 2023  
**Docs only** (no code changes in this pass)

---

## Overview

ReST (Cheng et al., ICCV 2023) is an **online** Multi-Camera Multi-Object Tracking (MC-MOT) system that fuses detections across views by treating association as **graph link prediction**, not as post-hoc matching of brittle single-camera tracklets.

Core idea:

1. **Spatial Association** — At each timestamp \(t\), build a spatial graph \(G^S_t\) over detections from all cameras. Edges exist only between different cameras. Message-passing + edge classification decides which detections are the same object across views.
2. **Graph Reconfiguration** — Merge connected components (same object across views) into compact nodes by averaging appearance/geometry features; combine with the previous temporal graph \(G^T_{t-1}\).
3. **Temporal Association** — Build a temporal graph \(G^T_t\) with edges across time (view-invariant). Speed + geometry + appearance cues associate objects across frames and assign tracklet IDs.

Unlike TRACTA / DyGLIP / LMGP-style pipelines that often start from single-camera tracklets (and inherit fragmentation / ID switches), ReST associates **raw multi-view detections** first spatially, then temporally—directly addressing cross-view tracklet fusion.

**Reported results (paper):**

| Dataset | Key metrics (ReST) |
|---------|-------------------|
| Wildtrack | IDF1 **85.7**, MOTA **81.6** (SOTA online; +3.8 IDF1 / +7.0 MOTA vs prior) |
| CAMPUS Garden1 / Garden2 / Parkinglot | MOTA 77.6 / 86.0 / 77.7; MT 100, ML 0 on overlapping FoV |
| PETS-09 S2.L1 | MOTA **92.3**, MOTP **99.7** (online; competitive vs offline SOTA) |

---

## Why it fits multi-angle CV

- **Cross-view first:** Spatial graph only links detections from *different* cameras → natural multi-angle correspondence.
- **Tracklet fusion without single-cam MOT:** Avoids feeding fragmented per-camera tracklets into a second fusion stage.
- **Ground-plane geometry:** Foot-point projection \(P_c\) into a common plane + relative distance features \(\Delta p_{ij}\) give a geometric cue complementary to ReID.
- **Temporal reconfiguration:** Connected components from spatial match become nodes for temporal tracking → fused multi-view evidence carries forward in time.
- **Online:** No future frames; suitable for live dual-camera loops.
- **Post-processing constraints:** Spatial: at most one instance per camera per component; Temporal: degree limited by window \(M\). Pruning + splitting reduce bad merges.

**Pipeline (inference):**  
detections@\(t\) → Spatial Graph + MPN → prune/split → Reconfigure with \(G^T_{t-1}\) → Temporal Graph + MPN → prune/split → track IDs.

---

## Pros

- Explicit **spatial-then-temporal** fusion matches dual overlapping cameras (e.g. two Orbbec Gemini 335L).
- Does **not** depend on a strong single-camera tracker as input.
- Online / frame-by-frame → alignable with `depthcamera.py --dual` tick loop.
- Ablations show **projection geometry** matters more than ReID alone → good when vials look similar (clear cup + white cap).
- Open code (MIT): https://github.com/chengche6230/ReST
- Strong Wildtrack occlusion recovery narrative matches “peer view sees what the other misses.”

## Cons / caveats

- Paper assumes **calibrated, roughly synchronous, static** cameras with a usable ground-plane homography; Orbbec dual setup needs extrinsic/plane calibration (or a simplified 2-cam substitute).
- Designed / evaluated for **people** (OSNet ReID, foot-point projection); vials need domain adaptation (cap-centered 3D point from depth instead of foot point; drop or swap person ReID).
- Full GNN stack (DGL + OSNet + trained SG/TG) is heavier than current sticky-peer heuristic; overkill for a single vial unless multi-object is a goal.
- Spatial constraint “one object per camera per component” is fine for vials; multi-instance same class still needs care.
- Training expects multi-view labeled sequences; vial dataset is still tiny post-wipe.
- Auditorium (non-overlapping FoV) is weaker than DyGLIP in the paper—overlapping dual Orbbec is the intended fit.

---

## Orbbec dual-camera apply notes

**Map to current Jessica Camera stack (`depthcamera.py --dual`, `locate_pair`, sticky peer memory):**

| ReST concept | Orbbec / vial analogue |
|--------------|------------------------|
| Per-cam detections \(B^t_c\) | YOLO + clear-vial gate boxes from cam 0 / cam 1 |
| Appearance \(d_{v_i}\) | Small embedding on crop (or skip; paper shows geometry-heavy ablations still strong) |
| Geometry \(p_{v_i}\) | Project **white-cap** 3D point (depth from Gemini 335L) into a shared bench/floor frame instead of foot-point |
| Spatial Association | Same-frame cross-cam link: same vial ↔ peer assist (upgrade over 15-frame sticky memory) |
| Temporal Association | Per-vial ID continuity when one cam blinks / occludes |
| Graph Reconfiguration | Average dual-view pose/features into one node before next tick |
| Distances | Keep **per-camera** range display (as today); fuse IDs / presence, not necessarily average cm |

**Practical adoption ladder (docs recommendation only):**

1. **Lightweight ReST-inspired fusion (no full GNN):** same-frame bipartite match on cap 3D distance + box IoU/aspect; inherit ID from peer if one cam misses (formalize current sticky peer).
2. **Calibrate** both Gemini 335L into a common plane (checkerboard / shared bench) so \(\Delta p_{ij}\) is meaningful.
3. **Optional:** port ReST spatial/temporal graphs later if multi-vial / clutter tracking is needed; freeze YOLO, train association on synced dual recordings.

**Not recommended now:** full OSNet person-ReID pipeline on vials; end-to-end ReST training before ~100+ dual-synced labeled frames exist.

---

## Citations & links

1. **Paper (ICCV 2023 OA PDF):**  
   https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_ReST_A_Reconfigurable_Spatial-Temporal_Graph_Model_for_Multi-Camera_Multi-Object_Tracking_ICCV_2023_paper.pdf

2. **Paper HTML page:**  
   https://openaccess.thecvf.com/content/ICCV2023/html/Cheng_ReST_A_Reconfigurable_Spatial-Temporal_Graph_Model_for_Multi-Camera_Multi-Object_Tracking_ICCV_2023_paper.html

3. **Supplemental:**  
   https://openaccess.thecvf.com/content/ICCV2023/supplemental/Cheng_ReST_A_Reconfigurable_ICCV_2023_supplemental.pdf

4. **Code:**  
   https://github.com/chengche6230/ReST

5. **arXiv-style mirror / discussion:**  
   https://www.alphaxiv.org/abs/2308.13229

**BibTeX:**
```bibtex
@InProceedings{Cheng_2023_ICCV,
  author    = {Cheng, Cheng-Che and Qiu, Min-Xuan and Chiang, Chen-Kuo and Lai, Shang-Hong},
  title     = {ReST: A Reconfigurable Spatial-Temporal Graph Model for Multi-Camera Multi-Object Tracking},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2023},
  pages     = {10051-10060}
}
```

**Related MCMOT baselines cited by ReST (context):**

- TRACTA (tracklet-to-target assignment): https://github.com/GehenHe/TRACTA — He et al., TIP 2020  
- DyGLIP (dynamic graph link prediction): Quach et al., CVPR 2021  
- LMGP (lifted multicut + geometry): Nguyen et al., CVPR 2022  
- Wildtrack dataset: Chavdarova et al., CVPR 2018  

---

## Verdict for this repo

**Best research pick for Agent 05 theme:** ReST is a strong, citable MCMOT / tracklet-fusion method with open code, online inference, and a spatial-first design that maps cleanly onto dual Orbbec same-frame boxing. Prefer a **geometry-first, ReST-inspired** peer fusion layer over dropping the full person-tracking stack onto vials until calibration + dual labeled data exist.
