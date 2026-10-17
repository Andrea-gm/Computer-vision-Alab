# Research Agent 10 — MVDeTr (Shadow Transformer / MVDet-family)

**Theme:** Transformer / modern multi-view detectors  
**Primary method:** **MVDeTr** — Multiview Detection with Shadow Transformer  
**Scope:** Practical takeaways for **two** Orbbec Gemini 335L views; **not** a mandate to replace YOLO / `locate_pair`.

---

## Overview

MVDeTr (Hou & Zheng, ACM MM 2021) upgrades the earlier **MVDet** (ECCV 2020) multiview detector with a **shadow transformer**: a Deformable-DETR-style encoder that fuses **per-camera features after projection onto a shared ground plane**. Convolution on that plane (MVDet) applies the same filter everywhere; perspective warp creates location- and camera-dependent “shadow-like” stretches, so translation-invariant fusion is a poor match. The shadow transformer learns **different attention offsets per ground location and per camera**, with learned **camera embeddings** plus a shared **ground-plane position embedding**.

Detection is formulated as **ground-plane keypoint / occupancy heatmap** regression (not full DETR set prediction with object queries), plus optional **per-view** foot-point and box losses. Training adds **view-coherent data augmentation**: random affine transforms per camera, then **invert** those warps on features before ground-plane projection so geometry stays consistent across views.

**Reported results (occupancy MODA):** Wildtrack **91.5%**, MultiviewX **93.7%** (vs MVDet 88.2% / 83.9%). Code trains on a **single RTX 2080 Ti**. Official repo also ships a convolutional MVDet-like path (`--world_feat conv` vs `deform_trans`).

**Related (for context, not primary pick):**

| Lineage | Idea | Fit for Jessica dual Orbbec |
|---------|------|-----------------------------|
| **MVDet → MVDeTr** | Project features to ground plane; fuse with conv or multiview deformable attention | Strong *ideas* for calibrated dual views; full BEV stack is heavy |
| **DETR3D** | 3D object queries → project into each camera → sample 2D features | Needs calibrated multi-cam + large 3D labels; overkill for one vial |
| **PETR / StreamPETR** | Encode 3D position into multi-view 2D features; DETR decoder | Same: auto-driving scale; useful only as “query ↔ multi-cam” intuition |

---

## Multi-angle / multi-view mechanism

1. **Sync + calibrate** — Multiple overlapping, time-aligned views with known intrinsics/extrinsics.
2. **Per-view CNN** — Shared ResNet-18 (stride reduced / dilated so features stay denser); optional 128-d bottleneck.
3. **Perspective project** — Warp each view’s feature map onto a common **z = 0** bird’s-eye grid via camera matrices.
4. **Shadow transformer (multiview deformable attention)** — At each ground cell, attend to `M × K × C` sparse sample points across **all** cameras (not single-image deformable attention). Camera + position embeddings steer which “shadow” pattern to trust.
5. **Head** — Ground-plane occupancy heatmap (+ offset); per-view auxiliary losses on feet / boxes.
6. **View-coherent aug** — Augment images independently → extract features → **unwarp** → project → joint supervise.

For **C = 2** the same math applies: fewer views means weaker occlusion fill-in, but attention still learns which camera to trust where their footprints differ.

---

## Pros

- Explicit **multi-view fusion** with geometry (ground plane + calibration), not just late NMS of independent boxes.
- Transformer fusion is **position- and camera-aware** — better match to perspective distortion than plain BEV convolution.
- **View-coherent augmentation** is a transferable training trick for any dual-camera labeled set.
- Efficient relative to full DETR3D/PETR stacks; open PyTorch code (Deformable DETR ops).
- Ablations show gains from: multiview deformable attn vs conv; vs single-image deformable; +aug; +per-view loss.

## Cons / mismatches for this repo

- Built for **crowded pedestrian occupancy** on a large calibrated floor grid (Wildtrack 7 cams / MultiviewX 6 cams), not a single clear vial with a white cap.
- Needs **extrinsics + a world ground plane**; dual Gemini 335L today run as two independent OpenCV streams with peer assist — no shared BEV.
- Output is **BEV occupancy**, not per-camera YOLO xyxy boxes used by depth-on-cap measurement.
- Small recovered dataset (~18 pairs) cannot train a ResNet+transformer BEV model; Attempt E YOLO remains the right detector.
- Building CUDA deformable ops and a full rewrite would dominate effort vs improving `locate_pair` / peer memory.

---

## Practical takeaways for two Orbbec cams (no full rewrite)

Borrow **ideas**, keep **yolov8s + locate_pair**:

1. **Treat peer assist as soft “attention”** — When cam A is confident and cam B misses, prefer A’s cue (already sticky peer memory / soft-retry). Conceptually: high weight on the reliable view, like MVDeTr’s camera-aware attention — without BEV.
2. **Prefer same-timestamp fusion** — MVDeTr assumes synchronized frames; keep dual tick → `locate_pair` before either `imshow` (already decided).
3. **Geometry only if calibrated later** — If extrinsics are measured, a lightweight check (“does A’s box project near B’s depth hit?”) is the MVDet-family lesson; skip full ground-plane heatmap.
4. **View-coherent labeling / aug (optional)** — When capturing dual stills, keep time-aligned pairs; if augmenting for dual training later, apply warps then restore geometric consistency before any cross-view loss.
5. **Per-view supervision stays primary** — MVDeTr still uses monocular losses; keep training YOLO per image; dual logic stays at inference association.
6. **Do not swap in DETR3D/PETR/MVDeTr weights** for live vial tracking — wrong domain, label format, and compute budget.

---

## Orbbec apply (Jessica Camera)

| Current system | MVDeTr-style lesson |
|----------------|---------------------|
| Shared YOLO weights on cam 0 & 1 | Keep; MVDeTr also shares backbone across views |
| `locate_pair` + sticky peer (15 frames) + soft-retry | Closest existing analogue to multiview attention / occlusion fill |
| Distance from **white cap depth** per camera | Keep per-cam depth; BEV occupancy would not replace cap ranging |
| Clear-cup depth gating | Orthogonal to transformers; keep |
| Dataset recovery / re-capture | Prioritize more **dual-synced** R/W/I stills if any cross-view loss is tried later |

**Recommendation:** Document MVDeTr as the strong multi-view transformer reference; **implement zero architecture change**. Evolve dual logic toward clearer “trust the peer when local evidence is weak,” and only revisit calibrated BEV if occlusion or FP problems persist after dataset rebuild.

---

## Citations / links

1. Hou, Y. & Zheng, L. (2021). *Multiview Detection with Shadow Transformer (and View-Coherent Data Augmentation)*. ACM MM 2021.  
   - arXiv: https://arxiv.org/abs/2108.05888  
   - HTML: https://ar5iv.labs.arxiv.org/html/2108.05888  
   - ACM: https://dl.acm.org/doi/10.1145/3474085.3475310  
   - Code: https://github.com/hou-yz/MVDeTr  

2. Hou, Y., Zheng, L. & Gould, S. (2020). *Multiview Detection with Feature Perspective Transformation*. ECCV 2020 (MVDet).  
   - arXiv: https://arxiv.org/abs/2007.07247  
   - Code: https://github.com/hou-yz/MVDet  

3. Zhu, X. et al. (2021). *Deformable DETR: Deformable Transformers for End-to-End Object Detection*. ICLR 2021.  
   - https://arxiv.org/abs/2010.04159  

4. Carion, N. et al. (2020). *End-to-End Object Detection with Transformers* (DETR). ECCV 2020.  
   - https://arxiv.org/abs/2005.12872  

5. Wang, Y. et al. (2022). *DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries*. CoRL.  
   - https://arxiv.org/abs/2110.06922  

6. Liu, Y. et al. (2022). *PETR: Position Embedding Transformation for Multi-View 3D Object Detection*. ECCV 2022.  
   - https://arxiv.org/abs/2203.05625  

7. StreamPETR (ICCV 2023) — temporal multi-view DETR-family (context only):  
   - https://github.com/exiawsh/StreamPETR  
