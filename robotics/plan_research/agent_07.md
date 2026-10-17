# Research Agent 07 — MVAug (Two-Level Multi-View Augmentation)

**Theme:** Domain generalization / multi-view data augmentation for detectors (train so one weak viewpoint still works).

**Primary method:** **MVAug** — *Two-Level Data Augmentation for Calibrated Multi-View Detection* (Engilberge et al., WACV 2023).

**Related DG slice (YOLO-friendly):** **DG-BEV Dynamic Perspective Augmentation** — homography-based random camera-pose warps (Wang et al., CVPR 2023).

---

## Overview

MVAug solves a core multi-camera training failure mode: geometric augmentations (crop, flip, affine, perspective) normally **break cross-view calibration**, so multi-view detectors skip them and overfit scarce labeled views. The method applies two complementary homography layers:

1. **View augmentation** — each camera image gets its own geometric transform \(H_v\); the ground-plane projection is updated as \(T'_v = H_v^{-1} T_v\) so views stay aligned after warp.
2. **Scene augmentation** — a shared transform \(H_S\) is applied on the common ground plane: \(T'_v = T_v H_S\) (or combined \(T'_v = H_v^{-1} T_v H_S\)).

Training also keeps a **per-view detection head** (image-plane loss) next to the fused scene head. That auxiliary view loss + view aug explicitly regularizes the backbone so **each camera alone** stays useful — i.e., a weak / occluded / unusual viewpoint still yields usable features when the peer view is missing or hard.

On WILDTRACK and MultiviewX, view+scene aug with a simple ResNet→ground-plane→scene detector beats prior multi-view baselines by a large margin. Code: [cvlab-epfl/MVAug](https://github.com/cvlab-epfl/MVAug).

---

## Multi-angle / weak-viewpoint behavior

| Mechanism | What it does for angles |
|-----------|-------------------------|
| Independent \(H_v\) per view | Simulates different crops, tilts, and perspectives **per camera** without discarding labels. |
| Shared \(H_S\) scene aug | Moves / rotates / scales the **joint** scene so the fusion path does not memorize one bench layout. |
| Per-view detector loss \(L_\text{image}\) | Forces features to remain discriminative **from a single viewpoint**, not only after fusion. |
| Alignment-preserving \(T'\) | Avoids the “augment then reverse-project” double-resample noise used by some earlier multi-view augs. |

**DG-BEV Dynamic Perspective Augmentation (complement):** when full multi-view ground-plane fusion is not used (e.g. Ultralytics YOLO), randomly perturb yaw/pitch/roll, recover a plane homography from ≥4 correspondences, and warp RGB + boxes. That diversifies extrinsics so a detector trained mostly from one Orbbec pose still works from the other.

**GeoShift (optional adaptation step):** if one Gemini remains systematically weak, learn a small set of homographies + Mean-Teacher on **unlabeled** frames from that camera (Vidit et al., CVPR 2023) — closer to unsupervised geometric DA than pure DG.

---

## Pros / cons

### Pros
- Directly targets **multi-view scarcity** and alignment-safe geometric diversity — the dual-Orbbec setting.
- View-level + scene-level split attacks overfitting in **backbone** vs **fusion** separately.
- Per-view loss is a natural fit for “one weak viewpoint still works.”
- Homography formulation covers flip / crop / affine / perspective with one update rule for \(T\).
- Open code and clear WACV results on standard multi-view benchmarks.
- Perspective-aug idea transfers to **single-stream YOLO** without adopting full BEV 3D stacks.

### Cons
- Full MVAug assumes **calibrated** cameras and a meaningful **ground-plane** projection; lab bench + vials need an approximate plane / extrinsic estimate.
- Original task is multi-person occupancy on the ground plane, not axis-aligned YOLO boxes on transparent vials.
- Scene aug does not help if you only train independent YOLO streams with no shared scene representation (unless you re-interpret scene aug as shared affine jitter on paired dual labels).
- Aggressive perspective warps can distort vial aspect (repo gates ~1.0–1.65) and white-cap geometry — keep warp strength mild.
- Clear plastic + depth gating is outside the paper’s RGB person-detection assumptions.

---

## Citations

1. Engilberge, M., Shi, H., Wang, Z., & Fua, P. (2023). **Two-Level Data Augmentation for Calibrated Multi-View Detection**. *WACV 2023*, pp. 128–136.  
   - PDF: https://openaccess.thecvf.com/content/WACV2023/papers/Engilberge_Two-Level_Data_Augmentation_for_Calibrated_Multi-View_Detection_WACV_2023_paper.pdf  
   - HTML: https://openaccess.thecvf.com/content/WACV2023/html/Engilberge_Two-Level_Data_Augmentation_for_Calibrated_Multi-View_Detection_WACV_2023_paper.html  
   - arXiv: https://arxiv.org/abs/2211.11663  
   - Code: https://github.com/cvlab-epfl/MVAug  

2. Wang, S., Zhao, X., Xu, H.-M., Chen, Z., Yu, D., Chang, J., Yang, Z., & Zhao, F. (2023). **Towards Domain Generalization for Multi-View 3D Object Detection in Bird-Eye-View** (DG-BEV). *CVPR 2023*, pp. 13333–13342.  
   - HTML: https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Towards_Domain_Generalization_for_Multi-View_3D_Object_Detection_in_Bird-Eye-View_CVPR_2023_paper.html  
   - arXiv: https://arxiv.org/abs/2303.01686  

3. Vidit, V., Engilberge, M., & Salzmann, M. (2023). **Learning Transformations To Reduce the Geometric Shift in Object Detection** (GeoShift). *CVPR 2023*, pp. 17441–17450.  
   - HTML: https://openaccess.thecvf.com/content/CVPR2023/html/Vidit_Learning_Transformations_To_Reduce_the_Geometric_Shift_in_Object_Detection_CVPR_2023_paper.html  
   - arXiv: https://arxiv.org/abs/2301.05496  
   - Code: https://github.com/vidit09/geoshift  

---

## Orbbec / vial apply (docs only — no code changes)

**Context:** Dual Gemini 335L (`depthcamera.py --dual`), shared YOLO weights (`bon-winner` / Attempt E), peer soft-retry via `locate_pair`, small recovered dataset (~18 pairs → rebuild toward 100+ Orbbec stills).

### Practical recipe (aligned with current stack)

1. **Capture dual stills** — when saving R/W/I review frames, prefer **time-synced pairs** from cam0 and cam1 of the same vial pose (even if training remains single YOLO).
2. **YOLO training aug (DG-BEV-style slice):** add mild `perspective=` / custom homography warps in `train_vial_model.py` (or Ultralytics hyp) so train boxes see other camera’s foreshortening; keep HSV-heavy policy from Attempt E; avoid the “strong aug” collapse of Attempt B.
3. **MVAug view-aug idea without full MVDet:** if both cameras are labeled for the same scene, apply independent geometric augs per image but **re-derive** YOLO box corners via the same \(H_v\) (and optionally a shared \(H_S\) on a bench-plane approx). Do not break tags between the two views if you later fuse.
4. **Weak-view regularization:** train with occasional **single-view dropout** of one camera’s sample in a batch (force the model to succeed from the remaining view) — MVAug’s \(L_\text{image}\) spirit without rewriting the live dual tracker.
5. **Optional later:** if cam1 stays weak after re-capture, run a GeoShift-like Mean-Teacher pass on unlabeled cam1 frames before fine-tune — only after the raw set is large enough.
6. **Do not** replace live `locate_pair` / clear-vial depth gating with ground-plane person heatmaps; keep MVAug as a **train-time diversity** tool for the existing YOLO + Orbbec pipeline.

### Success signal
After re-capture + split rebuild: measure per-camera test accuracy (cam0-only vs cam1-only frames) and dual `locate_pair` same-frame hit rate; target closing the gap on the historically weaker viewpoint without regressing Attempt E’s clear-cap behavior.
