# Research Agent 08 — MVTrans (Multi-View Perception of Transparent Objects)

**Theme:** Transparent / specular object detection beyond generic YOLO  
**Primary method:** **MVTrans** (ICRA 2023)  
**Fit to Jessica Camera:** Dual Gemini 335L already provide two calibrated-ish RGB views; clear plastic vials break Orbbec depth exactly as MVTrans assumes.

---

## Overview

Commodity RGB-D cameras (including Orbbec Gemini) assume opaque Lambertian surfaces. Clear cups/vials **refract and specularly reflect**, so depth holes and “see-through” far readings appear on the cup body — the same failure mode already used as a live gating cue in `depthcamera.py`.

**MVTrans** (Wang et al., ICRA 2023) treats that depth as unreliable and instead fuses **multiple overlapping RGB views**:

1. Shared SPP features per view  
2. **Plane-sweep** warping of support views into a 3D matching volume (photo-consistency across angles)  
3. Fuse with ResNet-50 2D context → low-res depth  
4. ResNet-FPN multi-task heads: **instance segmentation**, **full-res depth**, **3D oriented bbox + 6-DoF pose**

It is explicitly designed for **glass / plastic vessels** in lab and household scenes — closer to clear vials than generic COCO YOLO detectors.

They also release **Syn-TODD**: ~1996 photo-realistic scenes × 57 views, procedural transparent vessels + ShapeNet clutter, supporting RGB / stereo / multi-view training.

---

## Multi-angle / multi-view mechanism

| Piece | Role |
|-------|------|
| Reference + support RGB | One frame is the prediction frame; others constrain geometry |
| Homography plane-sweep | Back-project support features onto depth hypothesis planes |
| View-average pooling | Aggregate photo-consistency across N−1 supports |
| Ablation (2 / 3 / 5 views) | More views help hardest 3D tasks (bbox/pose); 2D seg gains less |

**Key empirical claim (KeyPose real data):** 2-view MVTrans already beats stereo KeyPose / SimNet on pose (AUC 92.7 vs 90.0 / 87.9); 5-view edges further (AUC 92.9). On cluttered Syn-TODD, depth RMSE drops from SimNet 1.229 → MVTrans ~0.12–0.13.

This maps cleanly to **two fixed Orbbec views** (`--dual` / `locate_pair`): treat cam0 as reference, cam1 as support (or vice versa), without needing eye-in-hand motion.

---

## Pros

- **Purpose-built for transparency** — not a generic detector fine-tuned on COCO.  
- **Does not trust raw RGB-D depth** on glass/plastic (matches vial depth-hole behavior).  
- **Native multi-view** — dual Gemini setup is a natural 2-view input, not a bolted-on peer hack.  
- **Multi-task outputs** — seg mask + recovered depth + 3D box/pose in one forward pass.  
- **Public code + weights** (2/3/5-view Syn-TODD checkpoints) and large synthetic pretrain set.  
- Outperforms ClearGrasp-style RGB-D pose pipelines when sensor depth is wrong.

## Cons

- **Heavy** vs YOLOv8s live tracker (matching volume + 3D CNN + FPN heads; paper trains on 4× A100).  
- Needs **known relative extrinsics** between views (plane-sweep uses K, R, t). Dual Orbbecs need a one-time stereo/extrinsic calibration, not just OpenCV indexes 0/1.  
- Syn-TODD / KeyPose classes ≠ “white-cap clear vial”; **domain transfer / fine-tune** still required.  
- Oriented 3D box + pose is more than current product needs (2D `vial` box + cap distance).  
- Sim-to-real and cluttered novel objects remain hard (paper notes Syn-TODD 3D mAP still modest).  
- Integration cost ≫ keeping Attempt E YOLO + clear-cup depth gate + sticky peer memory.

---

## Orbbec / Jessica apply notes

**Best near-term use (docs only recommendation):** borrow the *idea*, not a full MVTrans rewrite.

1. **Keep YOLO for 2D box** (Attempt E / `bon-winner`) as the fast detector.  
2. **Treat dual RGB as multi-view evidence** (already partly done via `locate_pair` peer assist): require consistent detection across cams, or lift 2D boxes with known extrinsics for a soft 3D check.  
3. **Keep using broken depth as a transparency cue** (depth holes through cup + solid white lid) — aligns with MVTrans’s “don’t trust raw depth on glass” premise; optionally explore **learned depth completion** (ClearGrasp-family) later only if cap ranging fails.  
4. If pursuing MVTrans-like fusion: calibrate Gemini pair once, run **2-view** checkpoint as research prototype offline; do not replace live YOLO until latency and vial-class fine-tune are proven.  
5. Dataset rebuild (~100+ Orbbec stills) should prefer **paired dual-cam frames** with shared timestamps so multi-view methods can be trained later.

**Not recommended now:** replace Attempt E with full MVTrans in `depthcamera.py` (latency, calibration, labeling of 3D pose/seg).

---

## Related literature (context, not the chosen method)

| Method | Angle | Note |
|--------|-------|------|
| TransLab / Trans10K | Single RGB boundary-aware seg | Strong 2D glass/plastic masks; no multi-view 3D |
| ClearGrasp | RGB-D depth completion | Fix depth after mask/normals; single view |
| KeyPose / SimNet | Stereo RGB pose | Direct ancestors; MVTrans generalizes to N views |
| GraspNeRF | Sparse multi-view NeRF grasp | Transparent/specular grasping; heavier than needed |
| TORM | Multi-view masks → DMTet reconstruction | Strong grasp demo; optimization-heavy |
| Survey: Jiang et al. “Robotic Perception of Transparent Objects” | Review | Multi-modal (RGB-D, polar, thermal) roadmap |

---

## Citations & links

1. **MVTrans paper (arXiv):** https://arxiv.org/abs/2302.11683  
2. **HTML (ar5iv):** https://ar5iv.labs.arxiv.org/html/2302.11683  
3. **Project page:** https://ac-rad.github.io/MVTrans/  
4. **Code:** https://github.com/ac-rad/MVTrans  
5. **Syn-TODD dataset:** https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/LQKTXE  
6. KeyPose (stereo transparent keypoints): https://arxiv.org/abs/1912.02805  
7. ClearGrasp: https://arxiv.org/abs/1910.02550  
8. TransLab / Trans10K: https://arxiv.org/abs/2003.13948 · https://xieenze.github.io/projects/TransLAB/TransLAB.html  
9. Robotic Perception of Transparent Objects (review): https://arxiv.org/abs/2304.00157 · https://sites.google.com/view/transperception  
10. Confusing Object Detection survey (glass/mirror/camouflage): https://www.techscience.com/cmc/v80n3/57919/html  

```bibtex
@inproceedings{wang2023mvtrans,
  title={MVTrans: Multi-View Perception of Transparent Objects},
  author={Wang, Yi Ru and Zhao, Yuchi and Xu, Haoping and Eppel, Sagi and Aspuru-Guzik, Alan and Shkurti, Florian and Garg, Animesh},
  booktitle={ICRA},
  year={2023}
}
```

---

## Verdict for plan.md race

| Criterion | Score |
|-----------|-------|
| Transparent-object specificity | Excellent |
| Multi-angle / dual-cam fit | Excellent (2-view native) |
| Drop-in replace for live YOLO | Poor (cost/latency/calib) |
| Ideas to steal now | High (extrinsics + cross-view consistency; distrust cup depth) |

**Recommendation:** Document MVTrans as the strongest **multi-view transparent perception** baseline for this theme; keep YOLO live; optionally prototype offline 2-view fusion after dual-cam extrinsic calibration and paired-frame recapture.
