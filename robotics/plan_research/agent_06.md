# Research Agent 06 — TransReID (Transformer-based Object Re-Identification)

**Theme:** Cross-camera person/object re-identification (ReID) adapted to vials — appearance embeddings to match the same vial across angles.

**Primary method:** **TransReID** (He et al., ICCV 2021) — pure Vision Transformer ReID with Jigsaw Patch Module (JPM) and Side Information Embeddings (SIE) for camera/viewpoint invariance.

---

## Overview

Object ReID learns a compact appearance embedding so that the same instance (person, vehicle, or — by adaptation — vial) is close in feature space across cameras, while different instances are far apart. TransReID is the first pure-transformer object ReID framework: an image crop is tokenized into overlapping patches, a `[cls]` token is used as the global embedding, and training uses ID (cross-entropy) + triplet losses with BNNeck.

Two ReID-specific modules matter for multi-angle vial matching:

1. **Jigsaw Patch Module (JPM)** — shift + shuffle of patch tokens before regrouping into local branches, encouraging perturbation-invariant features that still cover diverse parts (lid, cup edge, reflections) rather than one brittle local cue.
2. **Side Information Embeddings (SIE)** — learnable embeddings for camera ID and/or viewpoint, added to the token sequence so the backbone can factor out camera-style bias (illumination, white-balance, slight FOV differences between Orbbec units).

Reported SOTA-era results include ~88.2% mAP / 95.0% Rank-1 on Market-1501 and ~80.6% mAP / 96.9% Rank-1 on VeRi-776 (vehicle multi-camera), showing the same recipe works for non-person rigid objects — closer to vials than pure person-ReID CNNs.

---

## Multi-angle / cross-camera behavior

| Mechanism | Role for dual-angle vials |
|-----------|---------------------------|
| Global self-attention (ViT) | Relates lid, cup silhouette, and specular cues even when one camera sees more top face and the other more side profile |
| Overlapping patches | Preserves local neighborhood structure around patch boundaries (important for thin lid rim / cup edges) |
| JPM rearrange | Trains features robust to partial occlusion and crop jitter from imperfect YOLO boxes |
| SIE camera/view IDs | Explicitly models Gemini 335L #0 vs #1 style gap; optional discrete “front / side / top” view labels if collected |
| Cosine / L2 embedding match | At inference, crop → embed → compare; no geometric calibration required for identity association |

**Pipeline sketch for this repo:** YOLO (Attempt E) detects vial boxes on each Orbbec → crop (+ optional lid-focused crop) → TransReID embed → if cosine similarity ≥ τ, treat as same vial for peer assist / sticky memory; else keep independent tracks. Distances remain per-camera depth on the white cap.

---

## Pros

- Designed for **object** ReID (person + vehicle), not person-only — better conceptual fit for vials than person-centric stripes (PCB/MGN).
- **SIE** directly targets dual-camera style shift (two Gemini 335L on different USB ports / angles).
- **JPM** improves robustness to occlusions and misaligned boxes — relevant when live boxes are lid-heavy or soft-retried.
- Global receptive field without CNN downsampling loss of fine detail (cap texture, label stickers if any).
- Official PyTorch code; inference can use **global `[cls]` only** (paper notes near-parity vs full local features) for lower latency.
- Complements existing dual sticky-peer logic: ReID similarity can replace or gate peer assist instead of “any box on peer.”

## Cons

- ViT-Base is heavier than OSNet / YOLO-n; dual live streams may need GPU or a distilled/smaller variant (`vit_small`, DeiT-small).
- Same-class clear vials with identical white caps are **near-duplicates** — classic ReID assumes identity-discriminative appearance; may need auxiliary cues (depth hole pattern, relative 3D position, temporal tracklets).
- Needs identity-linked multi-view crops for training; current recovered dataset (~18 pairs) is far too small — must capture same vial under cam0/cam1 with shared IDs.
- Person/vehicle pretrained weights will not transfer perfectly to transparent plastic; expect domain-specific fine-tune or metric-learning from scratch on vial pairs.
- Open-set “is this the same vial?” needs a calibrated threshold; false merges across look-alike empties are a risk.

---

## Orbbec apply notes

1. **Keep YOLO for detect** (`runs/bon-winner` / Attempt E). ReID is association, not detection.
2. **Crop policy:** full vial box (lid through cup) resized to TransReID input (commonly 256×128 for persons; for vials prefer nearer-square e.g. 256×256 or 384×128 after validating aspect). Optionally a second crop on the white cap only for a fused embedding.
3. **SIE labels:** `camera_id ∈ {0,1}`; optional `view_id` from coarse pose bins if dual mounts are fixed.
4. **Dual tick:** after `locate_pair`, embed both crops; if sim ≥ τ, strengthen sticky peer memory / allow soft-retry only for matched peers; if sim low, do not copy the peer’s box identity.
5. **Depth stays per-camera** on the white cap — ReID does not average distance.
6. **Data collection:** for each physical vial (and empties), record synchronized or near-sync stills from both indexes with one identity ID; include hard negatives (mason jars, desk clutter) so embeddings reject non-vials even if YOLO misfires.
7. **Latency fallback:** if ViT is too slow, use TransReID global-only inference, or fall back to a lightweight OSNet embedding with the same dual-match API — keep the association interface identical.

---

## Citations / links

1. He, S., Luo, H., Wang, P., Wang, F., Li, H., Jiang, W. **TransReID: Transformer-based Object Re-Identification.** ICCV 2021.  
   - PDF: https://openaccess.thecvf.com/content/ICCV2021/papers/He_TransReID_Transformer-Based_Object_Re-Identification_ICCV_2021_paper.pdf  
   - arXiv: https://arxiv.org/abs/2102.04378  
   - Code: https://github.com/damo-cv/TransReID  

2. Zhou, K., Yang, Y., Cavallaro, A., Xiang, T. **Omni-Scale Feature Learning for Person Re-Identification (OSNet).** ICCV 2019. Lightweight CNN ReID alternative if TransReID is too heavy.  
   - PDF: https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Omni-Scale_Feature_Learning_for_Person_Re-Identification_ICCV_2019_paper.pdf  
   - Code: https://github.com/KaiyangZhou/deep-person-reid  

3. Specker, A., et al. **ReidTrack: Reid-Only Multi-Target Multi-Camera Tracking.** CVPRW 2023 (AI City). Clustering appearance embeddings within/across cameras with YOLOv8 — close systems pattern for dual Orbbec.  
   - PDF: https://openaccess.thecvf.com/content/CVPR2023W/AICity/papers/Specker_ReidTrack_Reid-Only_Multi-Target_Multi-Camera_Tracking_CVPRW_2023_paper.pdf  

4. Related camera-aware / view-decoupled ReID (optional follow-ons):  
   - CA-Jaccard (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_CA-Jaccard_Camera-aware_Jaccard_Distance_for_Person_Re-identification_CVPR_2024_paper.pdf  
   - View-decoupled Transformer / VDT (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_View-decoupled_Transformer_for_Person_Re-identification_under_Aerial-ground_Camera_Network_CVPR_2024_paper.pdf  

---

## Recommendation for Jessica Camera

Prefer **TransReID embeddings on YOLO crops** as the cross-camera association layer for `--dual` / `locate_pair`, with SIE camera IDs `{0,1}`. Do not replace Attempt E detection. Revisit only after multi-view identity-labeled Orbbec crops exist; until then sticky peer memory remains the pragmatic association heuristic.
