# Research Agent 01 — Weighted Boxes Fusion (WBF) late fusion / detection voting

**Theme:** Multi-view late fusion / detection voting across cameras (independent detectors, fuse boxes/scores).

**Primary method:** Weighted Boxes Fusion (WBF) with cross-view 3D association

## High-level overview

1. Run the **same vial YOLO detector independently** on each Orbbec RGB frame (late fusion: no shared backbone, no early image stitch).
2. For each detection, lift the box to a rough 3D point using the camera’s depth (prefer white-cap ROI, matching current plan gating).
3. **Associate** detections across the two views by 3D proximity (and optionally epipolar consistency if extrinsics are known). Unmatched views stay as single-camera hypotheses.
4. For each associated cluster, fuse boxes/scores with **Weighted Boxes Fusion (WBF)** (or a lighter majority/weighted vote): confidence-weighted average of overlapping proposals instead of NMS winner-take-all. Dual agreement **boosts** score; single-view weak boxes stay low or drop.
5. Keep per-camera overlays and distances; use the fused identity/score to decide “vial present” and to drive peer assist (same spirit as current `locate_pair` sticky memory, but with explicit score fusion).

WBF was designed for ensembling *different models on the same image*; for dual cameras the same fusion math applies after detections are mapped into a **shared space** (normalized same-view coords if views are aligned, or 3D/world after depth lift for angled dual Gemini 335L).

## Why it helps multi-angle vial ID

- Clear plastic + specular glare means one angle often misses or boxes the wrong shiny object; the other view may still see the white cap cleanly.
- Independent detectors + vote: a false positive that appears in only one view is down-weighted; a true vial seen in both gets a higher fused confidence.
- Fits the existing stack: already share one YOLO weights load across both cams; fusion is a post-`predict` step, not a new network.
- Complements `locate_pair` peer memory: peer assist today is temporal stickiness; WBF-style voting is same-tick score consensus.

## Pros / cons for clear vial + depth

| Pros | Cons |
|------|------|
| Modular; keep Attempt E weights as-is | Pure 2D IoU WBF **across** angled views is wrong — need depth→3D (or calib) association first |
| Late fusion survives one camera drop / USB glitch | Depth holes through clear cup make 3D lift noisy unless capped on white lid |
| Better recall under glare/occlusion than single view | Needs loose frame sync; large timestamp skew → wrong pairs |
| `ensemble-boxes` / voting libraries are battle-tested | Extra association heuristics; risk of dual-view FP if both cams see same distractor (mason jar) |

## Key citations

1. **Solovyev et al., Weighted Boxes Fusion** — Image and Vision Computing, 2021. [arXiv:1910.13302](https://arxiv.org/abs/1910.13302) · [HTML](https://arxiv.org/html/1910.13302v3) · [GitHub `ZFTurbo/Weighted-Boxes-Fusion`](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
2. **Late fusion multi-camera YOLO + association** — Farooq et al., “Multi-view Multi-camera Object Detection and Tracking: A YOLOv7 and DeepSORT-Based Approach” (independent YOLO per camera, then associate). [Journal page](https://www.propulsiontechjournal.com/index.php/journal/article/view/5357) · [PDF](https://www.propulsiontechjournal.com/index.php/journal/article/download/5357/3660/9274)
3. **Ultralytics — late vs early sensor fusion** (late fusion = merge boxes/scores; modular when one sensor fails). [Glossary](https://www.ultralytics.com/glossary/sensor-fusion)
4. **Epipolar-weighted multi-view detection merge** (confidence from epipolar consistency, then NMS merge) — Isaac & Breckon, 2020. [PDF](https://breckon.org/toby/publications/papers/isaac20multiview.pdf)
5. **Practical YOLO label ensemble / voting toolkit** — [abhiksark/detection-fusion](https://github.com/abhiksark/detection-fusion) (`weighted_vote`, `majority_vote`, affirmative NMS)

## Rough apply to dual Orbbec (Gemini 335L) stack

```text
cam0 RGB-D ──► YOLO (shared weights) ──► boxes0, scores0 ──► lid-depth → P0
cam1 RGB-D ──► YOLO (shared weights) ──► boxes1, scores1 ──► lid-depth → P1
                                              │
                    match if ||T01·P0 − P1|| < τ (or sticky peer ID)
                                              │
                         WBF / weighted vote on matched cluster
                                              │
              fused score → gate clear-vial + locate_pair / annotate
```

Practical knobs:

- **Association first:** use white-cap depth centroid in each cam; optional hand-eye / stereo extrinsic `T01` if available; else peer sticky ID as soft association (as today).
- **WBF params:** `iou_thr≈0.55` only after projecting peer box into the *same* image plane (or run WBF in normalized UV within one view after warping); for pure 3D vote, prefer distance clustering + `conf_fused = w·(s0+s1)` / majority “both saw vial”.
- **Clear-vial gate after fusion:** still require white lid + clear-cup depth holes on at least one view; do not let dual agreement alone pass a mason jar.
- **Distances:** keep per-camera (plan decision); fuse identity/score only, not range average.
- **Deps:** `pip install ensemble-boxes` for classic WBF when fusing same-plane proposals; otherwise a 20-line weighted vote on matched pairs is enough for two cams.

## Decision note for plan.md

Strong default for dual Gemini: **independent YOLO + 3D associate + WBF/weighted vote**. Lowest integration cost with current `depthcamera.py --dual` / `locate_pair`; upgrades peer assist from temporal stickiness to explicit multi-view score consensus.
