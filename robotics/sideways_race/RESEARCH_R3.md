# Round 3 — `J_research` notes

## Web / literature skim (classical, cap + transparent body)

| Idea | Why it matters here |
|------|---------------------|
| **Cap-first localization** (HSV/LAB white blob → contour/ellipse) | Only Lambertian, depth-valid part of a clear vial; body is optional in RGB. |
| **Invalid-depth blob + silhouette** (Lysenkov RSS’12; RealSense hole-fill guidance) | Active RGB-D often returns zeros through clear plastic; bridge holes **horizontally** only. |
| **Single opaque end** (beverage QC: one lid per cylinder) | Rejects wall slabs, double-bright clutter, upright lids. |
| **Low-angle / dark-field cap imaging** (PLOS cap-defect paper) | Motivates treating the cap as a compact ellipse when side-on, not a long horizontal white run. |
| **Depth vs. cap plane** | Cap median Z anchors body columns; far-plane stripes through plastic must not dominate. |

No training, no global orientation CNN, no keypoint detector as the primary signal.

## `J_research` method (distinct from B / B2 / F)

1. **Ellipse-validated cap seeds** — external white contours, `fitEllipse` eccentricity + fill, side-on aspect gate.
2. **Horizontal depth-hole reconstruction** — morph-close `(7×1)` on invalid depth; intersect with RGB body cue (low sat / horizontal rim Sobel); keep the connected component that contains the cap.
3. **Tight AABB** — union cap + body columns but **clamp vertical extent to the cap band** (fixes common overshoot vs. GT).
4. **Intrinsics length gate** — optional mm-scale check from cap Z (`vial_3d_dims.json`).
5. **Orphan-hole fallback** — if no cap seed fired, accept elongated hole blobs with **exactly one** white+depth cap run (same safety net as B2, without B’s PCA path).

Primary path is cap→attached hole component; fallback is hole→single-cap only.

## Implemented pipeline (`J_research`)

1. **Core (recall anchor):** `B_rotated_white_cap` + `B2` depth-hole fallback (same merge as live winner).
2. **Novel oriented path:** ellipse-validated compact cap → intersect attached horizontal depth-hole CC with RGB body cue → `minAreaRect` AABB (handles angled hard positives better than axis-aligned CC alone).
3. **Gated supplements:** oriented proposals plus race methods A (horizontal cap grow) and D (edge template) are accepted only when they do **not** overlap core (IoU < 0.38), show clear-cup hole fraction ≥ 0.35, and pass a single white-end score (stricter for D).

No training, no SIFT/ORB mainline, no edits to `depthcamera.py` / ROS.

## Eval (Round-3 synth manifest, 76 frames — 20261009)

| Method | Recall@0.5 | FP | Mean IoU |
|--------|------------|-----|----------|
| **J_research** | **0.925** | **0** | **0.754** |
| B2_improved | 0.725 | 0 | 0.632 |
| G_ensemble | 0.900 | 0 | 0.701 |

Run: `python sideways_race/eval_sideways.py --only method_j_research method_b2_improved`
