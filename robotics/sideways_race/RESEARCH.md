# Sideways clear plastic vial — practical RGB-D detection methods

**Target:** fallen / sideways clear plastic vial with an **opaque white screw cap**, imaged with a commodity RGB-D camera (e.g. RealSense).  
**Hard constraint:** clear plastic yields few stable RGB keypoints and unreliable / missing depth on the body. The **white cap** is the only Lambertian, high-contrast, depth-valid part of the object.

**Scope of this note:** ranked, implementable approaches (no code changes to `depthcamera`). URLs are primary references.

---

## Ranking criteria

Methods are ordered by **expected time-to-working-prototype × robustness for this exact object**, not by SOTA paper metrics:

1. Does it use the opaque white cap as the primary signal?
2. Does it avoid depending on keypoints / texture on clear plastic?
3. Can it run on a single RGB-D frame with ordinary tooling (OpenCV / YOLO / existing pretrained transparent nets)?
4. Path to a binary “sideways vial present + rough pose” decision for a race / gate.

---

## Ranked methods

### 1. Cap-first opaque anchor → body hypothesis (recommended first build)

**Idea:** Detect the **white screw cap** in RGB (and optionally refine with valid depth), then hypothesize the clear cylindrical body as a fixed-length, fixed-diameter extension along the cap’s axis (or along the major axis of the combined blob). Treat “sideways” as: cap axis nearly parallel to the support plane / image plane, not vertical.

**Why it fits:** Clear plastic has few keypoints; the cap does not. Cap returns usable depth on active stereo / structured-light sensors. Cap color and shape are stable across lighting relative to the body.

**Implementable pipeline:**

1. Color / HSV (or LAB) segment bright near-white blobs; optional circularity / ellipse fit for the cap face or short cylinder silhouette when lying on its side.
2. Or train a tiny detector with classes `white_cap` / `vial_body` (even ~200–400 labeled frames is enough for YOLOv8n-class models used in bottle/cap demos).
3. Take median depth in the cap mask (valid pixels only) → 3D point; estimate 2D orientation from cap ellipse or from a short PCA of the depth-valid cap cloud.
4. Ray-cast / back-project a cylinder of known vial length & diameter from the cap neck into the scene; score with:
   - fraction of **invalid / noisy depth** along the hypothesized body,
   - weak RGB edge response along the silhouette,
   - absence of opaque clutter at that pose.
5. Classify orientation: upright vs sideways from angle between cylinder axis and table normal (table from RANSAC on valid depth).

**Pros:** Fastest win; minimal dependence on transparent texture; works when body depth is almost empty.  
**Cons:** Needs approximate vial geometry; confusable with other white cylinders (markers, tape, other caps) without a body score.

**References:**

- Cap-centric industrial / demo detectors (RGB YOLO on bottle + cap):  
  https://github.com/farhan0715/Bottle-Bottle-Cap-Detection-System  
  https://github.com/s-esposito/CV-DetectTheBottleCap
- RGB-D YOLO → 3D association pattern (custom bottle training):  
  https://www.mdpi.com/1424-8220/24/9/2889  
  (FusionVision — YOLO on RGB-D scenes; custom training needed for bottles)
- Medicine-bottle RGB-D instance seg + PCA pose (axisymmetric bottles/caps):  
  https://iopscience.iop.org/article/10.1088/2631-8695/ae4558
- Vision cue that capped vials hide thread reflections (cap presence check):  
  https://arxiv.org/html/2506.03743

---

### 2. Classical depth-invalid (NaN / zero) blob + RGB silhouette association

**Idea:** On active depth cameras, clear plastic often appears as **missing depth** (and sometimes as wrong “see-through” depths). Use invalid-depth connected components as transparent candidates; confirm with RGB edges / GrabCut; require an adjacent **opaque white cap** with valid depth to accept a vial.

**Why it fits:** Turns the sensor failure mode into the detector. No need for keypoints on the clear wall. Sideways vials produce elongated invalid-depth blobs of known aspect ratio next to a white depth-valid cap.

**Implementable pipeline:**

1. **Disable aggressive hole-filling** in the depth pipeline for detection (filling erases the cue). See Intel guidance on holes vs. false fills:  
   https://realsenseai.com/wp-content/uploads/2019/07/Intel-RealSense-Depth-PostProcess.pdf
2. Binary mask: `depth == 0` (or confidence low) ∩ work region (table plane mask from RANSAC).
3. Morphology open/close; keep blobs with area / aspect ratio consistent with a sideways vial silhouette.
4. For each blob, crop RGB; refine silhouette (GrabCut / Canny + flood); check for a white opaque region at one end with **valid** depth.
5. Optional: silhouette matching against projected views of a simple CAD vial (Lysenkov-style) for pose.

**Pros:** No neural net required for a first detector; real-time on CPU; well-documented for Kinect/RealSense transparent objects.  
**Cons:** False positives from specular metal, black objects, occlusions, and IR shadows; Type-II “see-through” depths (sensor reports background through glass) need an extra invalidation step (e.g. mark pixels whose depth ≈ table behind a strong RGB edge).

**References:**

- Lysenkov et al., RSS 2012 — invalid depth as interest operator + silhouette pose:  
  https://www.roboticsproceedings.org/rss08/p35.pdf  
  https://www.roboticsproceedings.org/rss08/p35.html  
  Algorithm write-up: http://wg-perception.github.io/transparent_objects/algorithm.html
- RealSense RGB + depth + IR candidate classification:  
  https://iopscience.iop.org/article/10.1088/1742-6596/1183/1/012011/pdf
- Morphology + edge color similarity for transparent recognition (RealSense R200):  
  (Yun et al., “Recognition of transparent objects using 3D depth camera”, 2017 — often cited alongside the above; search title for PDF mirrors)
- Depth-missing-aware glass detection (modern DL version of the same cue):  
  https://jiaying.link/AAAI25-RGBDGlass/  
  https://dl.acm.org/doi/10.1609/aaai.v39i5.32558

---

### 3. Fine-tuned RGB detector / segmenter (lab vials) + light RGB-D gating

**Idea:** Ignore depth for primary detection. Train a modern detector (YOLOv8/YOLO11, or lab-focused DenseSSD) on your exact vial with classes such as `vial_sideways`, `vial_upright`, `white_cap`. Use depth only to reject wall/ceiling false positives and to lift the box to 3D.

**Why it fits:** Transparent chemical vessels are hard for COCO-pretrained models but become reliable after domain-specific training. DenseSSD reports >95% mAP on complex vial scenes (empty and filled). Cap class anchors the clear body when edges are weak.

**Implementable pipeline:**

1. Capture 300–1500 frames of your vial: upright, sideways, varied lighting, clutter; label boxes or masks.
2. Train YOLOv8n/s (or Ultralytics seg) with strong augmentation (blur, glare, exposure).
3. At runtime: detect → keep instances where a `white_cap` is near a `vial_*` box with plausible geometry.
4. RGB-D gate: require table-plane support depth under the box; optional invalid-depth ratio inside the body ROI above a threshold (ties to Method 2).

**Pros:** Best RGB robustness under clutter and when depth holes are messy; easy deployment; real-time.  
**Cons:** Needs a labeling pass; may still miss nearly invisible clear walls without the cap class; depth not used for geometry unless you add a second stage.

**References:**

- DenseSSD — transparent chemical vessel detection (vials, lab generalization):  
  https://www.nature.com/articles/s41524-024-01216-7
- FusionVision — YOLO + RGB-D segmentation / 3D association:  
  https://www.mdpi.com/1424-8220/24/9/2889
- Transparent segmentation priors (if you prefer masks over boxes): Trans2Seg / related work cited inside the DenseSSD paper above.

---

### 4. Transparent depth completion (TransCG / ClearGrasp) → cylinder or opaque pose stack

**Idea:** Run a pretrained **depth completion** network so the clear body gets a plausible depth surface; then run ordinary 3D primitives (cylinder RANSAC) or an opaque RGB-D pose network on the completed cloud.

**Why it fits:** Directly attacks “few keypoints + broken depth.” After completion, sideways vs upright is a trivial axis-vs-gravity test. Cap region can be fused from raw depth (trust raw depth on opaque cap; trust completed depth on body).

**Implementable pipeline:**

1. Start from **TransCG / DFNet** (real-world training, grasping baseline) or **ClearGrasp** (normals + mask + global optimization).
2. Replace / blend: `depth_out = where(cap_mask, depth_raw, depth_completed)`.
3. Segment vial (network mask or Method 1/3 box).
4. Fit cylinder or oriented bounding box; report lateral pose on the table.

**Pros:** Enables grasping / collision geometry, not just detection; pretrained weights and datasets exist.  
**Cons:** Heavier GPU dependency; ClearGrasp-style global opt can be slow; domain gap if your vial is much smaller / more specular than training objects; overkill if you only need a boolean “fallen vial.”

**References:**

- ClearGrasp paper + overview:  
  https://arxiv.org/abs/1910.02550  
  https://sites.google.com/view/cleargrasp  
  https://research.google/blog/learning-to-see-transparent-objects/  
  Code: https://github.com/Shreeyak/cleargrasp
- TransCG dataset + DFNet (recommended practical starting point):  
  https://arxiv.org/abs/2202.08471  
  https://github.com/graspnet/TransCG  
  https://graspnet.net/transcg
- ClearPose benchmark (350K+ RGB-D frames; depth completion + pose baselines including TransCG / FFB6D):  
  https://arxiv.org/abs/2203.03890  
  https://github.com/opipari/ClearPose

---

### 5. Learned transparent pose (ClearPose / FFB6D with completed depth, or KeyPose stereo keypoints)

**Idea:** Full 6DoF (or sparse 3D keypoint) estimation for known vial geometry. Use RGB-D pose networks on **completed** depth (ClearPose findings: raw depth hurts; completed depth helps), or bypass depth with **stereo keypoint** models (KeyPose) that do not rely on clear-plastic texture matching.

**Why it ranks last here:** Highest accuracy ceiling and best if you need precise pick pose, but slowest to productize (CAD, labeling, training, stereo or completion dependency). For a sideways **race / presence** check, Methods 1–3 usually suffice.

**Implementable variants:**

- **A. ClearPose-style:** instance mask → TransCG completion → FFB6D / similar RGB-D pose on known mesh.  
- **B. KeyPose-style:** if your camera exposes a stereo pair (many D400-series do), predict sparse 3D keypoints (cap center, vial tip, axis ends) from stereo RGB — explicitly designed for transparent objects where depth fails and RGB keypoints are scarce (keypoints are **learned**, not SIFT on the plastic).

**Pros:** Best pose quality; KeyPose matches the “few keypoints” problem statement directly.  
**Cons:** Data/CAD cost; KeyPose wants stereo not just a depth channel; more moving parts than a race detector needs.

**References:**

- ClearPose + pose benchmarks:  
  https://arxiv.org/abs/2203.03890  
  https://github.com/opipari/ClearPose
- KeyPose (stereo 3D keypoints for transparent objects):  
  https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_KeyPose_Multi-View_3D_Labeling_and_Keypoint_Estimation_for_Transparent_Objects_CVPR_2020_paper.html  
  https://research.google/blog/keypose-estimating-the-3d-pose-of-transparent-objects-from-stereo/  
  Code/data: https://github.com/google-research/google-research/tree/master/keypose  
  Project: https://sites.google.com/view/keypose
- RGB-D multimodal transparent detection (if you need a stronger segmenter before pose):  
  MFFNet: https://journals.sagepub.com/doi/10.1177/17298806241283373

---

## Suggested build order for this project

| Step | Method | Deliverable |
|------|--------|-------------|
| A | #1 Cap-first | Boolean + rough heading of sideways vial |
| B | #2 Depth-invalid score | Confirm clear body; reject random white disks |
| C | #3 Fine-tuned YOLO | Robustness under clutter / lighting |
| D | #4 Completion | Only if you need graspable geometry |
| E | #5 Full pose | Only if 6DoF is a hard requirement |

**Practical hybrid (often best):** Method **1 + 2** — detect white cap in RGB, require elongated invalid-depth (or completed-depth) region attached to the cap with vial aspect ratio. That hybrid directly exploits the opaque cap and the clear body’s depth failure without waiting on a large labeled set.

---

## Sensor checklist (RGB-D)

- Prefer **raw / lightly filtered depth** for Methods 2 and the hybrid; turn off hole-filling while detecting.
- Calibrate RGB↔depth extrinsics; evaluate masks in the depth frame.
- Estimate **table plane** from valid depth every frame or infrequently; “sideways” = cylinder axis ⟂ table normal within a threshold.
- Expect **glare** on the clear wall: do not require SIFT/ORB matches on the body.
- Cap can be overexposed (pure white): expose for the cap or use HDR / IR if available as an extra channel (see RealSense RGB+IR candidate papers above).

---

## Out of scope

- No modifications to `depthcamera` in this research pass.
- Polarization / multi-view light-field methods omitted (extra hardware).
- Chemical-contents classification omitted (detection / pose only).
