# Vial detection plan ? decisions

## What counts as the vial

- One class: **clear plastic vial** with an **opaque white screw cap**.
- The box must cover **lid top through clear cup bottom** (not lid-only).
- Not: mason jars, windows, walls, logos, hands, desk clutter, or blank empty benches.
- Distance is measured on the **white cap** (the clear cup does not return steady depth).
- Live gating treats ?clear? as: solid white lid **and** depth holes / far readings through the cup body (opaque jars fail that check).

## Model that is kept

- Best-of-N race on a fixed **70% train / 20% test / 10% validate** split (seed **42**).
- Path: `vial_dataset/split_70_10_20/` (rebuild with `make_split_70_10_20.py` from `raw/`). Wrong-ratio `split_702010/` removed.
- **Test accuracy:** conf 0.25 (strong cam); positive ? IoU ? 0.5 with GT box; empty negative ? zero detections.
- **Live weights (Round 3 train-race winner):** `runs/vial-live/weights/best.pt` (= `runs/train-race/r3-winner/weights/best.pt`; R3 polish from R1). Archives: `r1-winner`, `r2-winner`, `r3-winner`. Fine-tune base lineage: `runs/bon-winner/weights/best.pt` (Attempt E, yolov8s). `find_weights` prefers vial-live ? r1-winner ? bon-winner.
- Live ID uses **BoN #3** asymmetric gates on top (below); independent dual keeps local soft-clear but **no peer soft-locate inventing boxes**.
- Segmentation (Mask R-CNN / SAM): **not now**. Prefer more Orbbec photos + R/W/I review; revisit later only if the box is on the right object but still spills.

## Train-race Round 1 winner

- **Winner:** `train-r1-07-cf168274` ? *MVAug-inspired mild perspective + freeze/unfreeze* (phase1 `best.pt`).
- **Holdout (seed 42, 40/6/11 from raw; conf 0.25, IoU?0.5):** test **100%** (11/11), val **100%** (6/6), **recall 1.0 / FN 0**, empty FP **0**.
- **Rank key (user priority):** fewest FN / highest recall ? accuracy ? low empty FP. #7 beat #5 (also FN=0) on empty FP (0 vs 1) and accuracy (100% vs 90.9%).
- **Weights on MAIN:** `runs/train-race/r1-winner/weights/best.pt` (live later promoted to R3 ? see Round 3).
- **Full table:** `TRAIN_RACE_R1_RESULT.md`.

## BoN winner (vial vs not-a-vial) ? #3 Asymmetric per-camera thresholds

**Winner:** strategy **#3** ? *Asymmetric per-camera thresholds* (`bon-vid-03-ba9c8430`). Applied into main `depthcamera.py` + `camera_gates.json`.

### Algorithm overview (vial vs not-a-vial)

1. **Propose** with shared YOLO weights (both cams load one detector).
2. **Accept only clear plastic + white screw cap:** solid white lid band, interior structure/edges, and clear-cup depth holes (invalid/far under the cup; opaque jars and flat walls fail).
3. **Asymmetric gates:** camera **0** uses a **strong** profile (tighter conf/holes/aspect); camera **1** (default weak) uses a **permissive** profile so the failing dual view can still ID the same vial without accepting walls/mason jars.
4. **Live defaults** (see `camera_gates.json`): strong YOLO conf **0.15** (recall-oriented; race eval still uses 0.25), weak **0.05**; strong/weak both loosen structure floors vs the old 0.25/tight band so clear vials in view are not dropped.
5. Overlay label stays **`vial`**. Distances stay **per-camera** (never averaged). No depth thumbnail.

### 70/20/10 eval notes (seed 42)

- Split rebuilt from current `raw/` (**no wipe**): **20 train / 3 validate / 6 test** (29 labeled; 26 pos / 3 empty).
- Weights: `runs/bon-winner/weights/best.pt`. Race score IoU ? 0.5.
- Baseline conf 0.25: **83.3%** test (5/6). Weak profile conf 0.05 (primary for #3): **100%** test (6/6); recovered `vial_0005.jpg`. Empty negatives: **0 FP** at 0.05?0.35.
- Among 10 strategies, #3 tied #6 at 100% test; won on **simpler code** (no dual-weight ensemble) while keeping FP at 0 and `locate_pair`.

### Dual-cam / `locate_pair` (classic coupled path)

- **`locate_pair` same-time boxing remains available** via classic `--dual`: each dual tick reads both frames, runs strict clear-vial on both (with that cam?s `CameraGateProfile`), updates sticky peer memory, soft-retries missers, and returns **both** fixes before either window annotates/`imshow`.
- CLI: `--weak-camera` (default 1), `--cam0-conf`, `--cam1-conf`; base `--conf` sets the strong cam.

## Independent Dual Processes (preferred)

- **Goal:** each Orbbec runs like its **own program** ? find the clear vial + white cap from that camera?s RGB-D alone (YOLO + clear-vial gates + BoN #3 asymmetric per-cam thresholds). Sibling camera is **not** required for a box.
- **How to run:** `python depthcamera.py --dual-independent` (or `python run_dual_independent.py`). Same as `--dual --independent`. Single cams stay: `python depthcamera.py --camera 0` / `--camera 1`.
- **IPC (localhost UDP, base port 47800 + camera index):** each process publishes `{timestamp, camera_id, has_vial, box, distance_cm, conf}` and reads the peer. If both have a vial with distances, overlay shows **agree** / **disagree** within absolute cm **or** relative threshold (defaults 3 cm / 15%). **Never** average distances or replace the local cm with the peer?s.
- **If peer is missing / stale / no vial:** local cam still boxes and shows its own distance normally (`no peer` / `peer no vial` in the tip).
- **No soft peer assist in this mode:** no sticky peer that invents a box on the misser; no `locate_pair` soft-retry. Classic `--dual` keeps the old coupled path for comparison.
- **Process isolation:** each worker only `destroyWindow`s its own title on exit (no `destroyAllWindows` in the single-cam/`close()` path), so quitting one process does not kill the other?s OpenCV window.

## False not-in-view fix (2026-10-02)

**Winner:** Agent **A** ? local **`soft_clear`** on `locate_standalone` / independent dual (white lid + capped look + interior structure when clear-cup depth holes are weak). Strong YOLO default **0.18**. Walls/empty still rejected.

**Also kept on main (compatible):** **B** cap-first `detect_vial_by_cap` fallback (no peer invent); **C** live `camera_gates.json` weak conf **0.05** + honest overlay (`Low confidence` vs `No vial in view`). Weights still prefer `runs/vial-live` then `runs/bon-winner`. Notes: `FN_FIX_A.md` / `B` / `C`.

## Dual Orbbec cameras

- **Right now:** **yes ? a second camera is detected** (two Gemini 335L). OpenCV indexes: **0** and **1**. Index **2** is not a second camera.
- **How to run both (preferred):** `python depthcamera.py --dual-independent` or `python run_dual_independent.py`.
- **Classic coupled dual:** `python depthcamera.py --dual` or `python depthcamera.py --camera 0 --camera2 1` (not `--camera 2`) ? still uses `locate_pair` sticky soft peer assist.
- Check what is plugged in: `python depthcamera.py --list-cameras`
- Code warms each Orbbec until color+depth arrive, opens a window for each camera immediately, and prints a clear hint if only one opens.
- **Dual same-time boxing (`locate_pair`, classic `--dual` only):** each dual tick reads both frames, then `locate_pair` runs strict clear-vial on both (**with that cam?s asymmetric `CameraGateProfile` ? BoN #3**), updates sticky peer memory (**15** frames), soft-retries missers, and returns **both** fixes **before** either window annotates/`imshow`. Prefer **`--dual-independent`** when coupling hurts reliability. Distances stay per-camera (no averaging). Overlay still shows **`vial`**. No depth thumbnail.
- **Cleanup:** do **not** mass-delete worktrees (prior cleanup wiped `vial_dataset`). Keep race worktrees; apply winner into main only.
- Share **one** YOLO weights load per process; per-cam conf/gates differ (#3). Each Orbbec needs its own USB 3 port.
- If `--camera2` fails to open, camera 1 keeps running and the console explains which indexes were detected.

## Dataset recovery status (2026-10-02)

- **Loss:** Cleanup deleted Cursor worktrees that junctioned into `vial_dataset/`, wiping `raw/`, `review/` contents, `split_70_10_20/`, and `images/`/`labels/`. Dataset paths were **gitignored**, so git has no copies. Shadow copies / Previous Versions were not accessible without admin. Sticky-peer worktree has code only (no images).
- **Before wipe:** ~**119** labeled pairs (?88 `placed_*` composites + ?26 real Orbbec/`named` positives + 5 empty negatives; `vial_0013` quarantined). Race split was 83 / 24 / 12.
- **Recovered into `raw/` now:** **18** jpg+txt pairs:
  - **11** from `vial_photos/` (already restored before this pass): `01_`?`11_*`
  - **3** hard negatives from Recycle Bin: `empty_000`?`empty_002` (originally under `images/train/`)
  - **4** new live R/I labels written during recovery: `vial_0001`?`vial_0004` (not old-dataset restore)
- **Not found anywhere:** old `placed_*.jpg`, prior `vial_0005`?`vial_0020` (and siblings), `review/mislabeled/vial_0013`, prior `split_70_10_20` contents. Chat/workspace assets are mostly overlay screenshots, not clean labeled stills.
- **Rebuilt:** `split_70_10_20/` from current raw (seed 42): **12 train / 2 validate / 4 test** (15 positives, 3 empty negatives). Too small to trust prior race accuracies.
- **Review dirs recreated empty:** `pending/`, `wrong/`, `incorrect/`, `mislabeled/` (plus a few new `incorrect/` stills from today?s live run).
- **Re-capture needed:** yes ? aim to rebuild toward ~100+ real Orbbec stills (positives + hard negatives) via R/W/I review; `runs/bon-winner` weights remain usable meanwhile.

## Snapshots and review (learn from mistakes)

- While tracking: save clean stills (~every 3s when boxed), max **50 pictures per run** total.
- After **Q**, review each still:
  - **R** ? box is the vial ? `raw/` + YOLO label (train positive)
  - **W** ? not a vial ? `raw/` with **empty** `.txt` (hard negative for next train) + `review/wrong/`
  - **I** ? vial present but wrong/partial box ? draw full vial on the **same** review window ? correct label to `raw/`; failed box not trained
  - **N** ? skip
- Do **not** auto-train at end of a run. User runs train when ready.
- Mislabel example removed: `vial_0013` (box on mason jar) ? `vial_dataset/review/mislabeled/` (file lost in wipe; quarantine folder recreated empty).

## How to train after new labels

```powershell
cd E:\Desktop\Jessica\Camera
python train_vial_model.py label
python train_vial_model.py split
python train_vial_model.py train
# Fine-tunes from runs/bon-winner (Attempt E) by default; exports runs/vial-live/weights/best.pt
python depthcamera.py --dual-independent
# Classic coupled dual still available:
python depthcamera.py --dual
```

Holdout eval (IoU ? 0.5):

```powershell
python eval_asymmetric_split.py --weights runs\vial-live\weights\best.pt --out training_reliability_metrics.json
```

For a full re-race on the 70/20/10 split, rebuild with `make_split_70_10_20.py` or `python train_vial_model.py split`, then compare test accuracy % again and update this file.

## Round 1 train race (2026-10-02) ? accelerated

- **Closed early** via interim `best.pt` holdout eval (conf 0.25, IoU?0.5) on existing MAIN `split_70_10_20` (**40 / 6 / 11**, seed 42). Supersedes supervisor `91b7b705` waiting on full 10-way train completion.
- **Winner:** agent **#7** `train-r1-07-cf168274` ? MVAug mild perspective + freeze/unfreeze **phase1** checkpoint ? **100%** test (11/11), recall 1.0, 0 empty FP. Phase2 was worse (empty FP).
- **Applied:** `runs/train-race/r1-winner/weights/best.pt` and `runs/vial-live/weights/best.pt` (`find_weights` prefers vial-live).
- **Incomplete when stopped:** 8 / 10 trainers. Details: `TRAIN_RACE_R1_RESULT.md`.
- **Round 2 (lean):** done ? see below / `TRAIN_RACE_R2_RESULT.md`.

## Round 2 train race (2026-10-02) ? lean 3-way

- **Complete:** all 3 `RESULT.md` (`train-r2-01/02/03-*`). Gate: conf 0.25, IoU?0.5 on MAIN split 40/6/11 seed 42.
- **R2 winner:** agent **#2** `train-r2-02-c13f659b` ? low-LR AdamW+cos, freeze-5, weak-angle aug ? **100%** test, recall 1.0, 0 empty FP. Archived: `runs/train-race/r2-winner/weights/best.pt`.
- **Live at R2 close:** stayed R1 (pure tie, no clear beat). Details: `TRAIN_RACE_R2_RESULT.md`.

## Round 3 train race (2026-10-02) ? final lean 3-way

- **Complete:** all 3 `RESULT.md` (`train-r3-01/02/03-*`). Same holdout gate as R1/R2.
- **R3 winner (race archive):** agent **#2** `train-r3-02-e05d17b9` ? dual-seed lean FT; **100%** test. Archived `runs/train-race/r3-winner/`.
- **Live weights:** remain **R1** (`runs/vial-live` / `r1-winner`) unless a later run **clearly** beats holdout. Do not promote on tie.
- **All-rounds table:** `TRAIN_RACE_R3_RESULT.md`.

## User feedback ? weak-angle gap (still open)

- **Feedback:** One camera angle is much better than the other; fine-tune / recapture to close the gap (no peer soft-locate inventing boxes).
- **Why the lag:** unstratified race split; cam tags previously dropped on R/W/I?raw; cam1 under-sampled. See `TRAIN_RACE_R3_NOTES.md` + `r3-view-balance.json`.
- **Aggressive oversample FAILED (~16:11):** test **81.82%** (worse than live **100%**) ? NOT promoted. Archive `r3-weakangle-oversample/`. Notes: `TRAIN_RACE_R3_WEAKANGLE_RESULT.md`.
- **2026-10-02 ~16:20:** ?train it again? ? launched **mild** FT from vial-live: `train_r3_04_mild_live.py` (PID **6868**, log `_mild_live_train.log`). Small LR 4e-4, light aug, ?12 ep, new Orbbec train-only, **no** hardness/perspective oversample. Promote only on clear beat.

## Live tracker behavior (decided)

- Overlay labels the box **vial** (plus distance) via `annotate()` for single and dual.
- Prefer clear-capped vial (white lid + clear-cup depth cue); reject YOLO-only wall/desk boxes.
- Full-vial aspect roughly **1.0–1.65**; short lid-only boxes rejected on **R** / quarantined from train.
- **Pose-aware accept (live + R):** upright lid→cup; near-square **top-down** lid (h/w ~0.72–1.28); elongated **sideways/fallen** lid+cup (h/w ~0.45–0.85) with white-cap band on left/right + sane `vial_scale` (long≈H, short≈Ø). Flat lid-strips (h/w < 0.45) still rejected. Walls/windows still fail white-cap + scale + structure.
- Tiny-weight **ORB assist** (default on, `--no-orb-assist`): match a few keypoints on white-cap crops across cams for soft dual "orb agree" / `choose_box` boost (~0.08). Never invents boxes; empty matches behave as before. YOLO remains primary.
- Reach limit **1.5 m**; blur/dim depth past **600 cm**.
- `fit_vial_shape` shrinks loose boxes toward lid width and cup height (skips top-down / sideways).

## Research Agent 01 ? Weighted Boxes Fusion (WBF) late fusion / detection voting

**Theme:** Multi-view late fusion / detection voting (independent detectors, fuse boxes/scores).

**Primary method:** Weighted Boxes Fusion (WBF) with cross-view 3D association ? run the shared vial YOLO independently on each Orbbec RGB frame, lift white-cap detections to 3D with depth, associate across cameras by proximity (or sticky peer ID), then fuse matched boxes/scores with WBF or a weighted vote so dual agreement boosts confidence and single-view weak FPs stay low.

**Why multi-angle:** Clear plastic glare often blinds one view while the other still sees the opaque white cap; late fusion keeps the current dual stack modular and upgrades `locate_pair` peer assist from temporal stickiness to same-tick score consensus. Distances stay per-camera; fuse identity/score only. Still run clear-vial gating (white lid + clear-cup depth holes) after fusion so dual mason-jar agreement cannot pass.

**Pros / cons (vial + depth):** Pros ? modular with Attempt E weights; survives one-cam drop; better recall under glare. Cons ? pure 2D IoU across angled views is wrong (need depth?3D association); clear-cup depth holes make 3D lift noisy unless capped on the lid; needs loose sync.

**Citations:**
- Solovyev et al., Weighted Boxes Fusion ? https://arxiv.org/abs/1910.13302 ? https://github.com/ZFTurbo/Weighted-Boxes-Fusion
- Farooq et al., late-fusion multi-camera YOLOv7 + association ? https://www.propulsiontechjournal.com/index.php/journal/article/view/5357
- Ultralytics sensor fusion (late = merge boxes/scores) ? https://www.ultralytics.com/glossary/sensor-fusion
- Isaac & Breckon, epipolar-weighted multi-view detection merge ? https://breckon.org/toby/publications/papers/isaac20multiview.pdf
- Detection voting toolkit ? https://github.com/abhiksark/detection-fusion

**Apply to dual Gemini 335L:** shared `best.pt` ? per-cam predict ? lid-depth ? match ? `ensemble-boxes` WBF or simple dual-score vote ? clear-vial gate ? annotate. Full write-up: `plan_research/agent_01.md`.

## Research Agent 02 ? Epipolar Multi-View Detection Confidence (Isaac-Medina et al.)

**Theme:** Stereo / epipolar geometry + cross-view consistency (validate detections between the two Orbbecs).  
**Primary method:** Isaac-Medina, Willcocks, Breckon ? *Multi-view Object Detection Using Epipolar Constraints within Cluttered X-ray Security Imagery* (ICPR 2020). Detector-agnostic post-process: estimate \(F\), score box centroids by distance to the peer epipolar line, fuse that geometric probability with YOLO confidence / NMS. Detail: `plan_research/agent_02.md`.

- **Overview:** Run Attempt-E YOLO per cam ? project cap/box center via \(F\) ? epipolar distance ? multi-view confidence ? accept/boost only geometrically consistent peers (optional Yang normalized cross-distance).
- **Multi-angle benefit:** Rejects one-cam FPs (wall/desk lookalikes); confirms same physical vial without averaging depth; works when clear cups break feature matching (paper?s X-ray analogy).
- **Pros for clear vial+depth:** Cap center is a stable correspondence; cheap; no retrain; fits `locate_pair` peer assist.
- **Cons:** Needs rigid dual mount + decent \(F\); lid-only boxes bias centroids (use white-cap point); cannot invent a missing view; coplanar distractors on the same epipolar line still need clear-cup depth gates.
- **Citations:**
  - https://breckon.org/toby/publications/papers/isaac20multiview.pdf
  - https://doi.org/10.1109/ICPR48806.2021.9413007
  - https://github.com/KostadinovShalon/yolo_multiview
  - https://learnopencv.com/introduction-to-epipolar-geometry-and-stereo-vision/
  - https://github.com/labvisio/Multi-Object-Triangulation-and-3D-Footprint-Tracking (practical YOLO + \(d'_\mathrm{cross}\))
- **Orbbec sketch:** One-time stereo/\(F\) calibrate ? in `locate_pair`, score peer pairs on cap centers ? gate sticky soft-retry with epipolar threshold ? keep per-cam distances; no detector rewrite.

## Research Agent 03 ? MVTrans (Multi-View Transparent Perception)

- **Method:** MVTrans (ICRA 2023) ? multi-view RGB cost-volume network for transparent objects (depth + segmentation + 3D box + pose). Prefers multi-view geometry over broken RGB-D depth on clear plastic.
- **Multi-angle fit:** Dual Gemini 335L maps cleanly to published **2-view** weights; early fusion is stronger than today?s late `locate_pair` peer assist when one cam has glare/refraction. Needs a one-time cam0?cam1 extrinsic (not in live code yet).
- **Pros:** Built for transparent depth failure; joint 3D outputs; public 2-view checkpoints + Syn-TODD. **Cons:** Calibration required; heavier than YOLOv8s; syn?real gap for white-capped vials; don?t replace Attempt E without a re-race.
- **Orbbec sketch:** Keep `runs/bon-winner` live; offline sync RGB pair ? MVTrans 2-view ? project boxes; optional per-cam **TransCG/DFNet** (~17 ms) only to fill cup depth under existing gates. Docs/eval only ? no detector rewrite, no dataset wipe.
- **Write-up:** `plan_research/agent_03.md`
- **Citations:** https://arxiv.org/abs/2302.11683 ? https://ac-rad.github.io/MVTrans/ ? https://github.com/ac-rad/MVTrans ? ClearGrasp https://arxiv.org/abs/1910.02550 ? TransCG/DFNet https://arxiv.org/abs/2202.08471 ? https://www.graspnet.net/transcg

## Research Agent 04 ? ChArUco Stereo Extrinsic Calibration + Hartley?Sturm Optimal Triangulation

- **Theme:** Extrinsic calibration + 3D projection/triangulation of dual detections into a shared world frame (camera-0).
- **Method:** Offline ChArUco `stereoCalibrate` with `CALIB_FIX_INTRINSIC` ? `(R,T)`; online undistort paired lid/box points ? `triangulatePoints` (optionally Hartley?Sturm L2-optimal refine) ? metric XYZ in cam-0.
- **Multi-angle fit:** Partial FOV overlap OK (unique ChArUco IDs); Gemini 335L 8-pin sync; works with sparse YOLO boxes from `locate_pair` without dense cross-cam stereo.
- **Pros:** One metric vial pose; mature OpenCV stack; complements sticky-peer; scale from board square size; path to `calibrateMultiview` if more cams.
- **Cons:** Needs calibration session + remount recalib; bad peer correspondence ? bad 3D (use Sampson/`F` gate); weak geometry when rays nearly parallel; linear DLT suboptimal vs Hartley?Sturm.
- **Orbbec apply:** Calibrate/triangulate on RGB; keep D2C depth for lid/clear-cup gating; world origin = index 0; after `locate_pair`, triangulate when both boxes valid; prefer HW sync + AE off for calib.
- **Full write-up:** `plan_research/agent_04.md`
- **Citations:** [OpenCV multiview calib](https://docs.opencv.org/5.0/tutorials/calib3d/camera_multiview_calibration/multiview_calibration.html) ? [OpenCV calib3d](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html) ? [Hartley?Sturm PDF](https://perception.inrialpes.fr/Publications/1997/HS97/HartleySturm-cviu97.pdf) ? [Hartley?Zisserman MVG](https://www.cambridge.org/us/universitypress/subjects/computer-science/computer-graphics-image-processing-and-robotics/multiple-view-geometry-computer-vision-2nd-edition) ? [OpenCV stereo+triangulation walkthrough](https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html) ? [Orbbec Gemini 335L](https://store.orbbec.com/products/gemini-335l) ? [Orbbec G330 docs](https://www.orbbec.com/docs/g330-about-orbbec-gemini-330-series/) ? [Gemini datasheet sync](https://d1cd332k3pgc17.cloudfront.net/wp-content/uploads/2024/06/Orbbec-Gemini-330-Series-Datasheet-v1.1-2.pdf)

## Research Agent 05 ? ReST (Reconfigurable Spatial-Temporal Graph MCMOT)

- **Method:** ReST (Cheng et al., ICCV 2023) ? online multi-camera MOT via spatial-then-temporal graph link prediction; fuses cross-view detections before temporal tracklets (avoids brittle single-cam tracklet input).
- **Multi-angle fit:** Spatial graph links same-time detections across cameras; Graph Reconfiguration merges them; Temporal graph carries fused IDs over frames ? natural dual-Orbbec peer association + continuity when one view occludes.
- **Pros:** Online; open code; Wildtrack SOTA (IDF1 85.7 / MOTA 81.6); geometry/projection cues strong even with weak ReID; matches overlapping FoV dual cams.
- **Cons:** Needs camera?plane calibration; person/OSNet-centric; full GNN heavier than sticky-peer; vial training data insufficient for full port.
- **Orbbec apply:** Formalize `locate_pair` peer assist as same-frame spatial match on **cap 3D** in a shared bench frame; keep per-cam distances; optional later ReST SG/TG if multi-vial tracking grows. Details: `plan_research/agent_05.md`.
- **Citations:** [ICCV PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_ReST_A_Reconfigurable_Spatial-Temporal_Graph_Model_for_Multi-Camera_Multi-Object_Tracking_ICCV_2023_paper.pdf) ? [HTML](https://openaccess.thecvf.com/content/ICCV2023/html/Cheng_ReST_A_Reconfigurable_Spatial-Temporal_Graph_Model_for_Multi-Camera_Multi-Object_Tracking_ICCV_2023_paper.html) ? [code](https://github.com/chengche6230/ReST) ? [supplemental](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Cheng_ReST_A_Reconfigurable_ICCV_2023_supplemental.pdf)

## Research Agent 06 ? TransReID (Transformer Object ReID)

- **Theme:** Cross-camera appearance embeddings to match the same vial across Orbbec angles (ReID adapted from person/vehicle to vials).
- **Method:** TransReID (ICCV 2021) ? ViT backbone + Jigsaw Patch Module (JPM) + Side Information Embeddings (SIE) for camera/viewpoint IDs; ID + triplet embedding match at inference.
- **Multi-angle:** Global self-attention + overlapping patches handle lid-vs-side views; SIE absorbs Gemini 335L #0/#1 style gap; cosine similarity on crops associates peers without homography.
- **Pros:** True object ReID (person + vehicle benchmarks); SIE fits dual cams; JPM helps bad/partial YOLO boxes; official code; global-`[cls]`-only inference for speed.
- **Cons:** Heavier than OSNet; clear identical vials are weakly discriminative; needs identity-linked dual-view crops (current raw ~18 is insufficient); person/vehicle pretrain transfers poorly to transparent plastic.
- **Orbbec apply:** Keep YOLO Attempt E for detect; embed both `locate_pair` crops; gate sticky peer assist by ReID sim ? ?; depth stays per-camera on white cap; collect cam0/cam1 paired IDs during R/W/I recapture.
- **Full note:** `plan_research/agent_06.md`
- **Citations:**
  - https://openaccess.thecvf.com/content/ICCV2021/papers/He_TransReID_Transformer-Based_Object_Re-Identification_ICCV_2021_paper.pdf
  - https://arxiv.org/abs/2102.04378
  - https://github.com/damo-cv/TransReID
  - OSNet fallback: https://github.com/KaiyangZhou/deep-person-reid
  - ReidTrack (YOLO+embed clustering): https://openaccess.thecvf.com/content/CVPR2023W/AICity/papers/Specker_ReidTrack_Reid-Only_Multi-Target_Multi-Camera_Tracking_CVPRW_2023_paper.pdf

## Research Agent 07 ? MVAug (Two-Level Multi-View Augmentation)

- **Theme:** Domain generalization / multi-view data augmentation so one weak viewpoint still works.
- **Method:** **MVAug** (Engilberge et al., WACV 2023) ? view-level + scene-level homography augs that preserve cross-camera alignment; per-view detection loss regularizes single-camera features. YOLO-friendly slice: **DG-BEV Dynamic Perspective Augmentation** (random pose ? plane homography warps). Optional later: **GeoShift** on unlabeled weak-cam frames.
- **Why it fits:** Dual Gemini 335L + shared YOLO; scarce Orbbec labels; train-time viewpoint diversity without changing live `locate_pair` / clear-vial gating.
- **Orbbec apply (docs only):** Capture synced dual stills; mild perspective/homography augs in train (keep HSV-heavy E, avoid B?s strong-aug collapse); optional single-view dropout; measure cam0-only vs cam1-only accuracy after re-capture.
- **Full write-up:** `plan_research/agent_07.md`
- **Citations / links:**
  - MVAug PDF: https://openaccess.thecvf.com/content/WACV2023/papers/Engilberge_Two-Level_Data_Augmentation_for_Calibrated_Multi-View_Detection_WACV_2023_paper.pdf
  - MVAug arXiv: https://arxiv.org/abs/2211.11663
  - MVAug code: https://github.com/cvlab-epfl/MVAug
  - DG-BEV: https://arxiv.org/abs/2303.01686
  - GeoShift: https://arxiv.org/abs/2301.05496 ? https://github.com/vidit09/geoshift

## Research Agent 08 ? MVTrans (Multi-View Transparent Perception)

- **Method:** MVTrans (Wang et al., ICRA 2023) ? end-to-end **multi-view RGB** network for transparent/specular objects; plane-sweep matching volume ? depth + instance seg + 3D OBB/pose. **Does not trust raw RGB-D depth** on glass/plastic.
- **Why it fits:** Clear vials already produce Orbbec depth holes; dual Gemini 335L are a natural **2-view** input (paper ablations: 2/3/5 views; 2-view already beats KeyPose/SimNet on real pose).
- **Pros:** Purpose-built for glass/plastic vessels; multi-task 3D outputs; public code/weights + Syn-TODD (~2k scenes ? 57 views); outperforms RGB-D pose when depth is wrong.
- **Cons:** Heavy vs YOLOv8s; needs **extrinsics** between cams; class domain ? white-cap vial; overkill for current 2D box + cap ranging.
- **Orbbec apply:** Keep Attempt E YOLO live; steal multi-view consistency + distrust-cup-depth ideas; if prototyping, calibrate dual cams and run offline 2-view MVTrans ? do not replace `locate_pair` until latency/fine-tune proven. Prefer paired dual-frame recapture for future multi-view train.
- **Full write-up:** `plan_research/agent_08.md`
- **Links:** [arXiv](https://arxiv.org/abs/2302.11683) ? [project](https://ac-rad.github.io/MVTrans/) ? [code](https://github.com/ac-rad/MVTrans) ? [Syn-TODD](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/LQKTXE) ? related: [ClearGrasp](https://arxiv.org/abs/1910.02550) ? [TransLab](https://arxiv.org/abs/2003.13948) ? [transparent review](https://arxiv.org/abs/2304.00157)

## Research Agent 09 ? UKF Multi-View 3D State Fusion

- **Method:** Unscented / Kalman Bayesian fusion of dual-camera measurements into one 3D vial (cap) state; sequential per-camera updates (UKF if using 2D box projection; linear KF/EKF if using deprojected RGB-D 3D points). Production analog: NVIDIA DeepStream MV3DT confidence-weighted peer Kalman fusion.
- **Why multi-angle:** Cam0 and cam1 both update the same world-frame state; one-camera blinks still get predict + single-view update; depth holes get large \(R\); Mahalanobis gating rejects inconsistent boxes; covariance = live confidence (better than raw distance averaging).
- **Orbbec apply:** Calibrate extrinsics between Gemini 335L indexes 0 and 1; keep YOLO + clear-vial + `locate_pair` as measurement generators; after each dual tick, predict then update from each valid white-cap 3D point (or box), then show **fused** distance. Sticky peer memory can stay for detection assist; Kalman owns 3D state. Start simple (3D-point KF), upgrade to UKF+ellipsoid if needed.
- **Pros:** Principled dual fusion, occlusion tolerance, uncertainty-aware depth, real-time for one object, scales to N cams.
- **Cons:** Needs extrinsics; tune \(Q\)/\(R\); USB async timing; overkill if only peer boxing matters without shared 3D.
- **Docs:** `plan_research/agent_09.md`
- **Citations:**
  - Ma et al., UKF multi-camera 3D from boxes ? https://arxiv.org/abs/2511.17609
  - Julier & Uhlmann, Unscented filtering ? https://doi.org/10.1109/jproc.2003.823141
  - NVIDIA MV3DT Kalman peer fusion ? https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_MV3DT.html
  - Carraro et al., RGB-D camera-network Kalman ? https://arxiv.org/abs/1907.12112
  - Bredereke et al., YOLO + multi-cam EKF ? https://arxiv.org/abs/2503.04322

## Research Agent 10 ? MVDeTr (Shadow Transformer)

- **Method:** MVDeTr ? multiview detector that projects per-camera CNN features onto a shared ground plane, then fuses them with a **shadow transformer** (multiview deformable attention + camera/position embeddings). Successor to convolutional **MVDet**; related lineage: Deformable DETR / DETR3D / PETR (ideas only).
- **Multi-angle:** Needs synced, calibrated overlapping views. Features ? BEV project ? transformer attends differently per location/camera (?shadow-like? perspective distortion) ? ground-plane occupancy heatmap + optional per-view foot/box losses. **View-coherent aug:** warp each view, then invert before BEV so geometry stays consistent.
- **Pros:** Geometry-aware fusion; camera-adaptive attention beats plain BEV conv; open code trains on one 2080 Ti; strong Wildtrack/MultiviewX MODA. **Cons:** Pedestrian BEV occupancy, not vial xyxy + cap depth; needs extrinsics/world plane; tiny recovered dataset cannot train it; CUDA deformable ops + full stack ? value vs YOLO + `locate_pair`.
- **Orbbec apply (no rewrite):** Keep Attempt E YOLO + dual `locate_pair`. Borrow: same-timestamp peer trust when one view is weak; optional later ?does A project near B?? if calibrated; dual-synced capture for any future cross-view loss. Do **not** swap in MVDeTr/DETR3D/PETR for live tracking.
- **Detail note:** `plan_research/agent_10.md`
- **Citations:** [arXiv:2108.05888](https://arxiv.org/abs/2108.05888) ? [ACM MM](https://dl.acm.org/doi/10.1145/3474085.3475310) ? [code](https://github.com/hou-yz/MVDeTr) ? MVDet [arXiv:2007.07247](https://arxiv.org/abs/2007.07247) ? Deformable DETR [arXiv:2010.04159](https://arxiv.org/abs/2010.04159) ? DETR3D [arXiv:2110.06922](https://arxiv.org/abs/2110.06922) ? PETR [arXiv:2203.05625](https://arxiv.org/abs/2203.05625)

## 3D vial model from pictures ? identify on any cam (2026-10-02)

**User ask:** when one camera cannot find the vial, use pictures to know what the vial looks like in 3D, then identify it from either Orbbec.

**Chosen approach (practical, CPU-friendly):** a **parametric 3D vial** = opaque white **lid cylinder** + clear **cup cylinder**, sized in mm from labeled boxes + white-cap depth. Lift a strong-cam detection with lid depth ? reproject into the weak cam as a **search ROI / proposal**. Heavy NeRF / Gaussian / Open3D ICP deferred (no Open3D installed; clear-cup depth holes hurt ICP).

**Policy:** `--dual-independent` stays default with **no soft peer invent**. New **opt-in** flag `--3d-reproject-assist` may propose a geometrically reprojected box; local clear-vial / soft-clear / cap gates must still accept it. Distances stay per-camera. Does **not** promote weak YOLO weights; orthogonal to mild R3-04 FT.

**Calib:** size fit works from existing stills + `distance_mm` sidecars. Cross-cam reproject needs `stereo_extrinsics.json` (`R`,`t` cam0?cam1) from ChArUco/chessboard (Agent 04). Placeholder file is written by `python vial_3d_model.py fit` ? replace before trusting live assist.

**Module / CLI:** see `VIAL_3D_README.md` (copy-paste). Short:
```powershell
cd E:\Desktop\Jessica\Camera
python vial_3d_model.py fit
python vial_3d_model.py show
python vial_3d_identify.py score-stills --limit 20
python vial_3d_model.py reproject-demo --box 100,120,48,70 --distance-mm 400
python stereo_calib_stub.py status   # PLACEHOLDER until ChArUco
# A/B live (only after real extrinsics): default YOLO vs opt-in assist
python depthcamera.py --dual-independent
python depthcamera.py --dual --3d-reproject-assist
```

**vs more YOLO FT:** aggressive weak-angle oversample already failed holdout; geometry assist avoids that risk. Mild FT may still help appearance; 3D assist helps when the weak view is geometrically known but YOLO/gates miss.

**Capture next:** orbit / multi-angle stills (both cams); same-tick dual pairs with vial in both FOVs; ChArUco for extrinsics; keep mounts rigid after calib. Prefer R-accept that keeps `distance_mm` in raw sidecars (now preserved).

**Full write-up:** `plan_research/agent_3d_vial_model.md`.

## Research Supervisor ? Synthesis

**Constraint stack (do not undo):** Attempt E YOLO (`bon-winner`) + BoN **#3 asymmetric per-camera gates** + `locate_pair` sticky peer + clear vial / white-cap depth + **per-camera distances (no raw cm average)** + no Mask R-CNN/SAM mandate + protect `vial_dataset`.

### Top 3 for this dual Gemini 335L clear-vial setup

1. **Parametric 3D vial + gated reproject assist** (this section / `agent_3d_vial_model.md`) ? Fit lid+cup mm from stills; lift strong-cam lid depth; opt-in reproject ROI to weak cam after extrinsics. Respects dual-independent default.
2. **Epipolar peer gating (Agent 02 ? Isaac-Medina)** ? One-time $F$/stereo calibrate, score white-cap centers; pairs with 3D reproject. Cheap, no retrain, rejects one-cam wall/desk FPs, keeps per-cam range.
3. **ChArUco extrinsics + Hartley?Sturm triangulation (Agent 04)** ? Foundation for (1)?(2) and optional shared world XYZ of the lid; keep overlay cm per-camera.

*(WBF / dual-score vote after lid-3D associate ? Agent 01 ? remains a strong same-tick consensus upgrade once association exists.)*

### Try next (high-level only)

- Recapture synced dual stills (R/W/I) toward ~100+ labels; then mild **MVAug-style** perspective augs (Agent 07) on train only ? avoid Attempt B's strong-aug collapse.
- Defer full **MVTrans** (Agents 03/08 overlap), **MVDeTr** (10), **ReST** (05), **TransReID** (06) until data + calib exist; identical clear vials make appearance ReID weak.
- **UKF fused distance (09):** conflict with "never average distances" ? only as an optional labeled world-state overlay after extrinsics; do not replace per-cam lid ranging.

### Theme notes

- Agents **03** and **08** both land on MVTrans (RGB-D transparent vs transparent-literature briefs); treated as one literature cluster, not a merge error.
- Research sections are additive; BoN #3 / `locate_pair` / dataset recovery text above remains authoritative for live code.

- **R3 weak-angle oversample DONE (2026-10-02 ~16:11):** test **81.82%** (worse than live 100%) ? **not promoted**. Archive `runs/train-race/r3-weakangle-oversample/`.
- **R3-04 mild live FT DONE (2026-10-02 ~16:33):** pinned 40/6/11 + 127 Orbbec train-only ? test **81.82%** / recall 0.9 / empty FP 1 vs live 100% ? **not promoted**. Missed newest stills `vial_0172+`. Archive `runs/train-race/r3-04-mild-live/`.
- **Latest full-split train DONE (2026-10-02):** rebuild 70/10/20 from 202 labeled raw (**142/20/40**); mild FT from `vial-live`; test **100%** @0.25 / recall 1.0 / empty FP 0 on that holdout ? **tie vs live ? not promoted**. Archive `runs/train-race/latest-full-split/`.
- **Latest full-split-2 LAUNCHED (2026-10-02 ~17:00):** raw grew to **279** labeled pairs (**195/28/56**); mild FT from `vial-live` (?30 ep, patience 8, lr0=4e-4, freeze-5, no oversample). Log `runs/train-race/_latest_full_split_2_train.log`. Script `runs/train-race/train_latest_full_split_2.py`. **Do not promote** unless holdout clearly beats live (test@0.25 ? recall ? empty FP).
- **3D vial model refit (2026-10-02 ~17:00):** `python vial_3d_model.py fit` -> `vial_3d_dims.json` n=**301** (was 236); lid diam 55.22x16.55 / cup 50.8x58.67 / total H **75.22** mm. Leave `--3d-model-score-min` at **0.08**. Live loads JSON at process start (`load_dims`).
- **R3D Best-of-5 prior race DONE (2026-10-02):** pinned 70/30 seed=42 on 308 metric stills (**216 fit / 92 holdout**). Winner **Agent E** (distance-stratified) -> MAIN `vial_3d_dims.json` diam **54.22** x H **72.69** mm; holdout IoU>=0.5 correctness **0.6087** (tied C, won on mean 3D score). Archive `runs/train-race/r3d-race/`. Live: `python depthcamera.py --dual-independent` (restart to load winner). YOLO weights untouched; leave `--3d-model-score-min` at **0.08**.
