# Agent #03 — Template matching + ORB/AKAZE vial detector

Competition package under **`E:\Desktop\Jessica\Camera\ros2_ws\src\agent_03_template_orb`**.
Detects the clear plastic vial (opaque white screw cap) with classical CV only
(no YOLO weights).

## Dataset findings

| Item | Path / value |
|------|----------------|
| Raw stills | `Camera/vial_dataset/raw` — **373** jpg+txt (357 positives, 16 empty negatives) |
| Split | `Camera/vial_dataset/split_70_10_20` — **261 / 37 / 75** (seed 42) |
| Labels | YOLO `class cx cy w h` normalized, class `0: vial` |
| Extra stills | `Camera/vial_photos/` (11 early pairs; also folded into raw) |
| Image sizes | Mixed (Orbbec 640×480 dominant; some crops / phone stills) |
| Area frac (pos) | ~0.002 → ~0.97 (median ~0.022) — **large scale variance** |
| Aspect h/w | ~0.93 → ~1.67 (median ~1.19) — mostly upright; limited in-plane rotation in labels |

## Suitability check → **pivot within family**

| Method | Verdict | Why |
|--------|---------|-----|
| **ORB / AKAZE / SIFT alone** | **Poor fit as primary** | GT vial crops have **median ORB keypoints ≈ 0** (max ~9). Clear cup + smooth white cap lack stable corners/texture. Homography from inliers rarely fires. |
| **Multi-scale template matching** | **Primary (kept)** | Gray + Canny-edge NCC (`TM_CCOEFF_NORMED`) handles scale; templates built from train GT crops. |
| **Feature match + homography** | **Secondary assist** | Enabled when a template/scene pair yields enough good matches + RANSAC inliers; otherwise ignored. |

**Documented pivot:** stay in the template/feature family, but lead with **multi-scale template matching** (edge-augmented), not pure ORB/SIFT. White-cap consistency gate reduces empty-scene false positives.

## Layout

```
ros2_ws/src/agent_03_template_orb/
  package.xml  setup.py  setup.cfg  requirements.txt  README.md
  resource/agent_03_template_orb
  config/params.yaml
  launch/vial_detect.launch.py
  scripts/extract_templates.py
  scripts/evaluate.py
  data/templates/          # gray+edge PNGs + meta.json (extracted)
  agent_03_template_orb/
    vial_detector_node.py
    detector.py
    template_bank.py
    types.py
    image_codec.py
    msgs_compat.py
```

## Dependencies

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\agent_03_template_orb
pip install -r requirements.txt
python scripts/extract_templates.py
```

ROS2: same env as `orbbec_dual_sync` (Linux Humble/Jazzy or Windows RoboStack).
Optional `vision_msgs` for `Detection2DArray`; JSON String always published.

## Build & run

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws
# use existing build.cmd / pixi / WSL flow for this workspace
colcon build --packages-select agent_03_template_orb
# then source install (setup.bash / setup.bat)

ros2 launch agent_03_template_orb vial_detect.launch.py
# live:
ros2 launch agent_03_template_orb vial_detect.launch.py run_offline:=false
```

### Topics

| Topic | Type |
|-------|------|
| `/vial/detections` | `vision_msgs/Detection2DArray` (if available) |
| `/vial/detections_json` | `std_msgs/String` JSON (always) |
| `/vial/annotated` | `sensor_msgs/Image` |

### Key parameters (`config/params.yaml`)

| Param | Default | Meaning |
|-------|---------|---------|
| `run_offline` | `true` | Dataset playback vs live Image |
| `dataset_path` | `.../vial_dataset/raw` | Offline images |
| `templates_dir` | package `data/templates` | Template bank |
| `template_score_threshold` | `0.52` | NCC blend threshold |
| `enable_feature_match` | `true` | ORB/AKAZE/SIFT secondary |
| `feature_backend` | `AKAZE` | `AKAZE` \| `ORB` \| `SIFT` |
| `require_white_cap` | `true` | Reject boxes without bright low-sat cap band |

## Evaluate (no ROS required)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\agent_03_template_orb
python scripts/extract_templates.py
python scripts/evaluate.py --split test --conf 0.52
python scripts/evaluate.py --split test --no-feature --max-images 40
```

**Smoke (test, first 30, conf 0.52, template-only):** image accuracy **83.3%** (25/30 IoU≥0.5),
precision **0.89**, recall **0.83**, mean matched IoU **0.77**, ~1.2 FPS @ `match_max_width=480`.
Feature path contributed **0** hits on that subset (expected for textureless vials).

## Risks

- **Textureless vials:** feature path rarely contributes; do not expect SIFT/ORB recall alone.
- **Scale / clutter:** NCC can latch onto similar bright rectangles (paper, walls); white-cap gate helps but is not depth-aware.
- **Speed:** multi-template × multi-scale matching is heavier than YOLO; `match_max_width: 640` caps cost.
- **Pose:** train labels are mostly upright; sideways / top-down vials are weak without dedicated templates.
- **Empty negatives:** only 16 in dataset — tune `template_score_threshold` carefully for empty FP.

## Paths (absolute)

- Package: `E:\Desktop\Jessica\Camera\ros2_ws\src\agent_03_template_orb\`
- Templates: `E:\Desktop\Jessica\Camera\ros2_ws\src\agent_03_template_orb\data\templates\`
- Dataset: `E:\Desktop\Jessica\Camera\vial_dataset\`
