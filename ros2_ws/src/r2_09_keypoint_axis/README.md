# r2_09_keypoint_axis — Round-2 Agent #09

**Cap / base keypoint skeleton → oriented vial box along the vial axis.**

Round-1 failed orientation (AABB growth assumed upright). This package detects
at least two endpoints (white **cap** + far-end **base**), then builds an
oriented box whose long axis follows cap→base for upright *and* sideways poses.

## Method

1. **Classical white-cap localization** — HSV low-sat / high-value mask +
   morphology; keep side-on lids and compact end-caps.
2. **Far-end (base) estimate** — ray-march from the cap along an angular prior
   (vertical-biased for side-on lids, horizontal-biased for end-caps, both
   families always searched). Score dark clear-cup pixels, depth holes, and
   lateral contrast; require a directional margin so top-down / wall blobs do
   not invent an axis.
3. **Oriented box** — keypoints define axis angle; box length ≈ cap–base span,
   diameter ≈ lid size; publish OBB + AABB cover + `theta`.
4. **Weak labels / YOLO seed** — when no keypoint GT exists, localize the white
   cap inside a YOLO AABB and place the base at the opposite end of that box
   (`scripts/derive_weak_keypoints.py`). At inference, YOLO proposals seed the
   same weak axis (strong on real `vial_dataset` photos).

No learned keypoint head is required for the reported metrics; classical +
weak YOLO axis is the shipped path.

## Package layout

```
r2_09_keypoint_axis/
  r2_09_keypoint_axis/   # detector, cap_finder, axis_skeleton, ROS node
  scripts/evaluate.py
  scripts/derive_weak_keypoints.py
  launch/vial_detect.launch.py
  config/params.yaml
  results/
```

## Offline eval

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\r2_09_keypoint_axis

# Synth upright vs sideways (classical + depth holes; no YOLO seed)
python scripts/evaluate.py --synth-only --no-yolo-seed --conf 0.40

# Real split (YOLO-weak axis)
python scripts/evaluate.py --split-only --split test --conf 0.35

# Weak keypoint export
python scripts/derive_weak_keypoints.py --split train --max 200
```

### Latest metrics (2026-10-09)

**Synth** (`sideways_race/synth`, classical, `--no-yolo-seed`):

| Split | Recall@0.5 | mean IoU | mean ∠err° | axisFollow |
|-------|-----------:|---------:|-----------:|-----------:|
| upright (6) | **1.000** | **0.931** | **0.0** | **1.000** |
| sideways (40) | **0.750** | **0.888** | **0.5** | **0.750** |
| negatives (22) | fp **0**/22 | | | |

FIRE (axisFollow &lt; 0.75 on upright or sideways): **False**

**vial_dataset test** (YOLO-weak axis):

| | P | R | mean IoU | mean ∠err° | axisFollow |
|--|--:|--:|---------:|-----------:|-----------:|
| overall / upright-ish | 0.841 | 0.932 | 0.630 | 0.0 | 0.920 |

Artifacts: `results/eval_synth_classical.json`, `results/eval_split_yolo.json`.

## ROS2 run

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws
# Requires ROS2 / RoboStack env with colcon on PATH:
colcon build --packages-select r2_09_keypoint_axis
.\install\setup.ps1

# Offline synth replay
ros2 launch r2_09_keypoint_axis vial_detect.launch.py run_offline:=true

# Live camera
ros2 launch r2_09_keypoint_axis vial_detect.launch.py run_offline:=false

# Node without colcon (dev):
pip install -e E:\Desktop\Jessica\Camera\ros2_ws\src\r2_09_keypoint_axis
python -m r2_09_keypoint_axis.vial_detector_node
```

Publishes:

- `/vial/detections` — `vision_msgs/Detection2DArray` with `bbox.center.theta` = vial axis
- `/vial/detections_json` — JSON including `cap` / `base` keypoints + OBB
- `/vial/annotated` — skeleton + oriented polygon overlay

## Risks

- **Hard angled / occluded synth** still miss ~25% recall; axis is correct when matched.
- **RGB-only classical** without depth over-fires on white clutter — YOLO seed (or depth) recommended live.
- **Top-down lids** intentionally suppressed (no clear cap→base axis in-plane).
- Weak YOLO axis inherits YOLO box quality; bad AABBs → skewed bases.
