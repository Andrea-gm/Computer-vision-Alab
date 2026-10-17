# r2_10_fusion_multiorient — multi-orientation vial fusion

**Package:** `E:\Desktop\Jessica\Camera\ros2_ws\src\r2_10_fusion_multiorient`  
**Role:** Round-2 Agent #10 (fusion lead) — detect **upright and sideways** clear plastic vials with opaque white screw caps.

## Why Round-1 failed (even on “correct” orientation)

| Failure mode | Evidence in `agent_*` |
|---|---|
| **Upright-biased geometry** | Cap→body grow *down*, HOG window `(64,80)`, aspect gates ~0.95–1.75, templates from upright labels |
| **YOLO trained on upright stills** | `vial-live` strong on `split_*` test (~0.90 recall) but **0.00 sideways recall** on `sideways_race/synth` |
| **Ensemble ≠ multi-orient** | `agent_10` tied YOLO on upright; classical branch weak; no rotation TTA; no sideways eval |
| **Shape filters kill sideways** | Aspect / expansion priors reject fallen lid+cup AABBs (h/w ~0.4–0.8) |
| **Method-I without depth FPs** | Sideways race winner needs depth; RGB-only on upright photos floods FPs if naively fused |

## Fix (this package)

1. **YOLO (+ smart rotation TTA)** — native pose first; 90/180/270 only when native upright is weak  
2. **Depth-gated classical sideways** — prefer `sideways_race` Method-I when depth exists (0.95 recall / 0 FP on synth)  
3. **Orientation-aware NMS** — sideways hypotheses own overlapping YOLO lid-only boxes  
4. **Cap-axis refine** — YOLO only (classical sideways already full lid→cup)  
5. **Pose-aware empty veto** — depth + Method-I empty ⇒ veto weak *sideways* only; keep confident upright/top-down YOLO  
6. **Post-Method-I YOLO gate** — when Method-I already has a depth match, extra non-overlapping YOLO needs depth/body support or `conf >= 0.60` (param `extra_yolo_conf_after_method_i`)  
7. **Empty-scene YOLO floor** — `empty_scene_yolo_floor: 0.35` (param)  
8. **RGB-only classical sideways off** — `enable_rgb_classical_sideways: false` by default (ERROR_AUDIT)

Frame-level audit: `ERROR_AUDIT.md`. Post-gate audit JSON: `models/error_audit_post_gate.json`.

## Measured metrics (2026-10-09)

```text
python scripts/evaluate.py
```

Pre-gate baseline → post-gate (`models/eval_multiorient_post_gate.json`):

| Bucket | Precision | Recall | imgR@IoU≥0.5 | empty FP | n |
|---|---:|---:|---:|---:|---:|
| **Upright** (holdout) | 0.942 → **1.000** | 0.907 → **0.889** | 0.907 → 0.889 | 2 → **0** | 56 |
| **Sideways** (pos + true empty/neg) | 0.623 → **0.927** | **0.950** | **0.950** | **0** | 62 → **56** |
| **topdown_*** (pose control) | — | **0.000** (still missed) | 0 | — | 6 |
| **upright_*** synth (pose control) | — | **0.000** (still missed) | 0 | — | 6 |

Hard-sideways unrelated YOLO FPs (frame audit): **22 → 2**. Wrong frames overall: 43 → 29.

Evaluator honesty (ERROR_AUDIT): `topdown_*` / `upright_*` are reported as separate pose-control positives (not empty negatives). `real_wide_*` incomplete GT is excluded from headline metrics.

**Pass/fail vs Round-1 bar:** upright **PASS** · sideways **PASS** · overall **PASS**  
(Round-1 YOLO: upright R≈0.90, sideways unsupported; sideways baseline_depthcamera R=0.375.)

Full JSON: `models/eval_multiorient.json`. Notes: `SUPERVISOR_NOTES.md`.

## Layout

```
r2_10_fusion_multiorient/
  r2_10_fusion_multiorient/
    fusion.py           # cascade
    yolo_tta.py         # YOLO + rotation TTA
    classical.py        # upright + sideways (Method-I optional)
    nms.py              # orientation-aware NMS
    vial_detector_node.py
    image_codec.py / msgs_compat.py / types.py
  config/params.yaml
  launch/vial_detect.launch.py
  scripts/evaluate.py
  models/eval_multiorient.json
  requirements.txt
```

## Offline evaluate (no ROS)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\r2_10_fusion_multiorient
pip install -r requirements.txt
python scripts/evaluate.py
# ablations:
python scripts/evaluate.py --yolo-only --no-tta --conf 0.25
```

## Build / run (ROS2, Windows PowerShell)

Bare `colcon` / `ros2` fail until **Pixi (RoboStack)** and **`install\setup.bat`** are active. Use workspace wrappers from `ros2_ws` (always prefix **`.\`**):

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws
.\build_r2_10.cmd
.\launch_r2_10.cmd
```

Offline dataset replay (default `run_offline:=true` in `config/params.yaml`):

```powershell
.\launch_r2_10.cmd
```

Live dual-camera mode starts both publishers and opens two raw tkinter camera
windows independently of detector inference. Annotated windows follow as the
classical/Method-I detector processes frames:

```powershell
.\launch_r2_10.cmd run_offline:=false
```

Allow roughly 15–30 seconds. Do not run `launch_dual_sync.cmd` at the same
time because both launches would compete for the same USB cameras. If those
publishers are already running, reuse them instead:

```powershell
.\launch_r2_10.cmd run_offline:=false launch_cameras:=false
```

Live YOLO is disabled by default because Windows RoboStack
Torch/Ultralytics can natively terminate a ROS launch child with exit code 3.
Offline replay keeps the audited YOLO configuration. On a runtime where YOLO
has been independently proven not to abort, opt in explicitly:

```powershell
.\launch_r2_10.cmd run_offline:=false use_yolo_live:=true
```

That opt-in is in-process and is not crash-contained; do not use it on the
affected Windows runtime.

Override short-term fusion gates at launch (defaults live in `config/params.yaml`):

```powershell
.\launch_r2_10.cmd run_offline:=false `
  extra_yolo_conf_after_method_i:=0.60 `
  empty_scene_yolo_floor:=0.35 `
  enable_rgb_classical_sideways:=false
```

Equivalent PowerShell scripts: `.\build_r2_10.ps1`, `.\launch_r2_10.ps1` (or `.\launch_r2_10.cmd` if ExecutionPolicy blocks `.ps1`).

First-time ROS on this machine:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install_windows_ros2.ps1
```

Manual env (same session, after build):

```powershell
.\source_install.ps1
ros2 launch r2_10_fusion_multiorient vial_detect.launch.py run_offline:=false
```

Launch file: `launch/vial_detect.launch.py` — args include `config`,
`run_offline` (default `true`), `visualize` (default `true`),
`camera_preview` (default `true`), `use_yolo_live` (default `false`), and
`launch_cameras` (default `true`). `visualize` publishes/opens annotated
output; `camera_preview` controls the independent raw publisher windows.

| Topic | Type |
|---|---|
| sub `/orbbec/cam0\|cam1/color/image_raw` | `sensor_msgs/Image` |
| sub `/orbbec/cam0\|cam1/depth/image_raw` | `sensor_msgs/Image` (optional; enables sideways Method-I) |
| pub `/vial/detections` | `vision_msgs/Detection2DArray` (if installed) |
| pub `/vial/detections_json` | `std_msgs/String` JSON (always) |
| pub `/vial/annotated` | `sensor_msgs/Image` |

## Dependencies

See `requirements.txt`: `numpy`, `opencv-python`, `ultralytics`, `PyYAML`.  
ROS: `rclpy`, `sensor_msgs`, `std_msgs`, optional `vision_msgs`.  
Weights auto-resolve: `runs/vial-live/weights/best.pt` under `camera_root`.
