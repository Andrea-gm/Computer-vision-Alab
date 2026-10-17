# Round-2 Agent #08 — SAHI / tiled + multi-scale vial detection

ROS2 package: `E:\Desktop\Jessica\Camera\ros2_ws\src\r2_08_sahi_multiscale\`

Tiled (SAHI-style) inference + multi-scale full-frame passes + rotation TTA on
existing vial YOLO weights, so **small upright and sideways** vials are not
missed by single-scale `imgsz=640` inference.

## Tile strategy

| Knob | Default | Role |
|------|---------|------|
| Slice size | **320×320** | Each tile letterboxed to `imgsz=640` → ~2× effective resolution vs full 1280-wide frame |
| Overlap | **0.20** | Edge vials not clipped at tile borders |
| Full-frame scales | **1.0, 1.5** | Multi-scale on whole image (with rotation TTA) |
| Tile scales | **1.0 only** | Tiles only at native scale (FPS cap) |
| Rotation TTA | **0°, 90° CCW, 90° CW** on full-frame; tiles at **0° + 90° CCW** | Sideways vials map toward upright-trained appearance |
| Merge | NMS IoU **0.50–0.65** | Merge tile + full-frame + TTA hypotheses |

**Baseline mode** (`mode:=baseline`): one full-frame YOLO pass, no tiles / scales / TTA.

**Weights:** auto-resolve prefers `runs/bon-winner/weights/best.pt` (fires on both
orientations). Override with `model_path` / `--model` (vial-live, r1, r3, …).

## Metrics (prove recall ↑ vs single-scale)

Eval set: **small-object paste-up** — real `vial_dataset/raw` crops placed into
1280×720 canvases at max-side ≈64 px, upright or 90°-rotated (sideways).
IoU match ≥ 0.5, conf ≥ 0.25. Weights: `bon-winner`.

| Split | Mode | Recall | Precision | mean IoU | ms/frame | FPS |
|-------|------|--------|-----------|----------|----------|-----|
| **Upright** (n=20) | baseline | **0.100** | 1.000 | 0.804 | 584 | 1.71 |
| **Upright** (n=20) | **sahi** | **0.700** | 0.933 | 0.836 | 15974 | 0.06 |
| **Sideways** (n=20) | baseline | **0.050** | 1.000 | 0.757 | 429 | 2.33 |
| **Sideways** (n=20) | **sahi** | **0.700** | 0.933 | 0.851 | 13604 | 0.07 |

**Δ recall:** upright **+0.60**, sideways **+0.65** (SAHI − baseline).

**FPS tradeoff:** ~**2.0 → 0.07 FPS** (~**29×** slower). Use `mode:=baseline` for
live preview; `mode:=sahi` for high-recall offline / slow path.

Raw JSON: `eval_results.json`.

### Secondary: `sideways_race/synth` (domain stills)

Upright control GT from JSON notes (n=6); sideways positives (n=40). Same weights / IoU gate.

| Split | Mode | Recall | Δ vs baseline |
|-------|------|--------|---------------|
| Upright | baseline → sahi | 0.000 → **0.167** | **+0.17** |
| Sideways | baseline → sahi | 0.150 → **0.525** | **+0.38** |

Synth precision is low (cartoon RGB ≠ real training photos); recall still rises with
tiles + rotation TTA. JSON: `eval_synth.json`.

## Paths

| Item | Path |
|------|------|
| Package | `E:\Desktop\Jessica\Camera\ros2_ws\src\r2_08_sahi_multiscale\` |
| Weights (default) | `E:\Desktop\Jessica\Camera\runs\bon-winner\weights\best.pt` |
| Small-object generator | `r2_08_sahi_multiscale/small_object_set.py` |
| Synth domain | `E:\Desktop\Jessica\Camera\sideways_race\synth\` |
| Eval script | `scripts/evaluate.py` |
| Params | `config/params.yaml` |
| Launch | `launch/vial_detect.launch.py` |

## Run

### Offline eval (no ROS)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws\src\r2_08_sahi_multiscale
python scripts/evaluate.py
# upright + sideways small-object suite (default --suite both includes synth too)
python scripts/evaluate.py --suite small --small-per-orient 20
python scripts/evaluate.py --suite synth --out eval_synth.json
```

### ROS2

```bash
cd E:/Desktop/Jessica/Camera/ros2_ws
colcon build --packages-select r2_08_sahi_multiscale
# source install/setup.bash  (or Windows RoboStack equivalent)

ros2 launch r2_08_sahi_multiscale vial_detect.launch.py
ros2 launch r2_08_sahi_multiscale vial_detect.launch.py mode:=baseline
ros2 launch r2_08_sahi_multiscale vial_detect.launch.py run_offline:=false
```

### Topics

| Topic | Type |
|-------|------|
| `/vial/detections` | `vision_msgs/Detection2DArray` (if installed) |
| `/vial/detections_json` | `std_msgs/String` JSON (always) |
| `/vial/annotated` | `sensor_msgs/Image` |

## Layout

```
r2_08_sahi_multiscale/
  package.xml  setup.py  setup.cfg  requirements.txt  README.md
  config/params.yaml
  launch/vial_detect.launch.py
  scripts/evaluate.py
  r2_08_sahi_multiscale/
    detector.py          # baseline + SAHI/multiscale/TTA
    tiling.py            # slices, rotate/scale remap
    small_object_set.py  # upright/sideways stress stills
    vial_detector_node.py
    weights.py  types.py  image_codec.py  msgs_compat.py
```

## Notes

- Round-1 single-scale YOLO misses small / oddly framed vials; this package is the
  tiled recall recovery path, not a new trained architecture.
- `vial-live` is often silent on `sideways_race/synth` at conf 0.25; prefer
  `bon-winner` / r1 for orientation coverage (still an existing Camera weight).
- Precision stays high on the paste-up suite; the cost is latency (many passes).
