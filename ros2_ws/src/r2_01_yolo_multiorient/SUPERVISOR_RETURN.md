# Supervisor return — Round-2 Agent #01 (multi-orientation YOLO)

**Package:** `E:\Desktop\Jessica\Camera\ros2_ws\src\r2_01_yolo_multiorient\`  
**Weights:** `...\r2_01_yolo_multiorient\models\best.pt`  
**Also:** `E:\Desktop\Jessica\Camera\runs\r2-01-multiorient\weights\best.pt`

## Round-1 failure (measured)

| Model | Upright (split test) | Sideways (`sideways_race/synth` positives) |
|-------|----------------------|---------------------------------------------|
| `runs/vial-live` @0.25 | P≈0.99 R≈0.96 | **P=0 R=0** (0/40) |

Raw labels: median GT `h/w≈1.59`, only 1/357 boxes with `h/w<0.85`. Sideways was never in the train distribution.

## What we changed

1. **Multi-orient dataset** (`data/multi_orient/`): real split + 90°/270° rotated copies + `sideways_race/synth` sideways/upright/negatives.
2. **Fast train subset** (`data/multi_orient_fast/`): 60 upright + 40 synth sideways + 100 rotated + 20 neg.
3. **Fine-tune** YOLOv8n from `runs/vial/weights/best.pt`, `degrees=180`, 15 epochs, imgsz 416, CPU.
4. **Inference rotate-TTA** (optional, default on) — 90° CW/CCW remap; no upright-only classical fallback.
5. **`scripts/evaluate.py`** reports **separate upright / sideways / negative** metrics.

### Subset definition

- **upright:** manifest `upright` OR `upright_*` OR GT median `h/w ≥ 0.95`
- **sideways:** manifest `sideways` OR `*sideways*` OR `*_r90`/`*_r270` OR GT median `h/w < 0.95`
- **negative:** empty GT / empty|hard_negative synth

## Eval numbers (required run)

Command:

```powershell
python scripts\evaluate.py --model models\best.pt --conf 0.25
```

| Subset | N images | Precision | Recall | mean IoU | image acc (pos) |
|--------|---------:|----------:|-------:|---------:|----------------:|
| **Upright** | 109 | **0.863** | **0.927** | 0.759 | 0.927 |
| **Sideways** | 113 | **0.682** | **0.929** | 0.718 | 0.929 |
| Negative | 51 | — | — | — | empty_fp **45/51** |
| All | 273 | 0.640 | 0.928 | 0.738 | 0.928 |

Artifact: `models/eval_results.json`.

Alternate (stricter FP): `--conf 0.35 --no-tta` → upright R=0.927 P=0.935; sideways R=0.779 P=0.880; empty_fp 24/51 (`models/eval_results_conf35_notta.json`).

**Verdict vs Round-1:** sideways recall **0.00 → 0.93** (default TTA@0.25). Upright recall remains high (~0.93). Precision / empty-FP are the remaining weak spots (TTA + conf 0.25 over-fires).

## Build / run

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws
colcon build --packages-select r2_01_yolo_multiorient --merge-install
call install\setup.bat
ros2 launch r2_01_yolo_multiorient vial_detect.launch.py
# live:
ros2 launch r2_01_yolo_multiorient vial_detect.launch.py run_offline:=false
```

Build verified: `Finished <<< r2_01_yolo_multiorient` with `--merge-install`.

Params: `config/params.yaml` — `model_path`, `confidence_threshold`, `dataset_path`, `tta_rotate`.

## Honest failure modes

1. **Empty / hard-negative FP** high at conf 0.25 with TTA — raise conf to ~0.35 or disable TTA for live FP control.
2. **Sideways precision 0.68** — extra boxes on clutter; IoU-matched recall is strong.
3. **Domain:** sideways train/eval leans on synth + 90° rotated stills; real desk-on-side may differ.
4. **CPU latency** ~0.8 s/frame with TTA; set `tta_rotate: false` for FPS.
5. **Top-down** not optimized (tagged `other` when present).
