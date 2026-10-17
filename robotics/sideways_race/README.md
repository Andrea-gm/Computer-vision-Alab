# Sideways-vial Best-of-N race (eval only)

Harness to score independent sideways / fallen vial detectors. **Do not promote a winner into live `depthcamera.py` from this folder alone** — supervisor gate required.

## Method API (contract for agents)

Drop a module in `sideways_race/methods/`:

```python
NAME = "my_method"  # optional; defaults to filename stem

def detect_sideways(color_bgr, depth_mm, intrinsics=None) -> list[tuple[int, int, int, int]]:
    """Return zero or more axis-aligned boxes (x, y, w, h) for sideways lid+cup.
    color_bgr: uint8 HxWx3 BGR
    depth_mm: float32 HxW millimeters (0 = invalid / hole)
    intrinsics: dict {fx,fy,cx,cy[,width,height]} or None
    """
    ...
```

Rules:

- Detect **sideways / fallen** vials only (white lid on left or right + clear body).
- Return `[]` on empty scenes, upright vials, and top-down lids (those are negatives).
- Boxes must cover **lid through cup**, not lid-only strips.
- Pure OpenCV/numpy preferred; importing `depthcamera` helpers is OK if you do not open cameras.

## Metrics

| Metric | Definition |
|--------|------------|
| **recall@IoU≥0.5** | Fraction of positive (sideways) stills with best pred↔GT IoU ≥ 0.5 |
| **FP / fp_rate** | Negatives where the method returns any box |
| **mean IoU** | Mean best IoU on positives (0 if miss) |
| **rank_score** | `recall − 0.35·fp_rate + 0.15·mean_IoU` |

## Synth set

76 programmatically generated RGB-D stills under `sideways_race/synth/`:

- `sideways_left_*` / `sideways_right_*` — **positives**
- `topdown_*` / `upright_*` / `empty_*` — **negatives**
- `real_wide_*` — widest `vial_dataset` labels if found (currently squat top-down, treated as hard negatives)
- `hard_sideways_left_*` / `hard_sideways_right_*` — 16 additive Round-3
  positives with ±15–30° rotation, partial occlusion, dim exposure, detached
  white clutter, and weak/striped/missing depth
- `hard_negative_*` — 8 additive cap-like clutter negatives

The original 52 samples and IDs are retained. An **old-set** score means
entries whose `kind` does not start with `hard_`; a **full-set** score uses
all manifest entries.

Regenerate deterministically (default seed 42). This rewrites generated files
inside `synth/`, rebuilds `manifest.json`, and refreshes copied `real_wide`
controls when the sibling `vial_dataset` is available:

```bash
python sideways_race/generate_synth.py
# or regenerate immediately before scoring:
python sideways_race/eval_sideways.py --regen
```

## Run eval

From `Camera/`:

```bash
python sideways_race/eval_sideways.py
```

Useful flags:

```bash
python sideways_race/eval_sideways.py --regen
python sideways_race/eval_sideways.py --only baseline_depthcamera method_a_cap_horizontal_grow
```

Results JSON: `sideways_race/results/latest.json` and stamped `race_*.json`.

## Baseline

`methods/baseline_depthcamera.py` wraps current `detect_vial_by_cap` + sideways geometry filter when `depthcamera` is importable.
