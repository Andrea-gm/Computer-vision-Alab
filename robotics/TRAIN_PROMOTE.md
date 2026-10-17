# Weight promotion gate

Live YOLO weights are resolved by `find_weights` in this order (never “newest under runs/”):

1. `runs/vial-live/weights/best.pt`
2. `runs/train-race/r1-winner/weights/best.pt`
3. `runs/bon-winner/weights/best.pt`

## Safe train → promote

```powershell
cd E:\Desktop\Jessica\Camera
python train_vial_model.py split --seed 42
python train_vial_model.py train --seed 42
# Staging only: runs/vial-ft/weights/best.pt  (live NOT overwritten)

python eval_asymmetric_split.py --weights runs\vial-ft\weights\best.pt --out training_reliability_metrics.json
```

Promote **only** when the candidate **clearly beats** current live on holdout:

1. higher test accuracy @ conf 0.25 (IoU ≥ 0.5), then
2. higher recall / fewer FN, then
3. lower or equal empty-negative FP

Ties or regressions → **do not promote** (archive under `runs/train-race/…` if useful).

```powershell
# After a clear beat:
python train_vial_model.py train --export-live
# or: copy runs\vial-ft\weights\best.pt → runs\vial-live\weights\best.pt
python depthcamera.py --dual-independent
```

Do not wipe `vial_dataset/raw`. Rebuild splits with `train_vial_model.py split` / `make_split_70_10_20.py` only.
