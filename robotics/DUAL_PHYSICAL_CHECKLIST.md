# Dual Orbbec physical / USB checklist

Prints once at `--dual-independent` (and classic `--dual`) startup.

## Before live

1. **USB3** — each Gemini 335L on its own USB 3 port (not a shared hub / USB2).
2. **Close Orbbec Viewer** and any other app holding the cameras.
3. **Aim** — both FOVs cover the vial (white lid through clear cup bottom).
4. **Glare** — reduce specular glare on the clear cup (angle lights / move vial slightly).
5. **Weak camera** — default `--weak-camera 0` (permissive gates on cam0, the hard physical view). Swap with `--weak-camera 1` if index 1 is harder.
6. **Stereo extrinsics** (for `--3d-reproject-assist` only):
   - `python stereo_calib_stub.py status`
   - If PLACEHOLDER: hold a chessboard/ChArUco in both FOVs → `capture` → `calibrate`

## Weak-angle labels (cam R accepts)

You cannot automate physical aim. For the lagging camera (default weak index **0**):

```powershell
cd E:\Desktop\Jessica\Camera
python depthcamera.py --dual-independent
# After Q: review stills — prefer R on weak-cam frames where the box is full vial
python depthcamera.py --review-only
```

Pending stills waiting: `vial_dataset/review/pending/` (cam-tagged).  
Incorrect boxes needing redraw: `vial_dataset/review/incorrect/` (I → drag full vial).

Capture tips for weak cam:

- Orbit so cam1 sees **lid + cup** (not lid-only foreshortening).
- Same distances you use live (~20–80 cm).
- Include hard negatives (empty bench) with **W**.
- Keep mounts fixed after stereo calib.

## Live both-agreement mode

```powershell
cd E:\Desktop\Jessica\Camera
python depthcamera.py --dual-independent --3d-reproject-assist
# Edge assist on by default; disable with --no-edge-assist
# Without real extrinsics, reproject is refused (YOLO + edge still run)
```
