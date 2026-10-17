# Dual Orbbec time sync — ROS2 Nodes (colcon)

**Primary path: native Windows PowerShell (RoboStack / Pixi).** No WSL required.

`local_approx_sync.py` is **NOT ROS** (optional plain-Python debug). Do not treat
`python local_approx_sync.py` as the sync workflow.

**ROS / sync does not raise YOLO accuracy.** It only aligns frames
(`|t0 − t1| ≤ slop`, default 50 ms). Soft-peer invent off. Live 3D size-score off.

---

## Classic workflow (Windows) — copy-paste

**Fastest path after build:** one-shot launch — no separate `source_install` step.

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws
.\build.cmd
.\launch_dual_sync.cmd
```

`launch_dual_sync.cmd` / `run_dual_sync.ps1` activate Pixi, call `install\setup.bat`, set
`ORBBEC_CAMERA_ROOT`, and run `ros2 launch`. In PowerShell you **must** prefix local
scripts with `.\` (e.g. `.\source_install.cmd` — bare `source_install.cmd` is not found).

Manual loop (same as Linux conceptually): **colcon build → source install → ros2 launch**.
There is no bash `source` in PowerShell; use `.\source_install.cmd` / `.\source_install.ps1`
(imports `install\setup.bat`) or `call install\setup.bat` in cmd.

Verified on `E:\Desktop\Jessica\Camera\ros2_ws` with RoboStack Humble under `.pixi`.

### If PowerShell says “running scripts is disabled”

Default `ExecutionPolicy` blocks `.ps1`, so Pixi/ROS never activates (`colcon` /
`ros2` / `Enter-Ros2Base` look “not recognized”). Fix with either:

```powershell
# Option A — permanently allow local scripts for your user (recommended once):
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

```bat
REM Option B — use CMD entry points (no ExecutionPolicy; works from cmd or PowerShell):
cd /d E:\Desktop\Jessica\Camera\ros2_ws
build.cmd
launch_dual_sync.cmd
```

```powershell
# Option C — Bypass for one script (PowerShell stays Restricted):
cd E:\Desktop\Jessica\Camera\ros2_ws
powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_dual_sync.ps1
```

### Recommended: CMD (works under Restricted policy)

```bat
cd /d E:\Desktop\Jessica\Camera\ros2_ws

REM One-time only (RoboStack Humble + ultralytics in .pixi):
REM powershell -NoProfile -ExecutionPolicy Bypass -File .\install_windows_ros2.ps1

build.cmd
launch_dual_sync.cmd

REM Or manual source + launch in the same cmd window:
REM source_install.cmd
REM set ORBBEC_CAMERA_ROOT=E:\Desktop\Jessica\Camera
REM ros2 launch orbbec_dual_sync dual_sync_detect.launch.py
```

From PowerShell you can run the same `.cmd` files directly:

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws
.\build.cmd
.\launch_dual_sync.cmd
```

### PowerShell (after RemoteSigned or with Bypass -File)

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws

# One-time only (RoboStack Humble + ultralytics in .pixi):
# powershell -NoProfile -ExecutionPolicy Bypass -File .\install_windows_ros2.ps1

# 1) Activate Pixi/RoboStack so `colcon` and `ros2` exist in this shell
$pixiBin = Join-Path $env:LOCALAPPDATA "pixi\bin"
if (Test-Path $pixiBin) { $env:Path = "$pixiBin;$env:Path" }
. .\_ros_env.ps1
$script:RosWsRoot = (Get-Location).Path
Enter-Ros2Base | Out-Null
# (optional) interactive Pixi shell instead:  pixi shell

# 2) Build overlay (generates install\setup.bat — not bash)
colcon build --packages-select orbbec_dual_sync --merge-install

# 3) Source install space (Windows equivalent of: source install/setup.bash)
.\source_install.ps1
# Same idea via cmd:  .\source_install.cmd   or   call install\setup.bat

# 4) Camera import root for depthcamera / YOLO paths
$env:ORBBEC_CAMERA_ROOT = "E:\Desktop\Jessica\Camera"

# 5) Launch three ROS2 Nodes (cam0 + cam1 + sync_detect)
ros2 launch orbbec_dual_sync dual_sync_detect.launch.py
```

### Before launch (dual Orbbec USB)

1. Close **Orbbec Viewer** and any other app holding the cameras
2. Each camera on its **own USB 3** port (not a shared hub / USB2)
3. If open still fails: unplug both → wait a few seconds → replug → retry
4. Confirm one camera alone works (`python depthcamera.py --camera 0`) before dual ROS

Publishers stream color/depth only (no YOLO). YOLO loads in `orbbec_sync_detect`.
Launch staggers cam1 (~6s) and sync_detect (~12s) to avoid USB open races.

### What you should see

1. `ORBBEC_CAMERA_ROOT=E:\Desktop\Jessica\Camera` (set automatically by launch scripts)
2. `ros2 launch` starts `orbbec_cam0_pub`, then `orbbec_cam1_pub`, then `orbbec_sync_detect`
3. Publishers log `preview=off` — **expected** (raw USB preview disabled in YAML). The
   side-by-side annotated window comes from `orbbec_sync_detect` (`Preview ON …` in log;
   launch arg `preview:=true` by default). Headless: `preview:=false`.
4. With cameras plugged in: publishers stream; sync_detect runs ApproximateTime + detector
5. Without cameras: publishers exit with a clear error
   (`Failed to open Orbbec camera_index=…` / OpenCV `Camera index out of range`);
   sync_detect stays up waiting for topics

### Optional convenience scripts

Same steps wrapped for you (not required if you prefer the classic flow above):

```bat
cd /d E:\Desktop\Jessica\Camera\ros2_ws
build.cmd              REM Activate + colcon build --merge-install
launch_dual_sync.cmd   REM One-shot: activate + source install + ORBBEC_CAMERA_ROOT + ros2 launch
```

```powershell
# PowerShell equivalents (need RemoteSigned or Bypass -File):
cd E:\Desktop\Jessica\Camera\ros2_ws
.\build.ps1
.\run_dual_sync.ps1
```

Override config:

```bat
launch_dual_sync.cmd config:=E:\Desktop\Jessica\Camera\ros2_ws\src\orbbec_dual_sync\config\dual_sync.yaml
```

### Manual cmd.exe (same Pixi env)

```bat
cd /d E:\Desktop\Jessica\Camera\ros2_ws
call activate_ros2.cmd
colcon build --packages-select orbbec_dual_sync --merge-install
call install\setup.bat
set ORBBEC_CAMERA_ROOT=E:\Desktop\Jessica\Camera
ros2 launch orbbec_dual_sync dual_sync_detect.launch.py
```

### Orbbec / OpenCV / YOLO on Windows

This RoboStack pixi env already has OpenCV with **`CAP_OBSENSOR`** (verified) and
installs **ultralytics** into the same Python that runs `rclpy`. No separate
system Python bridge is required for Orbbec USB on native Windows.

- Desktop GUI / RViz / some C++ packages are flaky on Windows; this package is
  **ament_python** and builds **without** MSVC.
- Visual Studio 2022 C++ Build Tools only if `pixi install` or C++ deps fail:
  https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Prefer short paths; use `--merge-install` (avoid `--symlink-install`
  unless Developer Mode is on — otherwise WinError 1314).
- Do **not** paste Linux commands (`source`, `./build.sh`, `export`,
  `/mnt/e/...`) into PowerShell — they fail or resolve to the wrong path.
- Prefer `.\source_install.ps1` / `call install\setup.bat` over `.\install\setup.ps1`
  on RoboStack: the underlay often ships `Library\local_setup.bat` but not `.ps1`.

### Scripts

| Script | Role |
|--------|------|
| `build.cmd` / `source_install.cmd` / `launch_dual_sync.cmd` | **Preferred under Restricted policy** — pure CMD, no ExecutionPolicy |
| `activate_ros2.cmd` / `_ros_env.cmd` | Activate Pixi/RoboStack into current cmd session |
| `ros2_flow.cmd` | One-shot: build → source install → launch |
| `install_windows_ros2.ps1` | winget `pixi` if needed + `pixi install` + `ultralytics` (use Bypass -File if blocked) |
| `source_install.ps1` / `build.ps1` / `run_dual_sync.ps1` | PowerShell wrappers (RemoteSigned or Bypass -File) |
| `_ros_env.ps1` | Shared detection: Pixi / conda / `C:\dev\ros2_*` |
| `pixi.toml` | RoboStack Humble dependency pin |
| `enter_wsl.ps1` / `build.sh` | WSL/Linux alternate only |

### Install links

- RoboStack: https://robostack.github.io/GettingStarted.html  
- Humble Windows binary (fallback zip): https://docs.ros.org/en/humble/Installation/Windows-Install-Binary.html  
- Releases: https://github.com/ros2/ros2/releases  

| Option | Status |
|--------|--------|
| **RoboStack via Pixi** | Primary for this repo |
| Official Humble Windows zip | Fallback (~Patch 14); newer patches may omit Windows |
| WSL2 | Alternate only — see below |

---

## Nodes (rclpy)

| Launch name | Executable (`ros2 run`) | Class | Role |
|-------------|-------------------------|-------|------|
| `orbbec_cam0_pub` | `orbbec_camera_publisher` | `OrbbecCameraPublisher(Node)` | Orbbec index 0 → `sensor_msgs/Image` |
| `orbbec_cam1_pub` | `orbbec_camera_publisher` | `OrbbecCameraPublisher(Node)` | Orbbec index 1 → `sensor_msgs/Image` |
| `orbbec_sync_detect` | `orbbec_sync_detect` | `SyncDetectNode(Node)` | `ApproximateTimeSynchronizer` → `locate_standalone` |

Optional (no YOLO): `orbbec_sync_forward` → `SyncForwardNode(Node)`.

---

## Config (installed into share/)

After build (merge-install on Windows):

```text
install/share/orbbec_dual_sync/config/dual_sync.yaml
```

Source path:

```text
ros2_ws/src/orbbec_dual_sync/config/dual_sync.yaml
```

YAML keys: cam indexes, `slop_s`, `queue_size`, color/depth topics, `weak_camera`,
optional `model` (weights path), corroboration thresholds. **Preview:** `show_preview: false` on
both publishers (raw USB windows off); `show_preview: true` on `orbbec_sync_detect` in YAML,
overridden by launch arg `preview` (default `true`) — tkinter side-by-side annotated view; Q
closes preview only, Ctrl+C stops launch; `preview:=false` for headless.

---

## Topics (YAML defaults)

| Topic | Type |
|--------|------|
| `/orbbec/cam0/color/image_raw` | `sensor_msgs/Image` `bgr8` |
| `/orbbec/cam0/depth/image_raw` | `sensor_msgs/Image` `32FC1` (mm) |
| `/orbbec/cam1/color/image_raw` | same |
| `/orbbec/cam1/depth/image_raw` | same |
| `/orbbec/sync_detect/stats` | `std_msgs/String` |

No `cv_bridge` — encoding is manual in `image_codec.py`.

---

## Package layout

```
Camera/ros2_ws/
  build.cmd / source_install.cmd / launch_dual_sync.cmd / ros2_flow.cmd / _ros_env.cmd
  install_windows_ros2.ps1 / source_install.ps1 / build.ps1 / run_dual_sync.ps1 / _ros_env.ps1
  pixi.toml                 # RoboStack Humble (Windows)
  build.sh                  # Linux / WSL only
  verify_layout.py
  src/orbbec_dual_sync/     # ament_python
```

Static check (no ROS needed):

```powershell
cd E:\Desktop\Jessica\Camera\ros2_ws
python verify_layout.py
```

---

## Alternate — Linux / WSL2

Only if you want apt ROS inside bash. Do **not** paste these into PowerShell.

```bash
# Inside WSL or native Linux
cd /mnt/e/Desktop/Jessica/Camera/ros2_ws   # or your Linux path
source /opt/ros/$ROS_DISTRO/setup.bash
./build.sh
source install/setup.bash
export ORBBEC_CAMERA_ROOT=/mnt/e/Desktop/Jessica/Camera
ros2 launch orbbec_dual_sync dual_sync_detect.launch.py
```

From PowerShell, enter WSL first:

```powershell
wsl
# or:  .\ros2_ws\enter_wsl.ps1
```

USB cameras in WSL need `usbipd` attach. Prefer native Windows for Orbbec USB.

---

## Manual node start (after build + overlay)

```powershell
# Windows — after .\source_install.ps1 (or .\run_dual_sync.ps1 env)
ros2 run orbbec_dual_sync orbbec_camera_publisher --ros-args `
  --params-file E:\Desktop\Jessica\Camera\ros2_ws\install\share\orbbec_dual_sync\config\dual_sync.yaml `
  -r __node:=orbbec_cam0_pub
```

---

## NOT ROS (optional debug only)

`Camera/local_approx_sync.py` is a plain Python script. It does **not** use
`rclpy`, is **not** installed by colcon, and is **not** the supported path.
