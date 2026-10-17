# Windows equivalent of:  source install/setup.bash
# Calls install\setup.bat and imports env vars into THIS PowerShell session.
#
# Usage (from ros2_ws, after Pixi/RoboStack is active and colcon build succeeded):
#   .\source_install.ps1
#
# If ExecutionPolicy blocks .ps1, use CMD (env stays in that cmd window):
#   .\source_install.cmd
# Or Bypass into this session:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\source_install.ps1
#
# Env changes ($env:*) persist after this script exits (process-level).
# Prefer this over `.\install\setup.ps1` on RoboStack: the underlay ships
# Library\local_setup.bat but often NOT local_setup.ps1.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$pixiBin = Join-Path $env:LOCALAPPDATA "pixi\bin"
if (Test-Path -LiteralPath $pixiBin) {
    $env:Path = "$pixiBin;$env:Path"
}

. (Join-Path $Root "_ros_env.ps1")
$script:RosWsRoot = $Root

# Underlay first so ros2/colcon exist; setup.bat also chains Library\local_setup.bat
# but Pixi PATH/CONDA hooks are still needed for the CLI wrappers.
if (-not (Test-Ros2Available)) {
    if (-not (Enter-Ros2Base)) {
        Write-Host "ERROR: ROS2 underlay not available. Activate Pixi first, e.g.:" -ForegroundColor Red
        Write-Host "  .\install_windows_ros2.ps1   # one-time"
        Write-Host "  # then in a shell:"
        Write-Host "  #   ensure pixi on PATH, or run commands after Enter-Ros2Base via build/run scripts"
        Show-WindowsRosInstallHelp
        exit 1
    }
}

$setupBat = Join-Path $Root "install\setup.bat"
$localBat = Join-Path $Root "install\local_setup.bat"
if (-not (Test-Path -LiteralPath $setupBat) -and -not (Test-Path -LiteralPath $localBat)) {
    Write-Host "ERROR: install\setup.bat missing. Run colcon build first:" -ForegroundColor Red
    Write-Host "  colcon build --packages-select orbbec_dual_sync --merge-install"
    exit 1
}

# Mimic `source install/setup.bash`: call setup.bat and import `set` into this session.
$batToCall = if (Test-Path -LiteralPath $setupBat) { $setupBat } else { $localBat }
Write-Host "Sourcing (Windows): $batToCall"
Invoke-BatEnv -BatPath $batToCall

# Ensure overlay markers are visible even if bat only partially applied.
if (-not (Enter-WorkspaceOverlay)) {
    Write-Host "WARNING: workspace overlay helpers did not re-apply; relying on bat env import." -ForegroundColor Yellow
}

if (-not (Test-Ros2Available)) {
    Write-Host "ERROR: ros2 still not on PATH after sourcing install." -ForegroundColor Red
    exit 1
}

# Quick package visibility check (non-fatal if listing is slow/noisy).
$pkgOk = $false
try {
    $pkgs = & ros2 pkg list 2>$null
    if ($pkgs -match 'orbbec_dual_sync') { $pkgOk = $true }
} catch { }

Set-OrbbecCameraRoot

Write-Host "Install overlay active (like source install/setup.bash)."
if ($pkgOk) {
    Write-Host "ros2 pkg list: orbbec_dual_sync OK"
} else {
    Write-Host "ros2 on PATH; if launch fails, re-run colcon build --merge-install."
}
Write-Host ""
Write-Host "Recommended one-shot (env + launch):"
Write-Host "  .\launch_dual_sync.cmd"
Write-Host "  .\run_dual_sync.ps1"
Write-Host ""
Write-Host "Or in this same PowerShell session:"
Write-Host ("  ros2 launch orbbec_dual_sync dual_sync_detect.launch.py")
Write-Host "Publishers log preview=off (raw USB); sync_detect shows annotated preview (launch default preview:=true)."
Write-Host ""
Write-Host "SEE the cameras (ros2 topic list only prints names — not images):"
Write-Host "  With ROS launch running:  .\view_dual_cameras.cmd"
Write-Host "  Without ROS (direct USB): ..\robotics\view_dual_cameras.cmd"
