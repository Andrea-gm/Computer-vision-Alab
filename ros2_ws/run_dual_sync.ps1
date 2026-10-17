# One-shot: Pixi + install overlay + ORBBEC_CAMERA_ROOT + ros2 launch (no source_install needed).
# Uses install\setup.bat via env import - NOT source setup.bash.
#
# If ExecutionPolicy blocks .ps1, use CMD instead:
#   .\launch_dual_sync.cmd
# Or Bypass:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\run_dual_sync.ps1
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$LaunchArgs
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$pixiBin = Join-Path $env:LOCALAPPDATA "pixi\bin"
if (Test-Path -LiteralPath $pixiBin) {
    $env:Path = "$pixiBin;$env:Path"
}

. (Join-Path $Root "_ros_env.ps1")
$script:RosWsRoot = $Root

if (-not (Enter-Ros2Base)) {
    Write-Host "ERROR: ROS2 not available." -ForegroundColor Red
    Show-WindowsRosInstallHelp
    exit 1
}

if (-not (Enter-WorkspaceOverlay)) {
    Write-Host "ERROR: Workspace overlay missing. Run .\build.ps1 first." -ForegroundColor Red
    exit 1
}

Set-OrbbecCameraRoot

if (-not (Test-Ros2Available)) {
    Write-Host "ERROR: ros2 still not on PATH after setup." -ForegroundColor Red
    exit 1
}

if (-not (Wait-OrbbecPublishersStopped -TimeoutSeconds 10)) {
    exit 1
}

Write-Host ""
Write-Host "=== orbbec_dual_sync one-shot launch ===" -ForegroundColor Cyan
Write-Host "Env already sourced (install overlay + ORBBEC_CAMERA_ROOT). No .\source_install.ps1 needed."
Write-Host "Stable default: local cap/edge/I_hard_robust detection. Optional: use_yolo:=true."
Write-Host "Startup timing: cam0 now, cam1 at 20s, sync_detect at 35s (~15s after cam1)."
Write-Host "Seeing only two publisher process lines before 35s is normal; do not relaunch."
Write-Host "After 35s expect orbbec_sync_detect to start, then 'detection worker ready'."
Write-Host "Headless: add preview:=false to the launch args."
Write-Host ""

Write-Host "Launching: ros2 launch orbbec_dual_sync dual_sync_detect.launch.py"
Write-Host "(Nodes: orbbec_cam0_pub, orbbec_cam1_pub, orbbec_sync_detect)"
Write-Host "OpenCV CAP_OBSENSOR in this env: use pixi Python (already activated)."
if ($LaunchArgs -and $LaunchArgs.Count -gt 0) {
    & ros2 launch orbbec_dual_sync dual_sync_detect.launch.py @LaunchArgs
} else {
    & ros2 launch orbbec_dual_sync dual_sync_detect.launch.py
}
if ($null -eq $LASTEXITCODE) { exit 0 }
exit $LASTEXITCODE
