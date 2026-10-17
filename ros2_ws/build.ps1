# Native Windows build helper (RoboStack / Pixi). Primary path for this workspace.
# PowerShell cannot run Linux: source, ./build.sh, export, or /mnt/e/... paths.
#
# If ExecutionPolicy blocks .ps1, prefer CMD (no policy needed):
#   .\build.cmd
# Or Bypass once:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1
param(
    [switch]$Native,   # Explicit native build (default behavior when no -WslHelp)
    [switch]$WslHelp   # Print WSL/bash alternate instructions and exit
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$WslWs = '/mnt/e/Desktop/Jessica/Camera/ros2_ws'
$WslCam = '/mnt/e/Desktop/Jessica/Camera'

function Show-WslBuildInstructions {
    Write-Host ''
    Write-Host '============================================================' -ForegroundColor Yellow
    Write-Host '  Alternate: WSL bash (not required for native Windows)' -ForegroundColor Yellow
    Write-Host '============================================================' -ForegroundColor Yellow
    Write-Host ''
    Write-Host '  PowerShell cannot source Linux ROS. Use native path instead:'
    Write-Host '    .\build.ps1'
    Write-Host '    .\run_dual_sync.ps1'
    Write-Host ''
    Write-Host '  Or enter WSL, then bash:'
    Write-Host '       wsl'
    Write-Host '     or:  .\enter_wsl.ps1'
    Write-Host ''
    Write-Host '  Inside WSL (bash):'
    Write-Host ("       cd {0}" -f $WslWs)
    Write-Host '       source /opt/ros/$ROS_DISTRO/setup.bash'
    Write-Host '       ./build.sh'
    Write-Host '       source install/setup.bash'
    Write-Host ("       export ORBBEC_CAMERA_ROOT={0}" -f $WslCam)
    Write-Host '       ros2 launch orbbec_dual_sync dual_sync_detect.launch.py'
    Write-Host ''
    Write-Host '  Docs: ..\ROS_SYNC.md'
    Write-Host '============================================================' -ForegroundColor Yellow
    Write-Host ''
}

if ($WslHelp -and -not $Native) {
    Show-WslBuildInstructions
    exit 0
}

# --- Native Windows path (RoboStack / Pixi / official binary) ---

# Refresh PATH for winget-installed pixi
$pixiBin = Join-Path $env:LOCALAPPDATA 'pixi\bin'
if (Test-Path -LiteralPath $pixiBin) {
    $env:Path = "$pixiBin;$env:Path"
}

. (Join-Path $Root '_ros_env.ps1')
$script:RosWsRoot = $Root

function Fail([string]$Message, [switch]$InstallHelp) {
    Write-Host ''
    Write-Host ("ERROR: {0}" -f $Message) -ForegroundColor Red
    if ($InstallHelp) { Show-WindowsRosInstallHelp }
    Write-Host 'Install RoboStack first:  .\install_windows_ros2.ps1'
    Write-Host 'For WSL alternate:        .\build.ps1 -WslHelp'
    Write-Host 'local_approx_sync.py is NOT ROS (optional debug only).'
    exit 1
}

if (-not (Enter-Ros2Base)) {
    Fail 'ROS2 not found. Run .\install_windows_ros2.ps1 first.' -InstallHelp
}

if (-not (Test-ColconAvailable)) {
    Fail 'colcon not found after ROS activation. Re-run .\install_windows_ros2.ps1' -InstallHelp
}

if (-not (Test-Path "$Root\src\orbbec_dual_sync\package.xml")) {
    Fail 'Missing package at src/orbbec_dual_sync'
}
if (-not (Test-Path "$Root\src\orbbec_dual_sync\resource\orbbec_dual_sync")) {
    Fail 'Missing ament resource index marker resource/orbbec_dual_sync'
}

Set-OrbbecCameraRoot

Write-Host 'Building orbbec_dual_sync with colcon (Windows: --merge-install, no symlink)...'
# --merge-install avoids PATH length issues on Windows.
# Do NOT use --symlink-install unless Developer Mode (or admin) is enabled;
# otherwise WinError 1314 (privilege for creating symlinks).
& colcon build --packages-select orbbec_dual_sync --merge-install
if ($LASTEXITCODE -ne 0) {
    Fail ("colcon build failed (exit {0}). If you saw WinError 1314, avoid --symlink-install or enable Developer Mode." -f $LASTEXITCODE)
}

Write-Host ''
Write-Host 'Build OK - ROS2 Nodes under install\.'
Write-Host 'Recommended one-shot (sources install + ORBBEC_CAMERA_ROOT + launch):'
Write-Host '  .\launch_dual_sync.cmd'
Write-Host '  .\run_dual_sync.ps1'
Write-Host ''
Write-Host 'Manual env + launch (this session only):'
Write-Host '  .\source_install.ps1   # or .\source_install.cmd'
Write-Host ("  `$env:ORBBEC_CAMERA_ROOT = `"{0}`"" -f $env:ORBBEC_CAMERA_ROOT)
Write-Host '  ros2 launch orbbec_dual_sync dual_sync_detect.launch.py'
Write-Host 'Note: never run source_install.cmd without .\ in PowerShell.'
Write-Host ''
Write-Host 'Config: install\share\orbbec_dual_sync\config\dual_sync.yaml  (merge-install layout)'
Write-Host 'Do NOT use python local_approx_sync.py - that is NOT ROS.'
