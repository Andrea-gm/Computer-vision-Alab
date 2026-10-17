# Install RoboStack ROS 2 Humble for native Windows via Pixi (recommended path).
# Does NOT use WSL. Requires network. Prefer VS 2022 C++ for C++ packages;
# this ament_python workspace can build without MSVC.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# Ensure pixi on PATH (winget installs under %LOCALAPPDATA%\pixi\bin)
$pixiBin = Join-Path $env:LOCALAPPDATA "pixi\bin"
if (Test-Path -LiteralPath $pixiBin) {
    $env:Path = "$pixiBin;$env:Path"
}

function Ensure-Pixi {
    if (Get-Command pixi -ErrorAction SilentlyContinue) { return }
    Write-Host "pixi not found - installing with winget..."
    winget install --id prefix-dev.pixi -e --accept-package-agreements --accept-source-agreements
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
    if (-not (Get-Command pixi -ErrorAction SilentlyContinue)) {
        if (Test-Path -LiteralPath (Join-Path $pixiBin "pixi.exe")) {
            $env:Path = "$pixiBin;$env:Path"
        }
    }
    if (-not (Get-Command pixi -ErrorAction SilentlyContinue)) {
        throw "pixi still not on PATH. Close and reopen PowerShell, then re-run this script."
    }
}

Ensure-Pixi
Write-Host "pixi: $(pixi --version)"

if (-not (Test-Path -LiteralPath (Join-Path $Root "pixi.toml"))) {
    throw "Missing pixi.toml in $Root"
}

Write-Host ""
Write-Host "Installing RoboStack Humble packages (several GB; may take a while)..."
Write-Host "Channels: robostack-humble + conda-forge"
Write-Host ""

& pixi install --manifest-path (Join-Path $Root "pixi.toml")
if ($LASTEXITCODE -ne 0) {
    throw "pixi install failed (exit $LASTEXITCODE). Check network / disk space / VS Build Tools notes in ROS_SYNC.md."
}

Write-Host ""
Write-Host "Installing ultralytics into the same pixi env (sync_detect / YOLO)..."
& pixi run --manifest-path (Join-Path $Root "pixi.toml") pip install ultralytics
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: ultralytics pip install failed. Retry later with:" -ForegroundColor Yellow
    Write-Host "  pixi run pip install ultralytics"
} else {
    Write-Host "ultralytics OK"
}

# Probe ros2
. (Join-Path $Root "_ros_env.ps1")
$script:RosWsRoot = $Root
if (Enter-Ros2Base) {
    Write-Host "ros2 CLI: $(Get-Command ros2 | Select-Object -ExpandProperty Source)"
    if (Test-ColconAvailable) {
        Write-Host "colcon OK: $(Get-Command colcon | Select-Object -ExpandProperty Source)"
    }
    python -c "import cv2; print('opencv', cv2.__version__, 'CAP_OBSENSOR', hasattr(cv2,'CAP_OBSENSOR'))"
} else {
    Write-Host "Install finished but ros2 not yet visible in this shell." -ForegroundColor Yellow
    Write-Host "Try:  pixi shell   then   ros2 --help"
}

Write-Host ""
Write-Host "Next (PowerShell only):"
Write-Host "  cd $Root"
Write-Host "  .\build.ps1"
Write-Host "  .\run_dual_sync.ps1"
Write-Host ""
Write-Host "Manual (cmd.exe after pixi shell):"
Write-Host "  call install\setup.bat"
Write-Host "  set ORBBEC_CAMERA_ROOT=E:\Desktop\Jessica\Camera"
Write-Host "  ros2 launch orbbec_dual_sync dual_sync_detect.launch.py"
