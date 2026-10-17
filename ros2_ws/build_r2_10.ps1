# Native Windows build for r2_10_fusion_multiorient (RoboStack / Pixi).

# If ExecutionPolicy blocks .ps1, use:  .\build_r2_10.cmd

# Or Bypass:  powershell -NoProfile -ExecutionPolicy Bypass -File .\build_r2_10.ps1



$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

Set-Location $Root



$pixiBin = Join-Path $env:LOCALAPPDATA "pixi\bin"

if (Test-Path -LiteralPath $pixiBin) {

    $env:Path = "$pixiBin;$env:Path"

}



. (Join-Path $Root "_ros_env.ps1")

$script:RosWsRoot = $Root



function Fail([string]$Message, [switch]$InstallHelp) {

    Write-Host ""

    Write-Host ("ERROR: {0}" -f $Message) -ForegroundColor Red

    if ($InstallHelp) { Show-WindowsRosInstallHelp }

    Write-Host "Install RoboStack first:  .\install_windows_ros2.ps1"

    exit 1

}



if (-not (Enter-Ros2Base)) {

    Fail "ROS2 not found. Run .\install_windows_ros2.ps1 first." -InstallHelp

}



if (-not (Test-ColconAvailable)) {

    Fail "colcon not found after ROS activation. Re-run .\install_windows_ros2.ps1" -InstallHelp

}



if (-not (Test-Path "$Root\src\r2_10_fusion_multiorient\package.xml")) {

    Fail "Missing package at src/r2_10_fusion_multiorient"

}



Set-OrbbecCameraRoot

if (-not (Wait-OrbbecPublishersStopped -TimeoutSeconds 5)) {
    Write-Host "Build stopped before replacing an in-use publisher executable." -ForegroundColor Yellow
    exit 1
}



Write-Host "Building camera publisher + r2_10 with colcon (--merge-install)..."

& colcon build --packages-select orbbec_dual_sync r2_10_fusion_multiorient --merge-install

if ($LASTEXITCODE -ne 0) {

    Fail ("colcon build failed (exit {0})." -f $LASTEXITCODE)

}



Write-Host ""

Write-Host "Build OK."

Write-Host "One-shot launch:"

Write-Host "  .\launch_r2_10.cmd"

Write-Host "  .\launch_r2_10.ps1"

Write-Host "Live camera:"

Write-Host "  .\launch_r2_10.ps1 run_offline:=false"

Write-Host "Note: never run bare colcon/ros2 without .\build_r2_10.cmd or Enter-Ros2Base."

