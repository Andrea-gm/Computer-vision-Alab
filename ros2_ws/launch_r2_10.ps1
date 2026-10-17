# One-shot: Pixi + install overlay + ORBBEC_CAMERA_ROOT + ros2 launch (r2_10 vial fusion).

# If ExecutionPolicy blocks .ps1, use:  .\launch_r2_10.cmd

# Or Bypass:  powershell -NoProfile -ExecutionPolicy Bypass -File .\launch_r2_10.ps1

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

    Write-Host "ERROR: Workspace overlay missing. Run .\build_r2_10.cmd first." -ForegroundColor Red

    exit 1

}



Set-OrbbecCameraRoot



if (-not (Test-Ros2Available)) {

    Write-Host "ERROR: ros2 still not on PATH after setup." -ForegroundColor Red

    exit 1

}

$LiveCamera = $false
foreach ($arg in $LaunchArgs) {
    if ($arg -match '^run_offline:=false$') { $LiveCamera = $true }
}
if ($LiveCamera -and -not (Wait-OrbbecPublishersStopped -TimeoutSeconds 10)) {
    exit 1
}



Write-Host ""

Write-Host "=== r2_10_fusion_multiorient one-shot launch ===" -ForegroundColor Cyan

Write-Host "Env sourced (install overlay + ORBBEC_CAMERA_ROOT). No separate .\source_install.ps1 needed."

Write-Host "Default launch: offline dataset (run_offline:=true). Live camera: run_offline:=false"

Write-Host ""



Write-Host "Launching: ros2 launch r2_10_fusion_multiorient vial_detect.launch.py"

if ($LaunchArgs -and $LaunchArgs.Count -gt 0) {

    & ros2 launch r2_10_fusion_multiorient vial_detect.launch.py @LaunchArgs

} else {

    & ros2 launch r2_10_fusion_multiorient vial_detect.launch.py

}

if ($null -eq $LASTEXITCODE) { exit 0 }

exit $LASTEXITCODE

