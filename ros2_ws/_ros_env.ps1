# Shared helper: locate and activate ROS2 + workspace overlay on Windows.
# Dot-source from build.ps1 / run_dual_sync.ps1 - do not run alone unless debugging.
# If ExecutionPolicy blocks .ps1, use _ros_env.cmd / build.cmd / launch_dual_sync.cmd instead.
$ErrorActionPreference = "Stop"

if (-not $script:RosWsRoot) {
    $script:RosWsRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
}

function Get-CameraRoot {
    if ($env:ORBBEC_CAMERA_ROOT -and (Test-Path -LiteralPath $env:ORBBEC_CAMERA_ROOT)) {
        return (Resolve-Path -LiteralPath $env:ORBBEC_CAMERA_ROOT).Path
    }
    $jessicaRoot = Split-Path -Parent $script:RosWsRoot
    $roboticsRoot = Join-Path $jessicaRoot "robotics"
    if (Test-Path -LiteralPath $roboticsRoot) {
        return (Resolve-Path -LiteralPath $roboticsRoot).Path
    }
    $legacyRoot = Join-Path $jessicaRoot "Camera"
    if (Test-Path -LiteralPath $legacyRoot) {
        return (Resolve-Path -LiteralPath $legacyRoot).Path
    }
    throw "Robotics root not found. Set ORBBEC_CAMERA_ROOT (expected: $roboticsRoot)."
}

function Test-Ros2Available {
    return [bool](Get-Command ros2 -ErrorAction SilentlyContinue)
}

function Test-ColconAvailable {
    return [bool](Get-Command colcon -ErrorAction SilentlyContinue)
}

function Invoke-BatEnv {
    param(
        [Parameter(Mandatory = $true)][string]$BatPath
    )
    if (-not (Test-Path -LiteralPath $BatPath)) {
        throw "Setup script not found: $BatPath"
    }
    $tmp = [System.IO.Path]::GetTempFileName()
    try {
        # Capture env after call; keep ERRORLEVEL from bat if set fails.
        cmd.exe /c "`"$BatPath`" && set > `"$tmp`""
        if (-not (Test-Path -LiteralPath $tmp)) {
            throw "Failed to capture environment from $BatPath"
        }
        Get-Content -LiteralPath $tmp | ForEach-Object {
            if ($_ -match '^([^=]+)=(.*)$') {
                $name = $Matches[1]
                $value = $Matches[2]
                # Skip pseudo-vars that break PowerShell
                if ($name -in @('PROMPT', '_')) { return }
                Set-Item -Path "Env:$name" -Value $value
            }
        }
    }
    finally {
        Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue
    }
}

function Find-OfficialRosSetup {
    $candidates = @(
        "C:\dev\ros2_humble\local_setup.bat",
        "C:\dev\ros2_humble\setup.bat",
        "C:\pixi_ws\ros2-windows\local_setup.bat",
        "C:\pixi_ws\ros2-windows\setup.bat",
        "C:\dev\ros2_jazzy\local_setup.bat",
        "C:\dev\ros2_jazzy\setup.bat"
    )
    foreach ($p in $candidates) {
        if (Test-Path -LiteralPath $p) { return $p }
    }
    return $null
}

function Clear-AnacondaFromPath {
    # Anaconda/Miniconda DLLs on PATH break RoboStack rclpy ("DLL load failed"),
    # especially when the shell was started with conda (base) activated.
    $env:Path = (
        $env:Path -split ';' |
            Where-Object { $_ -and ($_ -notmatch '(?i)anaconda3|miniconda3|[\\/]anaconda|[\\/]miniconda') } |
            Select-Object -Unique
    ) -join ';'
    if ($env:CONDA_PREFIX -and ($env:CONDA_PREFIX -match '(?i)anaconda|miniconda')) {
        Remove-Item Env:CONDA_PREFIX -ErrorAction SilentlyContinue
        Remove-Item Env:CONDA_DEFAULT_ENV -ErrorAction SilentlyContinue
        Remove-Item Env:CONDA_PROMPT_MODIFIER -ErrorAction SilentlyContinue
        Remove-Item Env:CONDA_PYTHON_EXE -ErrorAction SilentlyContinue
        Remove-Item Env:CONDA_SHLVL -ErrorAction SilentlyContinue
    }
}

function Enter-PixiRosEnv {
    param([string]$WsRoot = $script:RosWsRoot)

    Clear-AnacondaFromPath

    $pixiToml = Join-Path $WsRoot "pixi.toml"
    if (-not (Get-Command pixi -ErrorAction SilentlyContinue)) {
        $pixiCandidate = Join-Path $env:LOCALAPPDATA "pixi\bin\pixi.exe"
        if (Test-Path -LiteralPath $pixiCandidate) {
            $env:Path = "$(Split-Path $pixiCandidate -Parent);$env:Path"
        }
    }
    if (-not (Get-Command pixi -ErrorAction SilentlyContinue)) { return $false }
    if (-not (Test-Path -LiteralPath $pixiToml)) { return $false }

    $envDir = Join-Path $WsRoot ".pixi\envs\default"
    if (-not (Test-Path -LiteralPath $envDir)) {
        return $false
    }

    # Preferred: official PowerShell hook (PATH / CONDA_PREFIX).
    try {
        $hookText = & pixi shell-hook --manifest-path $pixiToml -s powershell 2>$null | Out-String
        if ($hookText -and $hookText.Trim().Length -gt 0) {
            Invoke-Expression $hookText
        }
    }
    catch { }

    if (-not $env:CONDA_PREFIX) {
        $env:CONDA_PREFIX = $envDir
    }
    $env:Path = "$envDir;$envDir\Library\bin;$envDir\Scripts;$envDir\bin;$env:Path"

    # RoboStack ROS activation. Prefer .bat: this win-64 env ships local_setup.bat
    # but not local_setup.ps1 (activate.ps1 would fail).
    $rosActivateBat = Join-Path $envDir "etc\conda\activate.d\ros-humble-ros-workspace_activate.bat"
    $libSetupBat = Join-Path $envDir "Library\local_setup.bat"
    if (Test-Path -LiteralPath $rosActivateBat) {
        Invoke-BatEnv -BatPath $rosActivateBat
    }
    elseif (Test-Path -LiteralPath $libSetupBat) {
        Invoke-BatEnv -BatPath $libSetupBat
        $env:ROS_DISTRO = "humble"
        $env:AMENT_PREFIX_PATH = Join-Path $envDir "Library"
        $env:AMENT_PYTHON_EXECUTABLE = Join-Path $envDir "python.exe"
        $env:ROS_OS_OVERRIDE = "conda:win64"
        $env:PYTHONHOME = ""
    }

    if (-not $env:ROS_DISTRO) { $env:ROS_DISTRO = "humble" }
    if (-not $env:AMENT_PREFIX_PATH) {
        $env:AMENT_PREFIX_PATH = Join-Path $envDir "Library"
    }

    return (Test-Ros2Available -or Test-ColconAvailable)
}

function Enter-CondaRosEnv {
    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $conda) { return $false }
    $envs = @("ros_env", "ros-humble", "humble", "ros2")
    foreach ($name in $envs) {
        $hook = & conda shell.powershell activate $name 2>$null
        if ($LASTEXITCODE -eq 0 -and $hook) {
            Invoke-Expression ($hook -join "`n")
            if (Test-Ros2Available -or Test-ColconAvailable) { return $true }
        }
    }
    return $false
}

function Enter-Ros2Base {
    <#
    .SYNOPSIS
      Activate any available Windows ROS2 (RoboStack pixi, conda, or official binary).
    #>
    if (Test-Ros2Available) { return $true }

    if (Enter-PixiRosEnv) { return $true }
    if (Enter-CondaRosEnv) { return $true }

    $official = Find-OfficialRosSetup
    if ($official) {
        Invoke-BatEnv -BatPath $official
        return (Test-Ros2Available)
    }
    return $false
}

function Enter-WorkspaceOverlay {
    param([string]$WsRoot = $script:RosWsRoot)

    $ps1 = Join-Path $WsRoot "install\local_setup.ps1"
    $bat = Join-Path $WsRoot "install\local_setup.bat"
    $setupBat = Join-Path $WsRoot "install\setup.bat"
    $setupPs1 = Join-Path $WsRoot "install\setup.ps1"

    if (Test-Path -LiteralPath $ps1) {
        . $ps1
        return $true
    }
    if (Test-Path -LiteralPath $setupPs1) {
        . $setupPs1
        return $true
    }
    if (Test-Path -LiteralPath $bat) {
        Invoke-BatEnv -BatPath $bat
        return $true
    }
    if (Test-Path -LiteralPath $setupBat) {
        Invoke-BatEnv -BatPath $setupBat
        return $true
    }
    return $false
}

function Set-OrbbecCameraRoot {
    param([string]$CameraRoot = (Get-CameraRoot))
    $env:ORBBEC_CAMERA_ROOT = $CameraRoot
    Write-Host "ORBBEC_CAMERA_ROOT=$env:ORBBEC_CAMERA_ROOT"
}

function Wait-OrbbecPublishersStopped {
    param([int]$TimeoutSeconds = 10)

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            $publishers = @(
                Get-CimInstance Win32_Process -ErrorAction Stop |
                    Where-Object { $_.CommandLine -match 'orbbec_camera_publisher' }
            )
        } catch {
            Write-Warning "Could not check for leftover Orbbec publishers: $($_.Exception.Message)"
            return $true
        }
        if ($publishers.Count -eq 0) { return $true }
        if ((Get-Date) -lt $deadline) {
            Write-Host "Waiting for $($publishers.Count) previous Orbbec publisher(s) to exit..."
            Start-Sleep -Seconds 1
        }
    } while ((Get-Date) -lt $deadline)

    $pids = ($publishers | ForEach-Object { $_.ProcessId }) -join ", "
    Write-Host ""
    Write-Host "ERROR: leftover Orbbec publisher process(es): PID $pids" -ForegroundColor Red
    Write-Host "Stop them before relaunch so they release CAP_OBSENSOR:"
    foreach ($publisher in $publishers) {
        Write-Host "  taskkill /F /PID $($publisher.ProcessId)"
    }
    Write-Host "If a clean stop still cannot stream, unplug/replug both cameras."
    return $false
}

function Show-WindowsRosInstallHelp {
    Write-Host ""
    Write-Host "ROS2 / colcon not found on this Windows shell." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Recommended (RoboStack via Pixi - native Windows, no WSL):"
    Write-Host "  cd $($script:RosWsRoot)"
    Write-Host "  # Prefer CMD if ExecutionPolicy is Restricted:"
    Write-Host "  build.cmd"
    Write-Host "  launch_dual_sync.cmd"
    Write-Host "  # Or Bypass PowerShell scripts:"
    Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File .\install_windows_ros2.ps1"
    Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1"
    Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File .\run_dual_sync.ps1"
    Write-Host "  # Or permanently: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned"
    Write-Host ""
    Write-Host "Official binary (Humble last Windows zip ~2026-02; newer patches dropped Win10):"
    Write-Host "  https://docs.ros.org/en/humble/Installation/Windows-Install-Binary.html"
    Write-Host "  https://github.com/ros2/ros2/releases  (ros2-humble-*-windows-release-amd64.zip)"
    Write-Host "  Then each cmd:  call C:\dev\ros2_humble\local_setup.bat"
    Write-Host ""
    Write-Host "Jazzy Windows (pixi deps + zip):"
    Write-Host "  https://docs.ros.org/en/jazzy/Installation/Windows-Install-Binary.html"
    Write-Host "  Note: recent Jazzy patch releases may omit Windows binaries (Win10 EOL)."
    Write-Host ""
    Write-Host "Blocker: C++ packages / some tooling need Visual Studio 2022 with C++."
    Write-Host "  https://visualstudio.microsoft.com/visual-cpp-build-tools/"
    Write-Host "  ament_python packages (this workspace) usually do NOT need MSVC."
    Write-Host ""
    Write-Host "WSL2 remains an alternate (see ROS_SYNC.md); not required for this path."
    Write-Host ""
}
