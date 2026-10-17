@echo off
REM Shared helper: activate RoboStack/Pixi ROS2 into THIS cmd session.
REM Call from other .cmd files:  call "%~dp0_ros_env.cmd"
REM Does not require PowerShell ExecutionPolicy.

set "ROS_WS_ROOT=%~dp0"
REM strip trailing backslash for cleaner paths
if "%ROS_WS_ROOT:~-1%"=="\" set "ROS_WS_ROOT=%ROS_WS_ROOT:~0,-1%"
cd /d "%ROS_WS_ROOT%"

if not defined ORBBEC_CAMERA_ROOT set "ORBBEC_CAMERA_ROOT=%ROS_WS_ROOT%\..\robotics"
for %%I in ("%ORBBEC_CAMERA_ROOT%") do set "ORBBEC_CAMERA_ROOT=%%~fI"
if not exist "%ORBBEC_CAMERA_ROOT%\depthcamera.py" (
  echo ERROR: Robotics root not found at "%ORBBEC_CAMERA_ROOT%"
  echo Set ORBBEC_CAMERA_ROOT to the folder containing depthcamera.py.
  exit /b 1
)
echo ORBBEC_CAMERA_ROOT=%ORBBEC_CAMERA_ROOT%

REM Anaconda/Miniconda DLLs on PATH break RoboStack rclpy ("DLL load failed").
REM Scrub those entries before Pixi activate (common when PowerShell shows (base)).
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "$parts = $env:Path -split ';' | Where-Object { $_ -and ($_ -notmatch '(?i)anaconda3|miniconda3|[\\/]anaconda|[\\/]miniconda') }; ($parts | Select-Object -Unique) -join ';'"`) do set "PATH=%%P"
if defined CONDA_PREFIX (
  echo %CONDA_PREFIX% | findstr /I /C:"anaconda" /C:"miniconda" >nul
  if not errorlevel 1 (
    set "CONDA_PREFIX="
    set "CONDA_DEFAULT_ENV="
    set "CONDA_PROMPT_MODIFIER="
    set "CONDA_PYTHON_EXE="
    set "CONDA_SHLVL="
  )
)

set "PATH=%LOCALAPPDATA%\pixi\bin;%PATH%"

where pixi >nul 2>&1
if errorlevel 1 (
  echo ERROR: pixi not on PATH. Install: winget install prefix-dev.pixi
  echo Then run: powershell -NoProfile -ExecutionPolicy Bypass -File "%ROS_WS_ROOT%\install_windows_ros2.ps1"
  echo If you just installed pixi, close and reopen this terminal first.
  exit /b 1
)

if not exist "%ROS_WS_ROOT%\.pixi\envs\default\" (
  echo ERROR: Pixi env missing under .pixi\envs\default
  echo Run one-time install:
  echo   powershell -NoProfile -ExecutionPolicy Bypass -File "%ROS_WS_ROOT%\install_windows_ros2.ps1"
  exit /b 1
)

set "PIXI_HOOK=%TEMP%\orbbec_pixi_hook_%RANDOM%%RANDOM%.bat"
pixi shell-hook --manifest-path "%ROS_WS_ROOT%\pixi.toml" -s cmd > "%PIXI_HOOK%"
if errorlevel 1 (
  echo ERROR: pixi shell-hook failed
  if exist "%PIXI_HOOK%" del "%PIXI_HOOK%" >nul 2>&1
  exit /b 1
)
call "%PIXI_HOOK%"
del "%PIXI_HOOK%" >nul 2>&1

where colcon >nul 2>&1
if errorlevel 1 (
  where ros2 >nul 2>&1
  if errorlevel 1 (
    echo ERROR: colcon/ros2 not found after Pixi activate.
    echo Re-run: powershell -NoProfile -ExecutionPolicy Bypass -File "%ROS_WS_ROOT%\install_windows_ros2.ps1"
    exit /b 1
  )
)

exit /b 0
