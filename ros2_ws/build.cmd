@echo off
REM colcon build for orbbec_dual_sync — works under Restricted ExecutionPolicy.
REM Usage:  build.cmd
REM From PowerShell you can also:  cmd /c build.cmd

call "%~dp0_ros_env.cmd"
if errorlevel 1 exit /b 1

if not exist "%ROS_WS_ROOT%\src\orbbec_dual_sync\package.xml" (
  echo ERROR: Missing package at src\orbbec_dual_sync
  exit /b 1
)

echo Building orbbec_dual_sync with colcon (--merge-install)...
colcon build --packages-select orbbec_dual_sync --merge-install
if errorlevel 1 (
  echo ERROR: colcon build failed
  exit /b 1
)

echo.
echo Build OK.
echo.
echo Recommended one-shot launch (sources install + ORBBEC_CAMERA_ROOT for you):
echo   CMD:         launch_dual_sync.cmd
echo   PowerShell:  .\launch_dual_sync.cmd
echo                .\run_dual_sync.ps1
echo.
echo Manual env + launch (same cmd window must stay open):
echo   CMD:         source_install.cmd
echo                set ORBBEC_CAMERA_ROOT=%ORBBEC_CAMERA_ROOT%
echo                ros2 launch orbbec_dual_sync dual_sync_detect.launch.py
echo   PowerShell:  .\source_install.cmd
echo                .\source_install.ps1
echo                $env:ORBBEC_CAMERA_ROOT = "%ORBBEC_CAMERA_ROOT%"
echo                ros2 launch orbbec_dual_sync dual_sync_detect.launch.py
echo.
echo Note: In PowerShell, prefix .\ for scripts in this folder (source_install.cmd alone fails).
echo Docs: ..\robotics\ROS_SYNC.md
exit /b 0
