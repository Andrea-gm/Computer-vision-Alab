@echo off
REM colcon build for r2_10_fusion_multiorient — works under Restricted ExecutionPolicy.
REM Usage (cmd):  build_r2_10.cmd
REM Usage (PowerShell):  .\build_r2_10.cmd

call "%~dp0_ros_env.cmd"
if errorlevel 1 exit /b 1

if not exist "%ROS_WS_ROOT%\src\r2_10_fusion_multiorient\package.xml" (
  echo ERROR: Missing package at src\r2_10_fusion_multiorient
  exit /b 1
)

echo Building r2_10_fusion_multiorient with colcon (--merge-install)...
colcon build --packages-select r2_10_fusion_multiorient --merge-install
if errorlevel 1 (
  echo ERROR: colcon build failed
  exit /b 1
)

echo.
echo Build OK.
echo.
echo Recommended one-shot launch (Pixi + install overlay + ros2 launch):
echo   CMD:         launch_r2_10.cmd
echo   PowerShell:  .\launch_r2_10.cmd
echo                .\launch_r2_10.ps1
echo.
echo Offline dataset replay (default in params.yaml run_offline:=true):
echo   .\launch_r2_10.cmd
echo Live camera topics:
echo   .\launch_r2_10.cmd run_offline:=false
echo.
echo Manual env + launch (same cmd window must stay open):
echo   CMD:         source_install.cmd
echo                ros2 launch r2_10_fusion_multiorient vial_detect.launch.py run_offline:=false
echo   PowerShell:  .\source_install.ps1
echo                ros2 launch r2_10_fusion_multiorient vial_detect.launch.py run_offline:=false
echo.
echo Note: In PowerShell, prefix .\ for scripts in this folder. Bare colcon/ros2 fails without Pixi.
exit /b 0
