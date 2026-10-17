@echo off
REM One-shot: activate Pixi → colcon build → source install → launch dual sync.
REM Usage:  ros2_flow.cmd
REM Extra args are passed to ros2 launch.

call "%~dp0_ros_env.cmd"
if errorlevel 1 exit /b 1

echo === colcon build ===
colcon build --packages-select orbbec_dual_sync --merge-install
if errorlevel 1 (
  echo ERROR: colcon build failed
  exit /b 1
)

echo === source install ===
if exist "%ROS_WS_ROOT%\install\setup.bat" (
  call "%ROS_WS_ROOT%\install\setup.bat"
) else (
  call "%ROS_WS_ROOT%\install\local_setup.bat"
)

where ros2 >nul 2>&1
if errorlevel 1 (
  echo ERROR: ros2 not on PATH after install setup
  exit /b 1
)

echo === ros2 launch ===
ros2 launch orbbec_dual_sync dual_sync_detect.launch.py %*
exit /b %ERRORLEVEL%
