@echo off
REM Windows equivalent of: source install/setup.bash
REM Activates Pixi underlay + call install\setup.bat into THIS cmd session.
REM Usage (cmd):  source_install.cmd
REM Usage (PowerShell):  .\source_install.cmd   (.\ is required)
REM One-shot launch (no separate source):  launch_dual_sync.cmd  /  .\launch_dual_sync.cmd

call "%~dp0_ros_env.cmd"
if errorlevel 1 exit /b 1

if not exist "%ROS_WS_ROOT%\install\setup.bat" (
  if not exist "%ROS_WS_ROOT%\install\local_setup.bat" (
    echo ERROR: install\setup.bat missing. Run build.cmd first.
    exit /b 1
  )
)

if exist "%ROS_WS_ROOT%\install\setup.bat" (
  echo Sourcing: %ROS_WS_ROOT%\install\setup.bat
  call "%ROS_WS_ROOT%\install\setup.bat"
) else (
  echo Sourcing: %ROS_WS_ROOT%\install\local_setup.bat
  call "%ROS_WS_ROOT%\install\local_setup.bat"
)

where ros2 >nul 2>&1
if errorlevel 1 (
  echo ERROR: ros2 still not on PATH after sourcing install.
  exit /b 1
)

echo Install overlay active (like source install/setup.bash).
echo ORBBEC_CAMERA_ROOT=%ORBBEC_CAMERA_ROOT%
echo.
echo Recommended one-shot (env + launch in one step):
echo   launch_dual_sync.cmd          ^(cmd^)
echo   .\launch_dual_sync.cmd        ^(PowerShell^)
echo.
echo Or in THIS same window after sourcing:
echo   ros2 launch orbbec_dual_sync dual_sync_detect.launch.py
echo Publishers log preview=off ^(raw USB^); sync_detect shows the annotated preview ^(default ON^).
echo.
echo SEE cameras ^(ros2 topic list only prints names — not images^):
echo   With ROS launch running:  view_dual_cameras.cmd
echo   Without ROS ^(direct USB^): ..\robotics\view_dual_cameras.cmd
exit /b 0
