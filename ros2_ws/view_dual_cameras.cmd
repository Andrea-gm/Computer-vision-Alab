@echo off
REM Live dual preview via ROS topics (cam0 | cam1). Does NOT open USB.
REM Requires publishers: start launch_ros_dual_sync.cmd in another terminal first.
REM ros2 topic list only shows names - this window shows the images.

echo.
echo ============================================================
echo   DUAL PREVIEW - ROS topics (cam0 ^| cam1 + rack CV)
echo ============================================================
echo   WHEN TO USE: ROS dual sync / publishers already running.
echo   Start first (other terminal):  launch_ros_dual_sync.cmd
echo   Then run this to SEE both cameras live.
echo.
echo   Topics:
echo     /orbbec/cam0/color/image_raw
echo     /orbbec/cam1/color/image_raw
echo.
echo   NOTE: "ros2 topic list" does NOT show video - use this instead.
echo   If ROS is STOPPED and you need USB preview:
echo     ..\robotics\view_dual_cameras.cmd
echo.
echo   Q / Esc / close window quits this viewer only (not the launch).
echo ============================================================
echo.

call "%~dp0_ros_env.cmd"
if errorlevel 1 exit /b 1

if not exist "%ROS_WS_ROOT%\install\setup.bat" (
  if not exist "%ROS_WS_ROOT%\install\local_setup.bat" (
    echo ERROR: Workspace overlay missing. Run build.cmd first.
    exit /b 1
  )
)

if exist "%ROS_WS_ROOT%\install\setup.bat" (
  call "%ROS_WS_ROOT%\install\setup.bat"
) else (
  call "%ROS_WS_ROOT%\install\local_setup.bat"
)

where ros2 >nul 2>&1
if errorlevel 1 (
  echo ERROR: ros2 not on PATH after setup.
  exit /b 1
)

REM Prefer installed entry point; fall back to src so this works before rebuild.
set "SRC_MOD=%ROS_WS_ROOT%\src\orbbec_dual_sync"
REM robotics must be on PYTHONPATH for furnace/rack/arm helpers (arm_agents).
set "PYTHONPATH=%ORBBEC_CAMERA_ROOT%;%SRC_MOD%;%PYTHONPATH%"
set "KMP_DUPLICATE_LIB_OK=TRUE"

REM Explicit arm YOLO so the robot box shows on both panes.
set "DUAL_VIEW_ARGS=--ros-args -p enable_furnace_cv:=true -p enable_rack_cv:=true -p enable_arm_cv:=true -p enable_blue_cv:=true -p blue_show_mask:=true -p furnace_match_live:=true -p state_mode:=classify -p arm_conf:=0.25 -p arm_every_n:=3"


ros2 pkg executables orbbec_dual_sync 2>nul | findstr /I /C:"orbbec_dual_view" >nul
if not errorlevel 1 (
  echo Running: ros2 run orbbec_dual_sync orbbec_dual_view  [furnace + rack + arm YOLO]
  ros2 run orbbec_dual_sync orbbec_dual_view %DUAL_VIEW_ARGS% %*
  exit /b %ERRORLEVEL%
)

echo orbbec_dual_view not in install yet - running from src via python -m
where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: python not found. Rebuild: build.cmd  then re-run this.
  exit /b 1
)
python -m orbbec_dual_sync.dual_view_node %DUAL_VIEW_ARGS% %*
exit /b %ERRORLEVEL%
