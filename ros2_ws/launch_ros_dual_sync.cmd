@echo off
REM ROS dual camera publishers + cam1 rack overlay (vial YOLO off).
REM Overlay SUBSCRIBES to /orbbec/cam1/color/image_raw — does not steal USB.

echo.
echo ============================================================
echo   ROS dual sync — BOTH CAMERAS + rack overlay
echo ============================================================
echo   Publishers: cam0 then cam1 (staggered ~20s).
echo   Vial YOLO / sync_detect: OFF.
echo.
echo   VISIBLE WINDOWS (wait ~40s — do not relaunch early):
echo     1) orbbec_dual_view (cam0 ^| cam1+rack CV)  ^<-- BOTH + OPEN/CLOSED
echo     2) orbbec_rack_overlay                      ^<-- single cam1 CV
echo.
echo   Rack CV mode default: classify  (real OPEN/CLOSED)
echo   Pin: rack_state_mode:=force_closed  or  force_open
echo.
echo   Or view only (if launch already running), other terminal:
echo     .\view_dual_cameras.cmd
echo.
echo   USB-only dual view (ROS must be STOPPED):
echo     ..\robotics\view_dual_cameras.cmd
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

cmd /c "tasklist /FI \"IMAGENAME eq orbbec_camera_publisher.exe\" 2>nul | findstr /I orbbec_camera_publisher >nul"
if not errorlevel 1 (
  echo ERROR: ROS dual_sync is already running. Stop it before starting another.
  exit /b 1
)

echo Launching: ros2 launch orbbec_dual_sync dual_sync_detect.launch.py enable_detect:=false enable_rack_overlay:=true enable_dual_view:=true rack_state_mode:=classify %*
ros2 launch orbbec_dual_sync dual_sync_detect.launch.py enable_detect:=false enable_rack_overlay:=true enable_dual_view:=true rack_state_mode:=classify %*
exit /b %ERRORLEVEL%
