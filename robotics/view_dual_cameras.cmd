@echo off
REM Direct USB dual Orbbec preview (CAP_OBSENSOR). Conflicts with ROS publishers.
REM If launch_ros_dual_sync is running, use ros2_ws\view_dual_cameras.cmd instead.

echo.
echo ============================================================
echo   DUAL PREVIEW — direct USB (CAP_OBSENSOR)
echo ============================================================
echo   WHEN TO USE: ROS is STOPPED (no orbbec_camera_publisher).
echo   Opens Orbbec index 0 and 1 side-by-side. Q / Esc quits.
echo.
echo   If ROS dual sync IS running, do NOT use this — use:
echo     ..\ros2_ws\view_dual_cameras.cmd
echo ============================================================
echo.

set "ORBBEC_CAMERA_ROOT=%~dp0"
for %%I in ("%ORBBEC_CAMERA_ROOT%") do set "ORBBEC_CAMERA_ROOT=%%~fI"

cmd /c "tasklist /FI \"IMAGENAME eq orbbec_camera_publisher.exe\" 2>nul | findstr /I orbbec_camera_publisher >nul"
if not errorlevel 1 (
  echo ERROR: ROS orbbec_camera_publisher is still running.
  echo Stop ROS first, or use:  ..\ros2_ws\view_dual_cameras.cmd
  exit /b 1
)

set "VIAL_PYTHON="
for %%P in (
  "%USERPROFILE%\anaconda3\python.exe"
  "C:\Users\Ceder-ALAB\anaconda3\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
  "%USERPROFILE%\miniconda3\python.exe"
  "%USERPROFILE%\AppData\Local\anaconda3\python.exe"
  "%USERPROFILE%\AppData\Local\miniconda3\python.exe"
  "C:\ProgramData\anaconda3\python.exe"
) do if not defined VIAL_PYTHON if exist %%~P set "VIAL_PYTHON=%%~P"
if not defined VIAL_PYTHON (
  for /f "delims=" %%I in ('where python 2^>nul') do (
    echo %%I | findstr /I /C:"WindowsApps" >nul
    if errorlevel 1 if not defined VIAL_PYTHON set "VIAL_PYTHON=%%I"
  )
)
if not defined VIAL_PYTHON (
  echo ERROR: Python not found. Install Anaconda/Python 3.10+.
  exit /b 1
)

cd /d "%ORBBEC_CAMERA_ROOT%"
echo Launching: "%VIAL_PYTHON%" view_dual_cameras.py %*
"%VIAL_PYTHON%" view_dual_cameras.py %*
exit /b %ERRORLEVEL%
