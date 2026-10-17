@echo off
REM Non-ROS camera-1 rack focus via CAP_OBSENSOR (conflicts with ROS publishers).
REM Prefer launch_ros_dual_sync.cmd / launch_dual_sync.cmd for ROS + overlay UI.

echo.
echo ============================================================
echo   RACK FOCUS (CAP_OBSENSOR) — only if ROS is NOT running
echo ============================================================
echo   Prefer ROS + overlay instead:
echo     launch_ros_dual_sync.cmd
echo     launch_dual_sync.cmd
echo   Those keep publishers and show orbbec_rack_overlay.
echo ============================================================
echo.

set "ORBBEC_CAMERA_ROOT=%~dp0..\robotics"
for %%I in ("%ORBBEC_CAMERA_ROOT%") do set "ORBBEC_CAMERA_ROOT=%%~fI"

cmd /c "tasklist /FI \"IMAGENAME eq orbbec_camera_publisher.exe\" 2>nul | findstr /I orbbec_camera_publisher >nul"
if not errorlevel 1 (
  echo ERROR: ROS orbbec_camera_publisher is still running.
  echo Stop ROS first, or use launch_ros_dual_sync.cmd for the overlay window.
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
echo Launching: "%VIAL_PYTHON%" rack_lid_state.py focus
"%VIAL_PYTHON%" rack_lid_state.py focus %*
exit /b %ERRORLEVEL%
