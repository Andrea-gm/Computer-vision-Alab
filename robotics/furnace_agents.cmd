@echo off
REM Furnace multi-agent CV pipeline (Orbbec camera 0 only).
REM Sources ROS env so /orbbec/cam0 works when publishers are already up.
REM Does NOT launch dual_sync / dual_view / rack_overlay.
REM
REM Examples:
REM   furnace_agents.cmd seed
REM   furnace_agents.cmd run --seconds 120 --background --preview
REM   furnace_agents.cmd train
REM   furnace_agents.cmd status
REM   furnace_agents.cmd stop

set "ROBOTICS_ROOT=%~dp0"
if "%ROBOTICS_ROOT:~-1%"=="\" set "ROBOTICS_ROOT=%ROBOTICS_ROOT:~0,-1%"
set "ROS_WS_ROOT=%ROBOTICS_ROOT%\..\ros2_ws"

call "%ROS_WS_ROOT%\_ros_env.cmd"
if errorlevel 1 (
  echo WARNING: ROS env failed — USB CAP_OBSENSOR path only.
  set "PY=C:\Users\Ceder-ALAB\anaconda3\python.exe"
  if not exist "%PY%" set "PY=python"
) else (
  if exist "%ROS_WS_ROOT%\install\setup.bat" (
    call "%ROS_WS_ROOT%\install\setup.bat"
  ) else if exist "%ROS_WS_ROOT%\install\local_setup.bat" (
    call "%ROS_WS_ROOT%\install\local_setup.bat"
  )
  set "PY=python"
)

cd /d "%ROBOTICS_ROOT%"
"%PY%" "%ROBOTICS_ROOT%\furnace_agents.py" %*
exit /b %ERRORLEVEL%
