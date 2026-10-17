@echo off
REM SupervisorAgent + 5 rack CV agents (Capture/Label/Algo/Train/Judge).
REM Prefer --source ros while dual_sync publishers own the Orbbec USB.
REM
REM Examples:
REM   rack_cv_supervisor.cmd bootstrap-closed --source ros --n 8
REM   rack_cv_supervisor.cmd bootstrap-open --source ros --n 8
REM   rack_cv_supervisor.cmd status

set "ROBOTICS_ROOT=%~dp0"
if "%ROBOTICS_ROOT:~-1%"=="\" set "ROBOTICS_ROOT=%ROBOTICS_ROOT:~0,-1%"
set "ROS_WS_ROOT=%ROBOTICS_ROOT%\..\ros2_ws"

call "%ROS_WS_ROOT%\_ros_env.cmd"
if errorlevel 1 exit /b 1

if exist "%ROS_WS_ROOT%\install\setup.bat" (
  call "%ROS_WS_ROOT%\install\setup.bat"
) else if exist "%ROS_WS_ROOT%\install\local_setup.bat" (
  call "%ROS_WS_ROOT%\install\local_setup.bat"
)

cd /d "%ROBOTICS_ROOT%"
python "%ROBOTICS_ROOT%\rack_cv_supervisor.py" %*
exit /b %ERRORLEVEL%
