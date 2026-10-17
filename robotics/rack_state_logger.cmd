@echo off
REM Log rack OPEN/CLOSED transitions (needs ROS cam1 publisher running).
cd /d "%~dp0..\ros2_ws"
call _ros_env.cmd
if errorlevel 1 exit /b 1
if exist "%CD%\install\setup.bat" (
  call "%CD%\install\setup.bat"
) else if exist "%CD%\install\local_setup.bat" (
  call "%CD%\install\local_setup.bat"
)
set PYTHONPATH=%~dp0;%PYTHONPATH%
echo.
echo Rack state logger — Ctrl+C to stop.
echo Log: %~dp0rack_lid_state\state_transitions.log
echo Topics: /orbbec/cam1/rack_state  /orbbec/cam1/rack_state_event
echo.
python "%~dp0rack_state_logger.py" %*
exit /b %ERRORLEVEL%
