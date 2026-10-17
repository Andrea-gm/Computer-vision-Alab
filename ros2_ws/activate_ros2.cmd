@echo off
REM Activate Pixi/RoboStack into this cmd window (no ExecutionPolicy needed).
REM Usage:
REM   activate_ros2.cmd              - activate and leave you in the env
REM   activate_ros2.cmd <command...> - activate, run command, exit with its code

call "%~dp0_ros_env.cmd"
if errorlevel 1 exit /b 1

echo ROS_DISTRO=%ROS_DISTRO%
where colcon 2>nul
where ros2 2>nul

if "%~1"=="" (
  echo Pixi/ROS2 active. Next: build.cmd  ^|  launch_dual_sync.cmd  ^(one-shot^)
  echo PowerShell: .\build.cmd  ^|  .\launch_dual_sync.cmd  ^(.\ required^)
  exit /b 0
)

%*
exit /b %ERRORLEVEL%
