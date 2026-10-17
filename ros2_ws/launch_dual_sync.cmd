@echo off
REM Default dual-sync path = ROS publishers + cam1 rack overlay.
REM Same as launch_ros_dual_sync.cmd (keeps ROS cameras; does not use CAP_OBSENSOR focus).

echo.
echo === Dual sync: ROS cameras + rack overlay ===
echo (Not rack_lid_state CAP_OBSENSOR — that conflicts with ROS USB.)
echo Optional non-ROS focus only: launch_rack_focus.cmd  (stop ROS first)
echo.

call "%~dp0launch_ros_dual_sync.cmd" %*
exit /b %ERRORLEVEL%
