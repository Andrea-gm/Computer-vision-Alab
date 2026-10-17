@echo off
REM r2_10 vial detector launch is DISABLED. Use rack tools instead.

echo.
echo === Vial detection DISABLED (r2_10) ===
echo This launcher will NOT start vial_detect / YOLO / vial boxes.
echo Use rack open/closed tools:
echo.
echo   cd /d "%~dp0..\robotics"
echo   python rack_lid_state.py focus
echo   python rack_capture_agents.py run --camera 1 --seconds 30
echo.
echo Legacy r2_10 (not recommended): ros2 launch r2_10_fusion_multiorient vial_detect.launch.py
echo.
exit /b 2
