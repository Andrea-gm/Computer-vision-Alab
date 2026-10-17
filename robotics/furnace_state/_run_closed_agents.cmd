@echo off
cd /d C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
set PYTHONPATH=C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics;%PYTHONPATH%
echo === topic cam0 ===
ros2 topic info /orbbec/cam0/color/image_raw
echo === CaptureAgent + LabelAgent (CLOSED template match) ===
python C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\furnace_state\_closed_capture_label.py
echo exit=%ERRORLEVEL%
