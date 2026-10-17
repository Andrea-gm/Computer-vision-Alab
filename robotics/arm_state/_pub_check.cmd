@echo off
cd /d C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
echo === cam0 ===
ros2 topic info /orbbec/cam0/color/image_raw
echo === cam1 ===
ros2 topic info /orbbec/cam1/color/image_raw
