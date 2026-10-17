@echo off
cd /d C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
ros2 topic list
ros2 topic info /orbbec/cam1/color/image_raw
ros2 topic info /orbbec/cam0/color/image_raw
