@echo off
cd /d C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
python C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\arm_state\_hsv_probe.py
