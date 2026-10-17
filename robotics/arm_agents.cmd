@echo off
REM Arm capture / label / train (ROS publishers must be up for --source ros)
cd /d C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
set PYTHONPATH=C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics;%PYTHONPATH%
set KMP_DUPLICATE_LIB_OK=TRUE
python C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\arm_agents.py %*
