@echo off
cd /d C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
set PYTHONPATH=C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics;%PYTHONPATH%
python C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\arm_agents.py capture --seconds 240 --interval 0.2 --source ros
