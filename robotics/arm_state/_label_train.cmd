@echo off
cd /d C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
set PYTHONPATH=C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics;%PYTHONPATH%
echo === LABEL ===
python C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\arm_agents.py label --method color --stride 2
echo === TRAIN ===
python C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\arm_agents.py train --epochs 35 --imgsz 640
python C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\arm_agents.py status
