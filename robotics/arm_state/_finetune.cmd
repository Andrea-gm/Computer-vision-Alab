@echo off
cd /d C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
set PYTHONPATH=C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics;%PYTHONPATH%
set KMP_DUPLICATE_LIB_OK=TRUE
python C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\arm_agents.py train --epochs 8 --imgsz 640 --model "C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\arm_state\runs\arm_detect\weights\best.pt"
python C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\arm_agents.py status
