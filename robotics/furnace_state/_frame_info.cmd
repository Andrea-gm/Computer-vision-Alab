@echo off
cd /d C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
set PYTHONPATH=C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics;%PYTHONPATH%
python -c "import cv2,json; from pathlib import Path; p=Path(r'C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\furnace_state\live_for_slots.png'); im=cv2.imread(str(p)); print('shape', None if im is None else im.shape)"
