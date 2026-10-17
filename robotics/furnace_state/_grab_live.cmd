@echo off
cd /d C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
set PYTHONPATH=C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics;%PYTHONPATH%
python -c "import sys; sys.path.insert(0, r'C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics'); from furnace_agents import grab_frames_ros; import cv2; fs=grab_frames_ros(3, topic='/orbbec/cam0/color/image_raw', timeout_s=20, interval_s=0.4); print('n', len(fs));
import pathlib; pathlib.Path(r'C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\furnace_state\live_for_slots.png').write_bytes(b'') if False else None
cv2.imwrite(r'C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\furnace_state\live_for_slots.png', fs[-1]) if fs else print('noframe')"
