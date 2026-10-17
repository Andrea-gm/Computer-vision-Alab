@echo off
cd /d C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
set PYTHONPATH=C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics;%PYTHONPATH%
set KMP_DUPLICATE_LIB_OK=TRUE
echo Starting dual_view (furnace + rack + arm YOLO + baby-blue + 10cm proximity)...
ros2 run orbbec_dual_sync orbbec_dual_view --ros-args ^
  -p enable_furnace_cv:=true ^
  -p enable_rack_cv:=true ^
  -p enable_arm_cv:=true ^
  -p enable_blue_cv:=true ^
  -p blue_show_mask:=true ^
  -p blue_proximity_mm:=100.0 ^
  -p furnace_match_live:=true ^
  -p state_mode:=classify ^
  -p arm_conf:=0.25 ^
  -p arm_every_n:=3
