# See both Orbbec cameras

`ros2 topic list` only prints topic **names**. It never shows video.

## One command (ROS + dual window)

```powershell
cd C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws
.\launch_ros_dual_sync.cmd
```

Wait **~30 seconds**. Look for the window titled:

`orbbec_dual_view (cam0 | cam1)`

That is cam0 (left) and cam1 (right) live.

## If launch is already running

Second terminal:

```powershell
cd C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\ros2_ws
.\view_dual_cameras.cmd
```

## Without ROS (USB direct)

Stop any `launch_ros_dual_sync` / orbbec publishers first, then:

```powershell
cd C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics
.\view_dual_cameras.cmd
```
