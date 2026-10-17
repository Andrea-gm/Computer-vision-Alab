"""Launch two-stage orientation vial detector."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    pkg = Path(get_package_share_directory("r2_06_two_stage_orient"))
    default_params = str(pkg / "config" / "params.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("run_offline", default_value="true"),
            DeclareLaunchArgument(
                "image_topic", default_value="/orbbec/cam0/color/image_raw"
            ),
            DeclareLaunchArgument(
                "depth_topic", default_value="/orbbec/cam0/depth/image_raw"
            ),
            Node(
                package="r2_06_two_stage_orient",
                executable="vial_detector_node",
                name="vial_detector_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "run_offline": LaunchConfiguration("run_offline"),
                        "image_topic": LaunchConfiguration("image_topic"),
                        "depth_topic": LaunchConfiguration("depth_topic"),
                    },
                ],
            ),
        ]
    )
