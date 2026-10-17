"""Launch the Agent #10 ensemble vial detector node."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = Path(get_package_share_directory("agent_10_ensemble"))
    default_params = str(pkg_share / "config" / "ensemble_params.yaml")
    default_verifier = str(pkg_share / "models" / "verifier.pkl")

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("image_topic", default_value="/camera/color/image_raw"),
            DeclareLaunchArgument("verifier_path", default_value=default_verifier),
            DeclareLaunchArgument("use_yolo", default_value="true"),
            Node(
                package="agent_10_ensemble",
                executable="vial_detector_node",
                name="vial_detector_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "image_topic": LaunchConfiguration("image_topic"),
                        "verifier_path": LaunchConfiguration("verifier_path"),
                        "use_yolo": LaunchConfiguration("use_yolo"),
                    },
                ],
            ),
        ]
    )
