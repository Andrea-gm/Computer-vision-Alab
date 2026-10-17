"""Launch agent_06 vial_detector_node (instance segmentation)."""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("agent_06_segmentation")
    default_config = os.path.join(pkg_share, "config", "params.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("run_offline", default_value="true"),
            DeclareLaunchArgument("model_path", default_value=""),
            DeclareLaunchArgument(
                "image_topic", default_value="/camera/color/image_raw"
            ),
            Node(
                package="agent_06_segmentation",
                executable="vial_detector_node",
                name="vial_detector_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    {
                        "run_offline": ParameterValue(
                            LaunchConfiguration("run_offline"), value_type=bool
                        ),
                        "model_path": LaunchConfiguration("model_path"),
                        "image_topic": LaunchConfiguration("image_topic"),
                    },
                ],
            ),
        ]
    )
