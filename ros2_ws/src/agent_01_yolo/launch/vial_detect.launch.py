"""Launch agent_01_yolo vial_detector_node with config/params.yaml defaults."""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("agent_01_yolo")
    default_config = os.path.join(pkg_share, "config", "params.yaml")

    config = LaunchConfiguration("config")
    run_offline = ParameterValue(LaunchConfiguration("run_offline"), value_type=bool)
    visualize = ParameterValue(LaunchConfiguration("visualize"), value_type=bool)

    node = Node(
        package="agent_01_yolo",
        executable="vial_detector_node",
        name="vial_detector_node",
        output="screen",
        parameters=[
            config,
            {
                "run_offline": run_offline,
                "visualize": visualize,
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=default_config,
                description="YAML params (share/agent_01_yolo/config/params.yaml)",
            ),
            DeclareLaunchArgument(
                "run_offline",
                default_value="true",
                description="true=iterate dataset_path; false=subscribe image_topic",
            ),
            DeclareLaunchArgument(
                "visualize",
                default_value="true",
                description="Publish annotated Image on /vial/annotated",
            ),
            node,
        ]
    )
