"""Launch Round-2 Agent 07 depth any-pose vial detector."""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("r2_07_depth_anypose")
    default_params = os.path.join(share, "config", "params.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument(
                "color_topic", default_value="/orbbec/cam0/color/image_raw"
            ),
            DeclareLaunchArgument(
                "depth_topic", default_value="/orbbec/cam0/depth/image_raw"
            ),
            DeclareLaunchArgument(
                "camera_info_topic",
                default_value="/orbbec/cam0/color/camera_info",
            ),
            Node(
                package="r2_07_depth_anypose",
                executable="depth_anypose_node",
                name="r2_07_depth_anypose",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "color_topic": LaunchConfiguration("color_topic"),
                        "depth_topic": LaunchConfiguration("depth_topic"),
                        "camera_info_topic": LaunchConfiguration(
                            "camera_info_topic"
                        ),
                    },
                ],
            ),
        ]
    )
