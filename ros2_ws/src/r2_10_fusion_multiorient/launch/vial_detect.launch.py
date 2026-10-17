"""Launch r2_10_fusion_multiorient vial_detector_node."""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("r2_10_fusion_multiorient")
    default_config = os.path.join(pkg_share, "config", "params.yaml")
    orbbec_share = get_package_share_directory("orbbec_dual_sync")
    default_orbbec_config = os.path.join(orbbec_share, "config", "dual_sync.yaml")

    config = LaunchConfiguration("config")
    run_offline = ParameterValue(LaunchConfiguration("run_offline"), value_type=bool)
    visualize = ParameterValue(LaunchConfiguration("visualize"), value_type=bool)
    camera_preview = ParameterValue(
        LaunchConfiguration("camera_preview"), value_type=bool
    )
    launch_cameras = LaunchConfiguration("launch_cameras")
    live_camera_condition = IfCondition(
        PythonExpression(
            [
                "'",
                LaunchConfiguration("run_offline"),
                "' == 'false' and '",
                launch_cameras,
                "' == 'true'",
            ]
        )
    )

    camera_config = LaunchConfiguration("orbbec_config")
    publish_rate_hz = ParameterValue(
        LaunchConfiguration("publish_rate_hz"), value_type=float
    )
    cam_rate = {"publish_rate_hz": publish_rate_hz}
    cam0 = Node(
        package="orbbec_dual_sync",
        executable="orbbec_camera_publisher",
        name="orbbec_cam0_pub",
        output="screen",
        parameters=[camera_config, cam_rate, {"show_preview": camera_preview}],
        condition=live_camera_condition,
        respawn=True,
        respawn_delay=3.0,
    )
    cam1 = Node(
        package="orbbec_dual_sync",
        executable="orbbec_camera_publisher",
        name="orbbec_cam1_pub",
        output="screen",
        parameters=[camera_config, cam_rate, {"show_preview": camera_preview}],
        respawn=True,
        respawn_delay=3.0,
    )

    node = Node(
        package="r2_10_fusion_multiorient",
        executable="vial_detector_node",
        name="vial_detector_node",
        output="screen",
        parameters=[
            config,
            {
                "run_offline": run_offline,
                "visualize": visualize,
                "use_yolo_live": ParameterValue(
                    LaunchConfiguration("use_yolo_live"), value_type=bool
                ),
                "image_topic": LaunchConfiguration("image_topic"),
                "depth_topic": LaunchConfiguration("depth_topic"),
                "image_topic_2": LaunchConfiguration("image_topic_2"),
                "depth_topic_2": LaunchConfiguration("depth_topic_2"),
                "extra_yolo_conf_after_method_i": ParameterValue(
                    LaunchConfiguration("extra_yolo_conf_after_method_i"),
                    value_type=float,
                ),
                "empty_scene_yolo_floor": ParameterValue(
                    LaunchConfiguration("empty_scene_yolo_floor"),
                    value_type=float,
                ),
                "empty_method_i_keep_conf": ParameterValue(
                    LaunchConfiguration("empty_method_i_keep_conf"),
                    value_type=float,
                ),
                "enable_rgb_classical_sideways": ParameterValue(
                    LaunchConfiguration("enable_rgb_classical_sideways"),
                    value_type=bool,
                ),
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=default_config,
                description="YAML params",
            ),
            DeclareLaunchArgument(
                "run_offline",
                default_value="true",
                description="true=iterate dataset_path; false=subscribe image_topic",
            ),
            DeclareLaunchArgument(
                "visualize",
                default_value="true",
                description=(
                    "Publish /vial/annotated and open one annotated preview "
                    "(side-by-side in dual-camera live mode)"
                ),
            ),
            DeclareLaunchArgument(
                "use_yolo_live",
                default_value="false",
                description=(
                    "Opt in to in-process YOLO for live ROS. Keep false on "
                    "Windows RoboStack, where Torch/Ultralytics can abort the "
                    "node with native exit code 3. Offline use_yolo is unchanged."
                ),
            ),
            DeclareLaunchArgument(
                "launch_cameras",
                default_value="true",
                description=(
                    "In live mode, start both orbbec_dual_sync camera publishers. "
                    "Set false when publishers already run."
                ),
            ),
            DeclareLaunchArgument(
                "camera_preview",
                default_value="false",
                description=(
                    "In live mode, also open separate raw tkinter previews in "
                    "the camera publishers. Off by default so the combined "
                    "annotated detector preview is the primary UI."
                ),
            ),
            DeclareLaunchArgument(
                "orbbec_config",
                default_value=default_orbbec_config,
                description="orbbec_dual_sync publisher YAML",
            ),
            DeclareLaunchArgument(
                "publish_rate_hz",
                default_value="6.0",
                description=(
                    "Live camera publisher timer rate (overrides dual_sync.yaml "
                    "when launch_cameras:=true). Preview smoothness is usually "
                    "capped by detection, not this timer."
                ),
            ),
            DeclareLaunchArgument(
                "image_topic",
                default_value="/orbbec/cam0/color/image_raw",
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/orbbec/cam0/depth/image_raw",
            ),
            DeclareLaunchArgument(
                "image_topic_2",
                default_value="/orbbec/cam1/color/image_raw",
            ),
            DeclareLaunchArgument(
                "depth_topic_2",
                default_value="/orbbec/cam1/depth/image_raw",
            ),
            DeclareLaunchArgument(
                "extra_yolo_conf_after_method_i",
                default_value="0.60",
                description="Post-Method-I extra YOLO conf gate (ERROR_AUDIT)",
            ),
            DeclareLaunchArgument(
                "empty_scene_yolo_floor",
                default_value="0.35",
                description="Minimum YOLO conf floor (empty-scene clutter)",
            ),
            DeclareLaunchArgument(
                "empty_method_i_keep_conf",
                default_value="0.50",
                description="Keep upright/top-down YOLO when Method-I empty",
            ),
            DeclareLaunchArgument(
                "enable_rgb_classical_sideways",
                default_value="false",
                description="Allow RGB-only classical sideways without depth",
            ),
            cam0,
            TimerAction(
                # cam0 normally streams before this expires; the publisher's
                # cross-process mutex is the strict sequential-open gate.
                period=20.0,
                actions=[cam1],
                condition=live_camera_condition,
            ),
            node,
        ]
    )
