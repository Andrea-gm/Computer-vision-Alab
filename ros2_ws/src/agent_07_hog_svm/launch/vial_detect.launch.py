"""Launch HOG+SVM vial detector (live topic or offline dataset)."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("agent_07_hog_svm"))
    default_params = str(share / "config" / "params.yaml")
    default_model = str(share / "models" / "hog_svm_vial.json")

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            DeclareLaunchArgument("image_topic", default_value="/camera/color/image_raw"),
            DeclareLaunchArgument("dataset_path", default_value=""),
            DeclareLaunchArgument("model_path", default_value=default_model),
            DeclareLaunchArgument("confidence_threshold", default_value="0.5"),
            DeclareLaunchArgument("visualize", default_value="true"),
            Node(
                package="agent_07_hog_svm",
                executable="vial_detector_node",
                name="vial_detector_node",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "image_topic": LaunchConfiguration("image_topic"),
                        "dataset_path": LaunchConfiguration("dataset_path"),
                        "model_path": LaunchConfiguration("model_path"),
                        "confidence_threshold": LaunchConfiguration("confidence_threshold"),
                        "visualize": LaunchConfiguration("visualize"),
                    },
                ],
            ),
        ]
    )
