#!/usr/bin/env python3
"""Static layout check for orbbec_dual_sync (no ROS / colcon required).

Exits 0 if the ament_python package looks colcon-buildable as real ROS2 Nodes.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "src" / "orbbec_dual_sync"
PY = PKG / "orbbec_dual_sync"

REQUIRED = [
    PKG / "package.xml",
    PKG / "setup.py",
    PKG / "setup.cfg",
    PKG / "resource" / "orbbec_dual_sync",
    PKG / "launch" / "dual_sync_detect.launch.py",
    PKG / "config" / "dual_sync.yaml",
    PY / "__init__.py",
    PY / "camera_publisher.py",
    PY / "sync_detect_node.py",
    PY / "sync_forward_node.py",
    PY / "approx_time.py",
    PY / "detect_bridge.py",
    PY / "image_codec.py",
    PY / "camera_path.py",
]

ENTRY_POINTS = {
    "orbbec_camera_publisher": "orbbec_dual_sync.camera_publisher:main",
    "orbbec_sync_detect": "orbbec_dual_sync.sync_detect_node:main",
    "orbbec_sync_forward": "orbbec_dual_sync.sync_forward_node:main",
}

NODE_CLASSES = {
    "camera_publisher.py": "OrbbecCameraPublisher",
    "sync_detect_node.py": "SyncDetectNode",
    "sync_forward_node.py": "SyncForwardNode",
}


def _inherits_node(tree: ast.AST, class_name: str) -> bool:
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == class_name:
            for base in n.bases:
                if isinstance(base, ast.Name) and base.id == "Node":
                    return True
                if isinstance(base, ast.Attribute) and base.attr == "Node":
                    return True
    return False


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED:
        if not path.exists():
            errors.append(f"missing: {path.relative_to(ROOT)}")

    xml = (PKG / "package.xml").read_text(encoding="utf-8") if (PKG / "package.xml").exists() else ""
    if "<build_type>ament_python</build_type>" not in xml:
        errors.append("package.xml: expected <build_type>ament_python</build_type>")
    for dep in ("rclpy", "sensor_msgs", "message_filters", "launch_ros", "ament_index_python"):
        if dep not in xml:
            errors.append(f"package.xml: missing depend/exec_depend on {dep}")

    setup_txt = (PKG / "setup.py").read_text(encoding="utf-8") if (PKG / "setup.py").exists() else ""
    for exe, target in ENTRY_POINTS.items():
        needle = f"{exe} = {target}"
        if needle not in setup_txt:
            errors.append(f"setup.py entry_points missing: {needle}")
    for needle in (
        'resource/" + package_name',
        "config/dual_sync.yaml",
        "launch/dual_sync_detect.launch.py",
    ):
        if needle not in setup_txt:
            errors.append(f"setup.py data_files missing reference: {needle}")

    launch_txt = ""
    launch_path = PKG / "launch" / "dual_sync_detect.launch.py"
    if launch_path.exists():
        launch_txt = launch_path.read_text(encoding="utf-8")
        for needle in (
            "get_package_share_directory",
            "dual_sync.yaml",
            "parameters=[config]",
            'executable="orbbec_camera_publisher"',
            'executable="orbbec_sync_detect"',
            "TimerAction",
        ):
            if needle not in launch_txt:
                errors.append(f"launch: missing {needle!r}")

    yaml_txt = ""
    yaml_path = PKG / "config" / "dual_sync.yaml"
    if yaml_path.exists():
        yaml_txt = yaml_path.read_text(encoding="utf-8")
        for needle in (
            "orbbec_cam0_pub:",
            "orbbec_cam1_pub:",
            "orbbec_sync_detect:",
            "slop_s:",
            "camera_index:",
            "weak_camera:",
        ):
            if needle not in yaml_txt:
                errors.append(f"config/dual_sync.yaml: missing {needle!r}")

    for mod, cls in NODE_CLASSES.items():
        path = PY / mod
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
        if "from rclpy.node import Node" not in src and "rclpy.node" not in src:
            errors.append(f"{mod}: does not import rclpy.node.Node")
        if not _inherits_node(tree, cls):
            errors.append(f"{mod}: {cls} must subclass Node")
        if not any(
            isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body
        ):
            errors.append(f"{mod}: missing top-level main()")
        if "rclpy.spin" not in src:
            errors.append(f"{mod}: main must rclpy.spin the Node")

    sync_src = (PY / "sync_detect_node.py").read_text(encoding="utf-8") if (PY / "sync_detect_node.py").exists() else ""
    if "ApproximateTimeSynchronizer" not in sync_src:
        errors.append("sync_detect_node.py: missing message_filters ApproximateTimeSynchronizer")

    if errors:
        print("LAYOUT FAIL (would not pass colcon / ROS2 Node structure review):")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("LAYOUT OK: real ROS2 ament_python package (rclpy Nodes + launch + YAML).")
    print(f"  package root: {PKG}")
    print("  Nodes: OrbbecCameraPublisher, SyncDetectNode, SyncForwardNode")
    print("  launch: dual_sync_detect.launch.py loads parameters=[config YAML]")
    print("  config: config/dual_sync.yaml -> share/.../config/ after colcon")
    print("Next: cd ros2_ws && ./build.sh && source install/setup.bash")
    print("      ros2 launch orbbec_dual_sync dual_sync_detect.launch.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
