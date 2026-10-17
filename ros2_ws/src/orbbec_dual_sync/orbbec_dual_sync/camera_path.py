"""Locate the robotics tree so ROS nodes can import depthcamera helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def camera_root() -> Path:
    """Return the folder containing ``depthcamera.py``."""
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for key in ("ORBBEC_CAMERA_ROOT", "JESSICA_ROBOTICS_ROOT"):
        if value := os.environ.get(key):
            candidates.append(Path(os.path.expanduser(value.strip().strip('"'))))

    # Installed/source package paths both sit below ros2_ws; Jessica/robotics
    # is its sibling. Keep Camera as a transition fallback for older layouts.
    for parent in here.parents:
        if parent.name == "ros2_ws":
            candidates.extend((parent.parent / "robotics", parent.parent / "Camera"))
            break
    candidates.extend((Path.cwd(), Path.cwd().parent, Path.cwd().parent / "robotics"))

    for root in candidates:
        root = root.resolve()
        if (root / "depthcamera.py").is_file():
            return root
    for parent in here.parents:
        if (parent / "depthcamera.py").is_file():
            return parent
    raise FileNotFoundError(
        "Could not find robotics/depthcamera.py. Set ORBBEC_CAMERA_ROOT to "
        "the robotics folder."
    )


def ensure_camera_on_path() -> Path:
    root = camera_root()
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root
