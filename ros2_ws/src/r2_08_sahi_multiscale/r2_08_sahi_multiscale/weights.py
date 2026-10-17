"""Resolve Camera vial YOLO weights (same preference order as agent_01 / depthcamera)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def default_camera_root() -> Path:
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / "vial_dataset").is_dir() and (parent / "runs").is_dir():
            return parent
        if (parent / "vial_dataset").is_dir() and (parent / "yolov8s.pt").is_file():
            return parent
    for parent in here.parents:
        if parent.name == "ros2_ws":
            sibling = parent.parent / "robotics"
            if sibling.is_dir():
                return sibling
            break
    return here.parents[4]


def resolve_model_path(
    explicit: Optional[Union[str, Path]] = None,
    camera_root: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    root = Path(camera_root) if camera_root else default_camera_root()
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    # Prefer weights that fire on both upright crops and sideways synth stills.
    # vial-live is strong on some holdouts but often silent on sideways_race/synth
    # at conf≥0.25; bon-winner / r1 cover both orientations for SAHI eval.
    preferred = [
        root / "runs" / "bon-winner" / "weights" / "best.pt",
        root / "runs" / "train-race" / "r1-winner" / "weights" / "best.pt",
        root / "runs" / "train-race" / "r3-winner" / "weights" / "best.pt",
        root / "runs" / "vial-live" / "weights" / "best.pt",
        root / "yolov8s.pt",
        root / "yolov8n.pt",
    ]
    for path in preferred:
        if path.exists():
            return path
    return None
