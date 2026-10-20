"""Cell frame graph: ur_base (0,0,0) and fixed camera poses.

p_base = R @ p_cam + t_mm

Default layout: cam0 and cam1 are perpendicular (90 deg yaw about camera Y),
both looking inward at the cell. t is bootstrapped so the blue pedestal is origin.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# cam1 optical frame relative to cam0 when mounts are perpendicular (level cameras):
# yaw +90 deg about Y (OpenCV: X right, Y down, Z forward).
PERPENDICULAR_R_CAM1 = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
LAYOUT_PERPENDICULAR = "perpendicular_90"


@dataclass
class CamPose:
    frame_id: str
    R: np.ndarray  # 3x3
    t_mm: np.ndarray  # 3,
    ready: bool = False


@dataclass
class CellFrames:
    path: Path
    bootstrap_from_blue_base: bool
    fx_px: float
    rack_height_mm: float
    cam0: CamPose
    cam1: CamPose
    layout: str = LAYOUT_PERPENDICULAR

    def pose(self, cam: str) -> CamPose:
        return self.cam0 if cam == "cam0" else self.cam1


def default_cell_frames_path(robotics_root: Optional[Path] = None) -> Path:
    if robotics_root is not None:
        return Path(robotics_root) / "cell_frames.json"
    # Prefer ORBBEC_CAMERA_ROOT / robotics next to dual_view
    import os

    env = os.environ.get("ORBBEC_CAMERA_ROOT", "").strip()
    if env:
        return Path(env) / "cell_frames.json"
    return Path(__file__).resolve().parents[4] / "robotics" / "cell_frames.json"


def _pose_from_dict(d: dict, default_id: str) -> CamPose:
    R = np.asarray(d.get("R", np.eye(3)), dtype=np.float64).reshape(3, 3)
    t = np.asarray(d.get("t_mm", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    return CamPose(
        frame_id=str(d.get("frame_id", default_id)),
        R=R,
        t_mm=t,
        ready=bool(d.get("ready", False)),
    )


def apply_perpendicular_layout(cf: CellFrames) -> CellFrames:
    """cam0 = identity; cam1 = 90 deg yaw. Keeps existing t_mm / ready flags."""
    cf.layout = LAYOUT_PERPENDICULAR
    cf.cam0.R = np.eye(3, dtype=np.float64)
    cf.cam1.R = PERPENDICULAR_R_CAM1.copy()
    return cf


def load_cell_frames(path: Optional[Path] = None) -> CellFrames:
    path = Path(path) if path is not None else default_cell_frames_path()
    if not path.is_file():
        eye = np.eye(3, dtype=np.float64)
        z = np.zeros(3, dtype=np.float64)
        cf = CellFrames(
            path=path,
            bootstrap_from_blue_base=True,
            fx_px=520.0,
            rack_height_mm=400.0,
            cam0=CamPose("orbbec_cam0", eye.copy(), z.copy(), False),
            cam1=CamPose("orbbec_cam1", PERPENDICULAR_R_CAM1.copy(), z.copy(), False),
            layout=LAYOUT_PERPENDICULAR,
        )
        return cf
    data = json.loads(path.read_text(encoding="utf-8"))
    cams = data.get("cameras") or {}
    layout = str(data.get("layout") or LAYOUT_PERPENDICULAR)
    cf = CellFrames(
        path=path,
        bootstrap_from_blue_base=bool(data.get("bootstrap_from_blue_base", True)),
        fx_px=float(data.get("fx_px", 520.0)),
        rack_height_mm=float(data.get("rack_height_mm", 400.0)),
        cam0=_pose_from_dict(cams.get("cam0") or {}, "orbbec_cam0"),
        cam1=_pose_from_dict(cams.get("cam1") or {}, "orbbec_cam1"),
        layout=layout,
    )
    # Enforce known perpendicular prior unless user set a custom layout name.
    if layout == LAYOUT_PERPENDICULAR:
        apply_perpendicular_layout(cf)
    return cf


def save_cell_frames(cf: CellFrames) -> None:
    payload = {
        "world_frame": "ur_base",
        "origin_mm": [0.0, 0.0, 0.0],
        "layout": cf.layout,
        "notes": (
            "ur_base is (0,0,0) at the blue pedestal joint. "
            "p_base = R @ p_cam + t_mm. "
            "layout=perpendicular_90: cam0 R=I, cam1 R=yaw+90 about Y. "
            "bootstrap updates t so each camera maps the pedestal to the origin."
        ),
        "bootstrap_from_blue_base": bool(cf.bootstrap_from_blue_base),
        "fx_px": float(cf.fx_px),
        "rack_height_mm": float(cf.rack_height_mm),
        "cameras": {
            "cam0": {
                "frame_id": cf.cam0.frame_id,
                "R": cf.cam0.R.tolist(),
                "t_mm": cf.cam0.t_mm.reshape(3).tolist(),
                "ready": bool(cf.cam0.ready),
            },
            "cam1": {
                "frame_id": cf.cam1.frame_id,
                "R": cf.cam1.R.tolist(),
                "t_mm": cf.cam1.t_mm.reshape(3).tolist(),
                "ready": bool(cf.cam1.ready),
            },
        },
        "stereo_extrinsics": "stereo_extrinsics.json",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    cf.path.parent.mkdir(parents=True, exist_ok=True)
    cf.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def cam_to_base(p_cam_mm: np.ndarray, pose: CamPose) -> np.ndarray:
    p = np.asarray(p_cam_mm, dtype=np.float64).reshape(3)
    return pose.R @ p + pose.t_mm.reshape(3)


def bootstrap_pose_from_base_landmark(
    p_base_in_cam_mm: np.ndarray,
    *,
    R: Optional[np.ndarray] = None,
    frame_id: str = "orbbec_cam",
) -> CamPose:
    """Set translation so the seen pedestal maps to ur_base origin (R default I)."""
    R_use = np.eye(3, dtype=np.float64) if R is None else np.asarray(R, dtype=np.float64)
    p = np.asarray(p_base_in_cam_mm, dtype=np.float64).reshape(3)
    t = -R_use @ p
    return CamPose(frame_id=frame_id, R=R_use, t_mm=t, ready=True)


def format_base_retreat(p_joint_base_mm: np.ndarray) -> str:
    """Instruction to move a joint at p_joint_base toward ur_base (0,0,0)."""
    p = np.asarray(p_joint_base_mm, dtype=np.float64).reshape(3)
    # Vector joint → origin
    v = -p
    dist = float(np.linalg.norm(v))
    if dist < 20.0:
        return (
            "FUSED RETREAT -> ur_base (0,0,0): already near BASE "
            f"(~{dist / 10.0:.0f}cm) - pull gently to home"
        )
    parts: list[str] = []
    # Report in ur_base axes (same numeric axes as camera until R is calibrated)
    axes = (
        ("+X" if v[0] > 0 else "-X", abs(v[0])),
        ("+Y" if v[1] > 0 else "-Y", abs(v[1])),
        ("+Z" if v[2] > 0 else "-Z", abs(v[2])),
    )
    for name, mag in axes:
        if mag >= 30.0:
            parts.append(f"{name} ~{mag / 10.0:.0f}cm")
    if not parts:
        parts.append(f"toward origin ~{dist / 10.0:.0f}cm")
    return (
        "FUSED RETREAT -> move arm toward ROBOT BASE ur_base (0,0,0): "
        + ", ".join(parts)
        + f"  |  remaining ~{dist / 10.0:.0f}cm"
    )
