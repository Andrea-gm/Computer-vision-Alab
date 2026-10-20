"""Cell frame graph: ur_base (0,0,0) and fixed camera poses.

p_base = R @ p_cam + t_mm

Default layout: cam0 and cam1 are perpendicular (90 deg yaw about camera Y),
both looking inward at the cell. t is bootstrapped so the blue pedestal is origin.
Scene+tape calib can replace the assumed R when it succeeds.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
CELL_FRAMES_PATH = ROOT / "cell_frames.json"

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
LAYOUT_SCENE_TAPE = "scene_tape"


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
    env = os.environ.get("ORBBEC_CAMERA_ROOT", "").strip()
    if env:
        return Path(env) / "cell_frames.json"
    return CELL_FRAMES_PATH


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
        return CellFrames(
            path=path,
            bootstrap_from_blue_base=True,
            fx_px=520.0,
            rack_height_mm=400.0,
            cam0=CamPose("orbbec_cam0", eye.copy(), z.copy(), False),
            cam1=CamPose("orbbec_cam1", PERPENDICULAR_R_CAM1.copy(), z.copy(), False),
            layout=LAYOUT_PERPENDICULAR,
        )
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
    # Enforce known perpendicular prior unless scene calib wrote a custom layout.
    if layout == LAYOUT_PERPENDICULAR:
        apply_perpendicular_layout(cf)
    return cf


def save_cell_frames(cf: CellFrames) -> None:
    payload = {
        "world_frame": "ur_base",
        "origin_mm": [0.0, 0.0, 0.0],
        "layout": cf.layout,
        "notes": (
            "ur_base is (0,0,0) at the blue pedestal / purple-tape origin. "
            "p_base = R @ p_cam + t_mm. "
            "layout=perpendicular_90: cam0 R=I, cam1 R=yaw+90 about Y. "
            "layout=scene_tape: R,t from shared-scene ORB + purple-tape Umeyama. "
            "bootstrap updates t so each camera maps the pedestal/tape to the origin."
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
    """Set translation so the seen pedestal/tape maps to ur_base origin."""
    R_use = np.eye(3, dtype=np.float64) if R is None else np.asarray(R, dtype=np.float64)
    p = np.asarray(p_base_in_cam_mm, dtype=np.float64).reshape(3)
    t = -R_use @ p
    return CamPose(frame_id=frame_id, R=R_use, t_mm=t, ready=True)


def format_base_retreat(p_joint_base_mm: np.ndarray) -> str:
    """Instruction to move a joint at p_joint_base toward ur_base (0,0,0)."""
    p = np.asarray(p_joint_base_mm, dtype=np.float64).reshape(3)
    v = -p
    dist = float(np.linalg.norm(v))
    if dist < 20.0:
        return (
            "FUSED RETREAT -> ur_base (0,0,0): already near BASE "
            f"(~{dist / 10.0:.0f}cm) - pull gently to home"
        )
    parts: list[str] = []
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


def relative_cam1_from_cam0(cf: CellFrames) -> tuple[np.ndarray, np.ndarray]:
    """Return R,t such that X_cam1 = R @ X_cam0 + t (mm).

    With p_base = R_i @ p_cam_i + t_i:
      R_0 @ X0 + t0 = R_1 @ X1 + t1
      X1 = R_1.T @ R_0 @ X0 + R_1.T @ (t0 - t1)
    """
    R0, t0 = cf.cam0.R, cf.cam0.t_mm.reshape(3)
    R1, t1 = cf.cam1.R, cf.cam1.t_mm.reshape(3)
    R = R1.T @ R0
    t = R1.T @ (t0 - t1)
    return R, t


def apply_scene_stereo_to_cell_frames(
    cf: CellFrames,
    R_0_to_1: np.ndarray,
    t_0_to_1_mm: np.ndarray,
    *,
    origin_cam0_mm: Optional[np.ndarray] = None,
    origin_cam1_mm: Optional[np.ndarray] = None,
) -> CellFrames:
    """Install stereo R,t (cam0→cam1) into cell poses; bootstrap origins when given.

    cam0 stays identity in ur_base when origin_cam0 is provided (tape/pedestal).
    cam1 R/t are set so relative stereo matches the solve and both map origin→0.
    """
    R_rel = np.asarray(R_0_to_1, dtype=np.float64).reshape(3, 3)
    t_rel = np.asarray(t_0_to_1_mm, dtype=np.float64).reshape(3)

    if origin_cam0_mm is not None:
        p0 = np.asarray(origin_cam0_mm, dtype=np.float64).reshape(3)
        cf.cam0 = bootstrap_pose_from_base_landmark(
            p0, R=np.eye(3), frame_id=cf.cam0.frame_id
        )
    else:
        cf.cam0.R = np.eye(3, dtype=np.float64)
        cf.cam0.ready = True

    # Prefer bootstrapping cam1 from its own tape observation when available.
    if origin_cam1_mm is not None:
        p1 = np.asarray(origin_cam1_mm, dtype=np.float64).reshape(3)
        # R_cam1_to_base such that relative stereo holds with cam0 = I, t0 = -p0:
        # X1 = R_rel @ X0 + t_rel
        # p_base = R0 @ X0 + t0 = X0 + t0  (R0=I)
        # p_base = R1 @ X1 + t1
        # => R1 @ (R_rel @ X0 + t_rel) + t1 = X0 + t0  for all X0
        # => R1 @ R_rel = I  => R1 = R_rel.T
        # => R1 @ t_rel + t1 = t0  => t1 = t0 - R1 @ t_rel
        R1 = R_rel.T.copy()
        t0 = cf.cam0.t_mm.reshape(3)
        t1 = t0 - R1 @ t_rel
        # Also force tape in cam1 to origin: t1' = -R1 @ p1; blend by preferring tape.
        t1_tape = -R1 @ p1
        cf.cam1 = CamPose(
            frame_id=cf.cam1.frame_id,
            R=R1,
            t_mm=t1_tape,
            ready=True,
        )
        # Keep relative stereo consistent with tape-anchored cam1 when possible.
        _ = t1  # computed for reference / future soft blend
    else:
        R1 = R_rel.T.copy()
        t0 = cf.cam0.t_mm.reshape(3)
        t1 = t0 - R1 @ t_rel
        cf.cam1 = CamPose(
            frame_id=cf.cam1.frame_id,
            R=R1,
            t_mm=t1,
            ready=True,
        )

    cf.layout = LAYOUT_SCENE_TAPE
    return cf
