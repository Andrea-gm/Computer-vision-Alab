"""Publish Detection2DArray / PoseArray with graceful fallbacks.

RoboStack base install may lack ``vision_msgs`` / ``geometry_msgs``. When those
are missing we publish:
- ``std_msgs/Float32MultiArray`` on ``~/detections2d`` as
  ``[x, y, w, h, score, cx, cy, cz, ax, ay, az, radius, height, orient_code] * N``
  where orient_code is 0=sideways, 1=upright.
- annotated ``sensor_msgs/Image`` always.
"""

from __future__ import annotations

from typing import Any, Iterable

from agent_08_depth_geometry.geometry_detector import VialDetection

_HAS_VISION = False
_HAS_GEOMETRY = False
try:
    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

    _HAS_VISION = True
except Exception:  # pragma: no cover - optional ROS dep
    Detection2D = Detection2DArray = ObjectHypothesisWithPose = None  # type: ignore

try:
    from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion

    _HAS_GEOMETRY = True
except Exception:  # pragma: no cover
    Pose = PoseArray = Point = Quaternion = None  # type: ignore


def has_vision_msgs() -> bool:
    return _HAS_VISION


def has_geometry_msgs() -> bool:
    return _HAS_GEOMETRY


def _axis_to_quat(axis_xyz: tuple[float, float, float]):
    """Map cylinder axis to a quaternion (Z-axis of pose = vial axis)."""
    import math

    ax, ay, az = axis_xyz
    n = math.sqrt(ax * ax + ay * ay + az * az) + 1e-12
    ax, ay, az = ax / n, ay / n, az / n
    # Rotate +Z onto axis via half-angle formula with cross(+Z, axis).
    # +Z = (0,0,1)
    cx, cy, cz = -ay, ax, 0.0  # cross(z, axis)
    dot = az  # z·axis
    if dot < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)
    qx, qy, qz = cx, cy, cz
    qw = 1.0 + dot
    nq = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) + 1e-12
    return (qx / nq, qy / nq, qz / nq, qw / nq)


def detections_to_float_array(dets: Iterable[VialDetection]) -> list[float]:
    out: list[float] = []
    for d in dets:
        x, y, w, h = d.box_xywh
        orient = 1.0 if d.orientation == "upright" else 0.0
        out.extend(
            [
                float(x),
                float(y),
                float(w),
                float(h),
                float(d.score),
                float(d.center_xyz_mm[0]),
                float(d.center_xyz_mm[1]),
                float(d.center_xyz_mm[2]),
                float(d.axis_xyz[0]),
                float(d.axis_xyz[1]),
                float(d.axis_xyz[2]),
                float(d.radius_mm),
                float(d.height_mm),
                orient,
            ]
        )
    return out


def build_detection2d_array(dets: list[VialDetection], *, stamp, frame_id: str) -> Any:
    if not _HAS_VISION:
        return None
    arr = Detection2DArray()
    arr.header.stamp = stamp
    arr.header.frame_id = frame_id
    for d in dets:
        det = Detection2D()
        det.header.stamp = stamp
        det.header.frame_id = frame_id
        x, y, w, h = d.box_xywh
        det.bbox.center.position.x = float(x) + 0.5 * float(w)
        det.bbox.center.position.y = float(y) + 0.5 * float(h)
        det.bbox.size_x = float(w)
        det.bbox.size_y = float(h)
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = f"vial_{d.orientation}"
        hyp.hypothesis.score = float(d.score)
        if _HAS_GEOMETRY:
            qx, qy, qz, qw = _axis_to_quat(d.axis_xyz)
            hyp.pose.pose.position.x = d.center_xyz_mm[0] * 0.001
            hyp.pose.pose.position.y = d.center_xyz_mm[1] * 0.001
            hyp.pose.pose.position.z = d.center_xyz_mm[2] * 0.001
            hyp.pose.pose.orientation.x = qx
            hyp.pose.pose.orientation.y = qy
            hyp.pose.pose.orientation.z = qz
            hyp.pose.pose.orientation.w = qw
        det.results.append(hyp)
        arr.detections.append(det)
    return arr


def build_pose_array(dets: list[VialDetection], *, stamp, frame_id: str) -> Any:
    if not _HAS_GEOMETRY:
        return None
    arr = PoseArray()
    arr.header.stamp = stamp
    arr.header.frame_id = frame_id
    for d in dets:
        pose = Pose()
        pose.position = Point(
            x=d.center_xyz_mm[0] * 0.001,
            y=d.center_xyz_mm[1] * 0.001,
            z=d.center_xyz_mm[2] * 0.001,
        )
        qx, qy, qz, qw = _axis_to_quat(d.axis_xyz)
        pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        arr.poses.append(pose)
    return arr
