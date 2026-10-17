"""Publish Detection2DArray (theta = OBB angle) or JSON String fallback."""

from __future__ import annotations

import json
from typing import Optional, Sequence

from .obb_geometry import OBBDet

_VISION_OK: Optional[bool] = None


def vision_msgs_available() -> bool:
    global _VISION_OK
    if _VISION_OK is not None:
        return _VISION_OK
    try:
        from vision_msgs.msg import Detection2DArray  # noqa: F401

        _VISION_OK = True
    except Exception:  # noqa: BLE001
        _VISION_OK = False
    return _VISION_OK


def detections_to_json(
    dets: Sequence[OBBDet],
    *,
    frame_id: str,
    stamp_sec: int,
    stamp_nanosec: int,
    backend: str,
    inference_ms: float,
) -> str:
    payload = {
        "header": {
            "frame_id": frame_id,
            "stamp": {"sec": int(stamp_sec), "nanosec": int(stamp_nanosec)},
        },
        "backend": backend,
        "inference_ms": float(inference_ms),
        "box_type": "obb",
        "detections": [d.to_dict() for d in dets],
    }
    return json.dumps(payload)


def build_detection2d_array(
    dets: Sequence[OBBDet],
    *,
    frame_id: str,
    stamp,
) -> object:
    """Build vision_msgs/Detection2DArray with bbox.center.theta = angle_rad."""
    import math

    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

    arr = Detection2DArray()
    arr.header.stamp = stamp
    arr.header.frame_id = frame_id
    for d in dets:
        det = Detection2D()
        det.header.stamp = stamp
        det.header.frame_id = frame_id
        bbox = det.bbox
        theta = math.radians(float(d.angle_deg))
        if hasattr(bbox, "center") and hasattr(bbox.center, "position"):
            bbox.center.position.x = float(d.cx)
            bbox.center.position.y = float(d.cy)
            if hasattr(bbox.center, "theta"):
                bbox.center.theta = theta
        elif hasattr(bbox, "center"):
            bbox.center.x = float(d.cx)
            bbox.center.y = float(d.cy)
            if hasattr(bbox.center, "theta"):
                bbox.center.theta = theta
        bbox.size_x = float(d.width)
        bbox.size_y = float(d.height)
        hyp = ObjectHypothesisWithPose()
        if hasattr(hyp, "hypothesis"):
            hyp.hypothesis.class_id = str(d.cls_id)
            hyp.hypothesis.score = float(d.conf)
        else:
            if hasattr(hyp, "id"):
                hyp.id = str(d.cls_id)
            if hasattr(hyp, "score"):
                hyp.score = float(d.conf)
        det.results.append(hyp)
        if hasattr(det, "id"):
            det.id = f"{d.label}:{d.orientation}"
        arr.detections.append(det)
    return arr


def create_detection_publishers(node, detections_topic: str, detections_json_topic: str):
    from std_msgs.msg import String

    json_pub = node.create_publisher(String, detections_json_topic, 10)
    vision_pub = None
    using_vision = False
    if vision_msgs_available():
        from vision_msgs.msg import Detection2DArray

        vision_pub = node.create_publisher(Detection2DArray, detections_topic, 10)
        using_vision = True
    return vision_pub, json_pub, using_vision
