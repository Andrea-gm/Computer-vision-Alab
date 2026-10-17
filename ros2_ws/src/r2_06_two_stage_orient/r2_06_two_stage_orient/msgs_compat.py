"""Publish Detection2DArray when vision_msgs is installed; else JSON String."""

from __future__ import annotations

import json
from typing import List, Optional, Sequence

from .types import OrientedDet

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
    dets: Sequence[OrientedDet],
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
        "detections": [d.to_dict() for d in dets],
    }
    return json.dumps(payload)


def build_detection2d_array(
    dets: Sequence[OrientedDet],
    *,
    frame_id: str,
    stamp,
) -> object:
    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

    arr = Detection2DArray()
    arr.header.stamp = stamp
    arr.header.frame_id = frame_id
    for d in dets:
        det = Detection2D()
        det.header.stamp = stamp
        det.header.frame_id = frame_id
        cx = d.x + 0.5 * d.w
        cy = d.y + 0.5 * d.h
        bbox = det.bbox
        if hasattr(bbox, "center") and hasattr(bbox.center, "position"):
            bbox.center.position.x = float(cx)
            bbox.center.position.y = float(cy)
            if hasattr(bbox.center, "theta"):
                bbox.center.theta = 0.0
        elif hasattr(bbox, "center"):
            bbox.center.x = float(cx)
            bbox.center.y = float(cy)
            if hasattr(bbox.center, "theta"):
                bbox.center.theta = 0.0
        if hasattr(bbox, "size_x"):
            bbox.size_x = float(d.w)
            bbox.size_y = float(d.h)
        hyp = ObjectHypothesisWithPose()
        if hasattr(hyp, "hypothesis"):
            hyp.hypothesis.class_id = f"vial_{d.orientation}"
            hyp.hypothesis.score = float(d.conf)
        else:
            hyp.id = f"vial_{d.orientation}"
            hyp.score = float(d.conf)
        det.results.append(hyp)
        arr.detections.append(det)
    return arr


def create_detection_publishers(node, detections_topic: str, detections_json_topic: str):
    from std_msgs.msg import String

    json_pub = node.create_publisher(String, detections_json_topic, 10)
    vision_pub = None
    using_vision = vision_msgs_available()
    if using_vision:
        from vision_msgs.msg import Detection2DArray

        vision_pub = node.create_publisher(Detection2DArray, detections_topic, 10)
    return vision_pub, json_pub, using_vision
