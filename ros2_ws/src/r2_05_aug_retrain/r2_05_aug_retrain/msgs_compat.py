"""Publish Detection2DArray when vision_msgs is installed; else JSON String."""

from __future__ import annotations

import json
from typing import Optional, Sequence

from .detector import Det, det_to_dict

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
    dets: Sequence[Det],
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
        "detections": [det_to_dict(d) for d in dets],
    }
    return json.dumps(payload)


def build_detection2d_array(dets: Sequence[Det], *, frame_id: str, stamp) -> object:
    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

    arr = Detection2DArray()
    arr.header.stamp = stamp
    arr.header.frame_id = frame_id
    for d in dets:
        det = Detection2D()
        det.header.stamp = stamp
        det.header.frame_id = frame_id
        cx = 0.5 * (d.x1 + d.x2)
        cy = 0.5 * (d.y1 + d.y2)
        bw = max(0.0, d.x2 - d.x1)
        bh = max(0.0, d.y2 - d.y1)
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
        bbox.size_x = float(bw)
        bbox.size_y = float(bh)
        hyp = ObjectHypothesisWithPose()
        if hasattr(hyp, "hypothesis"):
            hyp.hypothesis.class_id = "vial"
            hyp.hypothesis.score = float(d.conf)
        else:
            hyp.id = "vial"
            hyp.score = float(d.conf)
        det.results.append(hyp)
        arr.detections.append(det)
    return arr


def create_detection_publishers(node, detections_topic: str, json_topic: str):
    from std_msgs.msg import String

    json_pub = node.create_publisher(String, json_topic, 10)
    vision_pub = None
    using_vision = False
    if vision_msgs_available():
        from vision_msgs.msg import Detection2DArray

        vision_pub = node.create_publisher(Detection2DArray, detections_topic, 10)
        using_vision = True
    return vision_pub, json_pub, using_vision
