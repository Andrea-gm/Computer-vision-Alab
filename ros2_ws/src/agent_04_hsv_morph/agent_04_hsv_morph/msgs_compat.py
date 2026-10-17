"""Publish Detection2DArray when vision_msgs is installed; else JSON String.

Documented fallback: many Windows RoboStack installs lack vision_msgs.
Downstream nodes can subscribe to ``detections_json_topic`` (std_msgs/String)
with the same schema either way.
"""

from __future__ import annotations

import json
from typing import List, Optional, Sequence, Tuple

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


def build_detection2d_array(
    dets: Sequence[Det],
    *,
    frame_id: str,
    stamp,
) -> object:
    """Build vision_msgs/Detection2DArray. Raises if vision_msgs missing."""
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
            hyp.hypothesis.class_id = str(d.cls_id)
            hyp.hypothesis.score = float(d.conf)
        else:
            if hasattr(hyp, "id"):
                hyp.id = str(d.cls_id)
            if hasattr(hyp, "score"):
                hyp.score = float(d.conf)
        det.results.append(hyp)
        if hasattr(det, "id"):
            det.id = d.label
        arr.detections.append(det)
    return arr


def create_detection_publishers(node, detections_topic: str, json_topic: str) -> Tuple[object, object, bool]:
    """Return (vision_pub_or_None, json_pub, using_vision)."""
    from std_msgs.msg import String

    json_pub = node.create_publisher(String, json_topic, 10)
    if vision_msgs_available():
        from vision_msgs.msg import Detection2DArray

        vision_pub = node.create_publisher(Detection2DArray, detections_topic, 10)
        return vision_pub, json_pub, True
    return None, json_pub, False
