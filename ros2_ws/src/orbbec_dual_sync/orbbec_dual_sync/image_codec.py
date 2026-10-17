"""Minimal sensor_msgs/Image <-> numpy without cv_bridge."""

from __future__ import annotations

import numpy as np


def bgr8_to_imgmsg(image: np.ndarray, *, frame_id: str, stamp) -> "object":
    from sensor_msgs.msg import Image

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR uint8, got {image.shape}")
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(image.shape[0])
    msg.width = int(image.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = int(msg.width * 3)
    msg.data = np.ascontiguousarray(image, dtype=np.uint8).tobytes()
    return msg


def depth_mm_to_imgmsg(depth_mm: np.ndarray, *, frame_id: str, stamp) -> "object":
    from sensor_msgs.msg import Image

    # 16UC1 is the standard ROS millimetre representation and halves DDS
    # traffic versus 32FC1. Invalid/non-finite pixels become zero.
    finite = np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.ascontiguousarray(np.clip(finite, 0.0, 65535.0), dtype=np.uint16)
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth, got {depth.shape}")
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(depth.shape[0])
    msg.width = int(depth.shape[1])
    msg.encoding = "16UC1"
    msg.is_bigendian = 0
    msg.step = int(msg.width * 2)
    msg.data = depth.tobytes()
    return msg


def imgmsg_to_numpy(msg) -> np.ndarray:
    encoding = (msg.encoding or "").lower()
    height = int(msg.height)
    width = int(msg.width)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if encoding in ("bgr8", "rgb8"):
        image = raw.reshape(height, width, 3)
        if encoding == "rgb8":
            image = image[:, :, ::-1].copy()
        else:
            image = image.copy()
        return image
    if encoding in ("mono8", "8uc1"):
        return raw.reshape(height, width).copy()
    if encoding in ("32fc1",):
        depth = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(height, width)
        return depth.copy()
    if encoding in ("16uc1", "mono16"):
        depth = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(height, width)
        return depth.astype(np.float32)
    raise ValueError(f"Unsupported encoding: {msg.encoding!r}")
