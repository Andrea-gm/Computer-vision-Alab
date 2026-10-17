"""Minimal sensor_msgs/Image <-> numpy without cv_bridge."""

from __future__ import annotations

import numpy as np


def bgr8_to_imgmsg(image: np.ndarray, *, frame_id: str, stamp) -> object:
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


def camera_info_to_intrinsics(msg) -> dict[str, float]:
    k = list(msg.k) if msg.k is not None else []
    if len(k) >= 6 and float(k[0]) > 1.0:
        return {
            "fx": float(k[0]),
            "fy": float(k[4]),
            "cx": float(k[2]),
            "cy": float(k[5]),
            "width": float(msg.width),
            "height": float(msg.height),
        }
    return {
        "fx": 0.0,
        "fy": 0.0,
        "cx": float(msg.width) * 0.5,
        "cy": float(msg.height) * 0.5,
        "width": float(msg.width),
        "height": float(msg.height),
    }
