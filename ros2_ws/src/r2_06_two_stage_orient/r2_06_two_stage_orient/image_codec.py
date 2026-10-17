"""Minimal sensor_msgs/Image ↔ numpy without cv_bridge."""

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
        gray = raw.reshape(height, width).copy()
        return np.stack([gray, gray, gray], axis=-1)
    if encoding in ("16uc1", "mono16"):
        # Depth as millimeters — expand to 3ch for annotate path only.
        depth = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(height, width)
        # Caller should use depth_imgmsg_to_mm for metric depth.
        g = np.clip(depth.astype(np.float32) / 16.0, 0, 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)
    raise ValueError(f"Unsupported encoding: {msg.encoding!r}")


def depth_imgmsg_to_mm(msg) -> np.ndarray:
    """Convert depth Image to float32 millimeters."""
    encoding = (msg.encoding or "").lower()
    height = int(msg.height)
    width = int(msg.width)
    if encoding in ("16uc1", "mono16"):
        depth = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(height, width)
        return depth.astype(np.float32)
    if encoding in ("32fc1",):
        depth = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(height, width)
        # Heuristic: values < 20 treated as meters → mm.
        if float(np.nanmedian(depth[depth > 0])) < 20.0:
            return depth * 1000.0
        return depth
    raise ValueError(f"Unsupported depth encoding: {msg.encoding!r}")
