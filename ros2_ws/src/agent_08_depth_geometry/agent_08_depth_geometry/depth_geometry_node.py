"""ROS2 node: RGB (+ optional depth + camera_info) → geometric vial detections.

Topics (defaults match Orbbec dual-sync publishers):
  ~/color          sensor_msgs/Image        (subscribe; remappable)
  ~/depth          sensor_msgs/Image        (subscribe; optional)
  ~/camera_info    sensor_msgs/CameraInfo   (subscribe; optional)

  ~/detections2d          vision_msgs/Detection2DArray  OR Float32MultiArray fallback
  ~/poses                 geometry_msgs/PoseArray       (optional)
  ~/detections_flat       std_msgs/Float32MultiArray    (always)
  ~/annotated             sensor_msgs/Image
"""

from __future__ import annotations

from typing import Any

import numpy as np

from agent_08_depth_geometry.geometry_detector import annotate, detect_vials
from agent_08_depth_geometry.image_codec import (
    bgr8_to_imgmsg,
    camera_info_to_intrinsics,
    imgmsg_to_numpy,
)
from agent_08_depth_geometry.ros_publish import (
    build_detection2d_array,
    build_pose_array,
    detections_to_float_array,
    has_geometry_msgs,
    has_vision_msgs,
)


class DepthGeometryNode:
    """Constructed after ``rclpy.init``; wraps an ``rclpy.node.Node``."""

    def __init__(self, node: Any):
        self.node = node
        n = node
        n.declare_parameter("color_topic", "/orbbec/cam0/color/image_raw")
        n.declare_parameter("depth_topic", "/orbbec/cam0/depth/image_raw")
        n.declare_parameter("camera_info_topic", "/orbbec/cam0/color/camera_info")
        n.declare_parameter("frame_id", "orbbec_cam0_color")
        n.declare_parameter("allow_pseudo_depth", True)
        n.declare_parameter("min_score", 0.25)
        n.declare_parameter("sync_slop_sec", 0.05)
        n.declare_parameter("require_depth", False)

        self._allow_pseudo = bool(n.get_parameter("allow_pseudo_depth").value)
        self._min_score = float(n.get_parameter("min_score").value)
        self._require_depth = bool(n.get_parameter("require_depth").value)
        self._frame_id = str(n.get_parameter("frame_id").value)
        self._intrinsics: dict | None = None
        self._latest_depth = None
        self._latest_depth_stamp = None

        color_topic = str(n.get_parameter("color_topic").value)
        depth_topic = str(n.get_parameter("depth_topic").value)
        info_topic = str(n.get_parameter("camera_info_topic").value)

        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import Float32MultiArray

        self._pub_flat = n.create_publisher(Float32MultiArray, "~/detections_flat", 10)
        self._pub_annot = n.create_publisher(Image, "~/annotated", 10)

        self._pub_det2d = None
        if has_vision_msgs():
            from vision_msgs.msg import Detection2DArray

            self._pub_det2d = n.create_publisher(Detection2DArray, "~/detections2d", 10)
            n.get_logger().info("Publishing vision_msgs/Detection2DArray on ~/detections2d")
        else:
            self._pub_det2d = n.create_publisher(Float32MultiArray, "~/detections2d", 10)
            n.get_logger().warn(
                "vision_msgs not installed — ~/detections2d is Float32MultiArray "
                "(same layout as ~/detections_flat)"
            )

        self._pub_poses = None
        if has_geometry_msgs():
            from geometry_msgs.msg import PoseArray

            self._pub_poses = n.create_publisher(PoseArray, "~/poses", 10)
            n.get_logger().info("Publishing geometry_msgs/PoseArray on ~/poses")
        else:
            n.get_logger().warn("geometry_msgs not installed — ~/poses disabled")

        n.create_subscription(CameraInfo, info_topic, self._on_info, 10)
        n.create_subscription(Image, depth_topic, self._on_depth, 10)

        # Prefer ApproximateTime sync when message_filters is available.
        self._use_sync = False
        try:
            from message_filters import ApproximateTimeSynchronizer, Subscriber

            self._color_sub = Subscriber(n, Image, color_topic)
            self._depth_sub = Subscriber(n, Image, depth_topic)
            slop = float(n.get_parameter("sync_slop_sec").value)
            self._sync = ApproximateTimeSynchronizer(
                [self._color_sub, self._depth_sub], queue_size=8, slop=slop
            )
            self._sync.registerCallback(self._on_synced)
            self._use_sync = True
            n.get_logger().info(
                f"ApproximateTime sync color={color_topic} depth={depth_topic} slop={slop}s"
            )
        except Exception as exc:
            n.get_logger().warn(f"message_filters sync unavailable ({exc}); color-only path")
            n.create_subscription(Image, color_topic, self._on_color_only, 10)

        # Always also listen to color for RGB-only / late depth cases.
        if self._use_sync:
            n.create_subscription(Image, color_topic, self._on_color_fallback, 10)

        n.get_logger().info(
            f"agent_08_depth_geometry ready | real-depth preferred | "
            f"pseudo_depth={self._allow_pseudo} require_depth={self._require_depth}"
        )

    def _on_info(self, msg) -> None:
        self._intrinsics = camera_info_to_intrinsics(msg)

    def _on_depth(self, msg) -> None:
        try:
            self._latest_depth = imgmsg_to_numpy(msg)
            self._latest_depth_stamp = msg.header.stamp
        except Exception as exc:
            self.node.get_logger().warn(f"depth decode failed: {exc}")

    def _on_synced(self, color_msg, depth_msg) -> None:
        try:
            color = imgmsg_to_numpy(color_msg)
            depth = imgmsg_to_numpy(depth_msg)
        except Exception as exc:
            self.node.get_logger().warn(f"sync decode failed: {exc}")
            return
        self._latest_depth = depth
        self._publish(color, depth, color_msg.header.stamp, color_msg.header.frame_id)

    def _on_color_only(self, color_msg) -> None:
        try:
            color = imgmsg_to_numpy(color_msg)
        except Exception as exc:
            self.node.get_logger().warn(f"color decode failed: {exc}")
            return
        depth = self._latest_depth
        if depth is None and self._require_depth:
            return
        self._publish(color, depth, color_msg.header.stamp, color_msg.header.frame_id)

    def _on_color_fallback(self, color_msg) -> None:
        """If sync is quiet, still process color with latest depth / pseudo."""
        # Avoid double-fire when sync is healthy: only run if no recent depth stamp match.
        if self._latest_depth is None:
            if self._require_depth:
                return
            try:
                color = imgmsg_to_numpy(color_msg)
            except Exception:
                return
            self._publish(color, None, color_msg.header.stamp, color_msg.header.frame_id)

    def _publish(self, color, depth, stamp, frame_id: str) -> None:
        frame = frame_id or self._frame_id
        dets = detect_vials(
            color,
            depth,
            self._intrinsics,
            allow_pseudo_depth=self._allow_pseudo and not self._require_depth,
            min_score=self._min_score,
        )
        flat = detections_to_float_array(dets)
        from std_msgs.msg import Float32MultiArray

        flat_msg = Float32MultiArray()
        flat_msg.data = flat
        self._pub_flat.publish(flat_msg)

        if self._pub_det2d is not None:
            if has_vision_msgs():
                msg = build_detection2d_array(dets, stamp=stamp, frame_id=frame)
                if msg is not None:
                    self._pub_det2d.publish(msg)
            else:
                fb = Float32MultiArray()
                fb.data = flat
                self._pub_det2d.publish(fb)

        if self._pub_poses is not None:
            poses = build_pose_array(dets, stamp=stamp, frame_id=frame)
            if poses is not None:
                self._pub_poses.publish(poses)

        annotated = annotate(color, dets)
        self._pub_annot.publish(
            bgr8_to_imgmsg(annotated, frame_id=frame, stamp=stamp)
        )


def main(args=None) -> None:
    import rclpy
    from rclpy.node import Node

    rclpy.init(args=args)
    node = Node("agent_08_depth_geometry")
    DepthGeometryNode(node)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
