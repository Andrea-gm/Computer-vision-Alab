"""ApproximateTime-sync two color streams; republish matched pairs (+ depths).

Use this when you only want stamped synced topics / saved frames, without YOLO.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image

from .approx_time import stamp_msg_to_seconds
from .image_codec import imgmsg_to_numpy


class SyncForwardNode(Node):
    def __init__(self) -> None:
        super().__init__("orbbec_sync_forward")
        self.declare_parameter("slop_s", 0.05)
        self.declare_parameter("queue_size", 10)
        self.declare_parameter("cam0_color_topic", "/orbbec/cam0/color/image_raw")
        self.declare_parameter("cam1_color_topic", "/orbbec/cam1/color/image_raw")
        self.declare_parameter("cam0_depth_topic", "/orbbec/cam0/depth/image_raw")
        self.declare_parameter("cam1_depth_topic", "/orbbec/cam1/depth/image_raw")
        self.declare_parameter("synced_cam0_color_topic", "/orbbec/synced/cam0/color")
        self.declare_parameter("synced_cam1_color_topic", "/orbbec/synced/cam1/color")
        self.declare_parameter("synced_cam0_depth_topic", "/orbbec/synced/cam0/depth")
        self.declare_parameter("synced_cam1_depth_topic", "/orbbec/synced/cam1/depth")
        self.declare_parameter("save_synced_dir", "")
        self.declare_parameter("use_depth_in_sync", True)

        slop = float(self.get_parameter("slop_s").value)
        qsize = int(self.get_parameter("queue_size").value)
        self._save_dir = str(self.get_parameter("save_synced_dir").value).strip()
        self._use_depth = bool(self.get_parameter("use_depth_in_sync").value)
        self._pair_i = 0

        c0 = str(self.get_parameter("cam0_color_topic").value)
        c1 = str(self.get_parameter("cam1_color_topic").value)
        d0 = str(self.get_parameter("cam0_depth_topic").value)
        d1 = str(self.get_parameter("cam1_depth_topic").value)

        self.pub_c0 = self.create_publisher(
            Image, str(self.get_parameter("synced_cam0_color_topic").value), 10
        )
        self.pub_c1 = self.create_publisher(
            Image, str(self.get_parameter("synced_cam1_color_topic").value), 10
        )
        self.pub_d0 = self.create_publisher(
            Image, str(self.get_parameter("synced_cam0_depth_topic").value), 10
        )
        self.pub_d1 = self.create_publisher(
            Image, str(self.get_parameter("synced_cam1_depth_topic").value), 10
        )

        self._sub_c0 = Subscriber(self, Image, c0)
        self._sub_c1 = Subscriber(self, Image, c1)
        if self._use_depth:
            self._sub_d0 = Subscriber(self, Image, d0)
            self._sub_d1 = Subscriber(self, Image, d1)
            self._sync = ApproximateTimeSynchronizer(
                [self._sub_c0, self._sub_d0, self._sub_c1, self._sub_d1],
                qsize,
                slop,
            )
            self._sync.registerCallback(self._on_four)
        else:
            self._sync = ApproximateTimeSynchronizer(
                [self._sub_c0, self._sub_c1], qsize, slop
            )
            self._sync.registerCallback(self._on_two)

        if self._save_dir:
            Path(self._save_dir).mkdir(parents=True, exist_ok=True)

        self.get_logger().info(
            f"ApproximateTime sync slop={slop*1000:.0f}ms queue={qsize} "
            f"depth_in_sync={self._use_depth}"
        )

    def _on_two(self, color0: Image, color1: Image) -> None:
        self._emit(color0, None, color1, None)

    def _on_four(
        self, color0: Image, depth0: Image, color1: Image, depth1: Image
    ) -> None:
        self._emit(color0, depth0, color1, depth1)

    def _emit(
        self,
        color0: Image,
        depth0: Image | None,
        color1: Image,
        depth1: Image | None,
    ) -> None:
        self.pub_c0.publish(color0)
        self.pub_c1.publish(color1)
        if depth0 is not None:
            self.pub_d0.publish(depth0)
        if depth1 is not None:
            self.pub_d1.publish(depth1)
        self._pair_i += 1
        dt_ms = abs(stamp_msg_to_seconds(color0.header.stamp) - stamp_msg_to_seconds(color1.header.stamp)) * 1000.0
        if self._pair_i % 30 == 0:
            self.get_logger().info(f"synced pairs={self._pair_i} last_dt={dt_ms:.1f}ms")
        if self._save_dir:
            self._save(color0, color1)

    def _save(self, color0: Image, color1: Image) -> None:
        try:
            a = imgmsg_to_numpy(color0)
            b = imgmsg_to_numpy(color1)
        except Exception as exc:
            self.get_logger().warn(f"save decode failed: {exc}")
            return
        root = Path(self._save_dir)
        stem = f"pair_{self._pair_i:06d}"
        cv2.imwrite(str(root / f"{stem}_cam0.jpg"), a)
        cv2.imwrite(str(root / f"{stem}_cam1.jpg"), b)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SyncForwardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
