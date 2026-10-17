"""ROS2 vial detector node — offline dataset OR live Image subscription."""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional


def main(args: Optional[List[str]] = None) -> None:
    try:
        import rclpy
        from rclpy.node import Node
    except ImportError as exc:
        print(
            "rclpy is not available in this Python. Source ROS2 / RoboStack first. "
            f"Import error: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)

    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    from r2_05_aug_retrain.detector import VialYoloDetector, annotate, list_images
    from r2_05_aug_retrain.image_codec import bgr8_to_imgmsg, imgmsg_to_numpy
    from r2_05_aug_retrain.msgs_compat import (
        build_detection2d_array,
        create_detection_publishers,
        detections_to_json,
        vision_msgs_available,
    )

    class VialDetectorNode(Node):
        def __init__(self) -> None:
            super().__init__("vial_detector_node")
            self._declare()
            self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
            self.dataset_path = self.get_parameter("dataset_path").get_parameter_value().string_value
            self.run_offline = self.get_parameter("run_offline").get_parameter_value().bool_value
            self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
            self.camera_root = self.get_parameter("camera_root").get_parameter_value().string_value
            self.conf = self.get_parameter("confidence_threshold").get_parameter_value().double_value
            self.iou = self.get_parameter("iou_threshold").get_parameter_value().double_value
            self.imgsz = int(self.get_parameter("imgsz").get_parameter_value().integer_value)
            self.visualize = self.get_parameter("visualize").get_parameter_value().bool_value
            self.annotated_topic = (
                self.get_parameter("annotated_topic").get_parameter_value().string_value
            )
            self.detections_topic = (
                self.get_parameter("detections_topic").get_parameter_value().string_value
            )
            self.detections_json_topic = (
                self.get_parameter("detections_json_topic").get_parameter_value().string_value
            )
            self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
            self.offline_rate_hz = (
                self.get_parameter("offline_rate_hz").get_parameter_value().double_value
            )
            self.offline_loop = self.get_parameter("offline_loop").get_parameter_value().bool_value
            self.log_every_n = int(
                self.get_parameter("log_every_n").get_parameter_value().integer_value
            )

            self.detector = VialYoloDetector(
                self.model_path or None,
                camera_root=self.camera_root or None,
                confidence_threshold=self.conf,
                iou_threshold=self.iou,
                imgsz=self.imgsz,
            )
            self.vision_pub, self.json_pub, self.using_vision = create_detection_publishers(
                self, self.detections_topic, self.detections_json_topic
            )
            self.annotated_pub = self.create_publisher(Image, self.annotated_topic, 10)

            self._fps_times: Deque[float] = deque(maxlen=60)
            self._frame_i = 0
            self._offline_idx = 0
            self._offline_images: List[Path] = []

            self.get_logger().info(
                f"backend={self.detector.backend} model={self.detector.model_path} "
                f"conf={self.conf} vision_msgs={self.using_vision} "
                f"(available={vision_msgs_available()})"
            )

            if self.run_offline:
                self._start_offline()
            else:
                self.create_subscription(Image, self.image_topic, self._on_image, 10)
                self.get_logger().info(f"Subscribing to {self.image_topic}")

        def _declare(self) -> None:
            self.declare_parameter("image_topic", "/camera/color/image_raw")
            self.declare_parameter("dataset_path", "")
            self.declare_parameter("run_offline", False)
            self.declare_parameter("model_path", "")
            self.declare_parameter("camera_root", "")
            self.declare_parameter("confidence_threshold", 0.25)
            self.declare_parameter("iou_threshold", 0.45)
            self.declare_parameter("imgsz", 640)
            self.declare_parameter("visualize", True)
            self.declare_parameter("annotated_topic", "/vial/annotated")
            self.declare_parameter("detections_topic", "/vial/detections")
            self.declare_parameter("detections_json_topic", "/vial/detections_json")
            self.declare_parameter("frame_id", "camera_color_optical_frame")
            self.declare_parameter("offline_rate_hz", 5.0)
            self.declare_parameter("offline_loop", False)
            self.declare_parameter("log_every_n", 10)

        def _start_offline(self) -> None:
            folder = Path(self.dataset_path) if self.dataset_path else Path()
            self._offline_images = list_images(folder)
            if not self._offline_images:
                self.get_logger().error(f"No images in dataset_path={folder}")
                return
            period = 1.0 / self.offline_rate_hz if self.offline_rate_hz > 0 else 0.05
            self.create_timer(period, self._offline_tick)
            self.get_logger().info(
                f"Offline mode: {len(self._offline_images)} images from {folder}"
            )

        def _offline_tick(self) -> None:
            if not self._offline_images:
                return
            if self._offline_idx >= len(self._offline_images):
                if self.offline_loop:
                    self._offline_idx = 0
                else:
                    self.get_logger().info("Offline playback finished.")
                    return
            path = self._offline_images[self._offline_idx]
            self._offline_idx += 1
            import cv2

            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                return
            stamp = self.get_clock().now().to_msg()
            self._process(bgr, stamp)

        def _on_image(self, msg: Image) -> None:
            bgr = imgmsg_to_numpy(msg)
            self._process(bgr, msg.header.stamp)

        def _process(self, bgr, stamp) -> None:
            import cv2

            dets, stats = self.detector.predict(bgr)
            now = time.perf_counter()
            self._fps_times.append(now)
            self._frame_i += 1

            sec = int(getattr(stamp, "sec", 0))
            nsec = int(getattr(stamp, "nanosec", 0))
            js = detections_to_json(
                dets,
                frame_id=self.frame_id,
                stamp_sec=sec,
                stamp_nanosec=nsec,
                backend=stats.backend,
                inference_ms=stats.inference_ms,
            )
            self.json_pub.publish(String(data=js))
            if self.using_vision and self.vision_pub is not None:
                self.vision_pub.publish(
                    build_detection2d_array(dets, frame_id=self.frame_id, stamp=stamp)
                )
            if self.visualize:
                ann = annotate(bgr, dets)
                self.annotated_pub.publish(
                    bgr8_to_imgmsg(ann, frame_id=self.frame_id, stamp=stamp)
                )

            if self._frame_i % max(1, self.log_every_n) == 0:
                fps = 0.0
                if len(self._fps_times) >= 2:
                    dt = self._fps_times[-1] - self._fps_times[0]
                    if dt > 0:
                        fps = (len(self._fps_times) - 1) / dt
                self.get_logger().info(
                    f"frame={self._frame_i} dets={len(dets)} "
                    f"infer_ms={stats.inference_ms:.1f} fps~{fps:.1f}"
                )

    rclpy.init(args=args)
    node = VialDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
