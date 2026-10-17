"""ROS2 vial detector node — offline dataset OR live Image subscription.

Publishes:
  - vision_msgs/Detection2DArray on detections_topic (if vision_msgs installed)
  - std_msgs/String JSON on detections_json_topic (always)
  - sensor_msgs/Image annotated frame on annotated_topic (when visualize=true)
"""

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
            "rclpy is not available in this Python. On Linux: source /opt/ros/$ROS_DISTRO/setup.bash. "
            "On Windows RoboStack: conda activate the ros env, then source local_setup. "
            f"Import error: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)

    import cv2
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    from r2_04_cap_body_cascade.cascade_detector import (
        CapBodyCascadeDetector,
        CascadeParams,
        list_images,
    )
    from r2_04_cap_body_cascade.image_codec import bgr8_to_imgmsg, imgmsg_to_numpy
    from r2_04_cap_body_cascade.msgs_compat import (
        build_detection2d_array,
        create_detection_publishers,
        detections_to_json,
        vision_msgs_available,
    )

    class VialDetectorNode(Node):
        def __init__(self) -> None:
            super().__init__("vial_detector_node")
            self._declare()
            g = self.get_parameter
            self.image_topic = g("image_topic").get_parameter_value().string_value
            self.dataset_path = g("dataset_path").get_parameter_value().string_value
            self.run_offline = g("run_offline").get_parameter_value().bool_value
            self.visualize = g("visualize").get_parameter_value().bool_value
            self.annotated_topic = g("annotated_topic").get_parameter_value().string_value
            self.detections_topic = g("detections_topic").get_parameter_value().string_value
            self.detections_json_topic = (
                g("detections_json_topic").get_parameter_value().string_value
            )
            self.frame_id = g("frame_id").get_parameter_value().string_value
            self.offline_rate_hz = g("offline_rate_hz").get_parameter_value().double_value
            self.offline_loop = g("offline_loop").get_parameter_value().bool_value
            self.log_every_n = int(g("log_every_n").get_parameter_value().integer_value)

            params = CascadeParams(
                confidence_threshold=g("confidence_threshold").get_parameter_value().double_value,
                max_detections=int(g("max_detections").get_parameter_value().integer_value),
                white_s_max=int(g("white_s_max").get_parameter_value().integer_value),
                white_v_min=int(g("white_v_min").get_parameter_value().integer_value),
                nms_iou=g("nms_iou").get_parameter_value().double_value,
            )
            self.detector = CapBodyCascadeDetector(params)
            self.vision_pub, self.json_pub, self.using_vision = create_detection_publishers(
                self, self.detections_topic, self.detections_json_topic
            )
            self.annotated_pub = self.create_publisher(Image, self.annotated_topic, 10)

            self._fps_times: Deque[float] = deque(maxlen=60)
            self._frame_i = 0
            self._offline_idx = 0
            self._offline_images: List[Path] = []
            self._offline_timer = None

            self.get_logger().info(
                f"backend={self.detector.backend} conf={params.confidence_threshold} "
                f"vision_msgs={self.using_vision} (available={vision_msgs_available()})"
            )
            if self.run_offline:
                self._start_offline()
            else:
                self.create_subscription(Image, self.image_topic, self._on_image, 10)
                self.get_logger().info(f"Subscribing to {self.image_topic}")

        def _declare(self) -> None:
            defaults = {
                "image_topic": "/camera/color/image_raw",
                "dataset_path": "E:/Desktop/Jessica/robotics/vial_dataset/split_70_10_20/images/test",
                "run_offline": True,
                "visualize": True,
                "annotated_topic": "/vial/annotated",
                "detections_topic": "/vial/detections",
                "detections_json_topic": "/vial/detections_json",
                "frame_id": "camera_color_optical_frame",
                "offline_rate_hz": 5.0,
                "offline_loop": False,
                "log_every_n": 10,
                "confidence_threshold": 0.45,
                "max_detections": 3,
                "white_s_max": 60,
                "white_v_min": 175,
                "nms_iou": 0.35,
            }
            for k, v in defaults.items():
                self.declare_parameter(k, v)

        def _start_offline(self) -> None:
            folder = Path(self.dataset_path)
            self._offline_images = list_images(folder)
            if not self._offline_images:
                self.get_logger().error(f"No images under {folder}")
                return
            period = 1.0 / max(0.1, float(self.offline_rate_hz))
            self._offline_timer = self.create_timer(period, self._offline_tick)
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
                    self.get_logger().info("Offline dataset complete.")
                    if self._offline_timer is not None:
                        self._offline_timer.cancel()
                    return
            path = self._offline_images[self._offline_idx]
            self._offline_idx += 1
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                return
            self._process(bgr, stamp=self.get_clock().now().to_msg())

        def _on_image(self, msg: Image) -> None:
            try:
                bgr = imgmsg_to_numpy(msg)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"decode failed: {exc}")
                return
            self._process(bgr, stamp=msg.header.stamp)

        def _process(self, bgr, stamp) -> None:
            t0 = time.perf_counter()
            dets, stats = self.detector.predict(bgr)
            self._fps_times.append(time.perf_counter())
            self._frame_i += 1
            fps = 0.0
            if len(self._fps_times) >= 2:
                dt = self._fps_times[-1] - self._fps_times[0]
                if dt > 0:
                    fps = (len(self._fps_times) - 1) / dt

            if self.using_vision and self.vision_pub is not None:
                self.vision_pub.publish(
                    build_detection2d_array(dets, frame_id=self.frame_id, stamp=stamp)
                )
            payload = detections_to_json(
                dets,
                frame_id=self.frame_id,
                stamp_sec=int(getattr(stamp, "sec", 0)),
                stamp_nanosec=int(getattr(stamp, "nanosec", 0)),
                backend=self.detector.backend,
                inference_ms=stats.inference_ms,
            )
            msg = String()
            msg.data = payload
            self.json_pub.publish(msg)

            if self.visualize:
                ann = self.detector.annotate(bgr, dets)
                self.annotated_pub.publish(
                    bgr8_to_imgmsg(ann, frame_id=self.frame_id, stamp=stamp)
                )

            if self._frame_i % max(1, self.log_every_n) == 0:
                axes = ",".join(f"{k}:{v}" for k, v in stats.axis_methods.items()) or "-"
                self.get_logger().info(
                    f"n={len(dets)} ms={stats.inference_ms:.1f} fps~{fps:.1f} "
                    f"caps={stats.n_caps} linked={stats.n_linked} axis[{axes}] "
                    f"wall={(time.perf_counter()-t0)*1000:.1f}"
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
