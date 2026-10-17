"""ROS2 vial detector node — SAHI / multi-scale YOLO.

Offline dataset iteration or live Image subscription. Publishes
vision_msgs/Detection2DArray (if available) + JSON String + annotated Image.
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
            "rclpy is not available in this Python. Source ROS 2 / RoboStack first. "
            f"Import error: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)

    import cv2
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    from r2_08_sahi_multiscale.detector import SahiMultiscaleDetector, list_images
    from r2_08_sahi_multiscale.image_codec import bgr8_to_imgmsg, imgmsg_to_numpy
    from r2_08_sahi_multiscale.msgs_compat import (
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
            self.model_path = g("model_path").get_parameter_value().string_value
            self.camera_root = g("camera_root").get_parameter_value().string_value
            self.conf = g("confidence_threshold").get_parameter_value().double_value
            self.iou = g("iou_threshold").get_parameter_value().double_value
            self.imgsz = int(g("imgsz").get_parameter_value().integer_value)
            self.mode = g("mode").get_parameter_value().string_value
            self.slice_w = int(g("slice_width").get_parameter_value().integer_value)
            self.slice_h = int(g("slice_height").get_parameter_value().integer_value)
            self.overlap = g("overlap_ratio").get_parameter_value().double_value
            scales_str = g("image_scales").get_parameter_value().string_value
            self.image_scales = [float(x) for x in scales_str.split(",") if x.strip()]
            self.rotation_tta = g("rotation_tta").get_parameter_value().bool_value
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

            self.detector = SahiMultiscaleDetector(
                self.model_path or None,
                camera_root=self.camera_root or None,
                confidence_threshold=self.conf,
                iou_threshold=self.iou,
                imgsz=self.imgsz,
                mode=self.mode,
                slice_width=self.slice_w,
                slice_height=self.slice_h,
                overlap_ratio=self.overlap,
                image_scales=self.image_scales or [1.0, 1.25],
                rotation_tta=self.rotation_tta,
            )
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
                f"r2_08 SAHI node mode={self.detector.mode} backend={self.detector.backend} "
                f"weights={self.detector.model_path} conf={self.conf} "
                f"slice={self.slice_w}x{self.slice_h} overlap={self.overlap} "
                f"scales={self.image_scales} rot_tta={self.rotation_tta}"
            )
            if not self.using_vision:
                self.get_logger().warn(
                    "vision_msgs not installed — detections as JSON only on "
                    f"{self.detections_json_topic}"
                )

            if self.run_offline:
                self._offline_images = list_images(self.dataset_path)
                if not self._offline_images:
                    self.get_logger().error(f"No images in {self.dataset_path}")
                period = 1.0 / self.offline_rate_hz if self.offline_rate_hz > 0 else 0.05
                self._offline_timer = self.create_timer(period, self._offline_tick)
            else:
                self.create_subscription(Image, self.image_topic, self._on_image, 10)

        def _declare(self) -> None:
            self.declare_parameter("image_topic", "/camera/color/image_raw")
            self.declare_parameter(
                "dataset_path", "E:/Desktop/Jessica/robotics/sideways_race/synth"
            )
            self.declare_parameter("run_offline", True)
            self.declare_parameter(
                "model_path",
                "E:/Desktop/Jessica/robotics/runs/bon-winner/weights/best.pt",
            )
            self.declare_parameter("camera_root", "E:/Desktop/Jessica/robotics")
            self.declare_parameter("confidence_threshold", 0.25)
            self.declare_parameter("iou_threshold", 0.45)
            self.declare_parameter("imgsz", 640)
            self.declare_parameter("mode", "sahi")
            self.declare_parameter("slice_width", 320)
            self.declare_parameter("slice_height", 320)
            self.declare_parameter("overlap_ratio", 0.25)
            self.declare_parameter("image_scales", "1.0,1.5")
            self.declare_parameter("rotation_tta", True)
            self.declare_parameter("visualize", True)
            self.declare_parameter("annotated_topic", "/vial/annotated")
            self.declare_parameter("detections_topic", "/vial/detections")
            self.declare_parameter("detections_json_topic", "/vial/detections_json")
            self.declare_parameter("frame_id", "camera_color_optical_frame")
            self.declare_parameter("offline_rate_hz", 2.0)
            self.declare_parameter("offline_loop", False)
            self.declare_parameter("log_every_n", 5)

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
            # Prefer *_color.png in synth folders
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                return
            self._process(bgr)

        def _on_image(self, msg: Image) -> None:
            try:
                bgr = imgmsg_to_numpy(msg)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"Image decode failed: {exc}")
                return
            self._process(bgr, stamp=msg.header.stamp)

        def _process(self, bgr, stamp=None) -> None:
            t0 = time.perf_counter()
            dets, stats = self.detector.predict(bgr)
            now = time.perf_counter()
            self._fps_times.append(now)
            self._frame_i += 1

            if stamp is None:
                stamp = self.get_clock().now().to_msg()

            json_str = detections_to_json(
                dets,
                frame_id=self.frame_id,
                stamp_sec=int(stamp.sec),
                stamp_nanosec=int(stamp.nanosec),
                backend=stats.backend,
                inference_ms=stats.inference_ms,
                mode=stats.mode,
            )
            self.json_pub.publish(String(data=json_str))
            if self.vision_pub is not None:
                arr = build_detection2d_array(dets, frame_id=self.frame_id, stamp=stamp)
                self.vision_pub.publish(arr)
            if self.visualize:
                annotated = self.detector.annotate(bgr, dets)
                self.annotated_pub.publish(
                    bgr8_to_imgmsg(annotated, frame_id=self.frame_id, stamp=stamp)
                )

            if self._frame_i % max(1, self.log_every_n) == 0:
                fps = 0.0
                if len(self._fps_times) >= 2:
                    dt = self._fps_times[-1] - self._fps_times[0]
                    if dt > 0:
                        fps = (len(self._fps_times) - 1) / dt
                self.get_logger().info(
                    f"frame={self._frame_i} dets={len(dets)} "
                    f"infer_ms={stats.inference_ms:.1f} passes={stats.num_passes} "
                    f"slices={stats.num_slices} fps~{fps:.2f} wall_ms={(now-t0)*1000:.1f}"
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
