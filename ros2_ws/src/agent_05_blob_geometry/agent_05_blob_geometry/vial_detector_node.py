"""ROS2 vial detector node — offline dataset OR live Image subscription.

Publishes:
  - vision_msgs/Detection2DArray on detections_topic (if vision_msgs installed)
  - std_msgs/String JSON on detections_json_topic (always)
  - sensor_msgs/Image annotated frame on annotated_topic (when visualize=true)

Logs rolling FPS and per-frame detection count.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import cv2


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

    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    from agent_05_blob_geometry.blob_detector import BlobParams, VialBlobDetector, list_images
    from agent_05_blob_geometry.image_codec import bgr8_to_imgmsg, imgmsg_to_numpy
    from agent_05_blob_geometry.msgs_compat import (
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

            params = BlobParams(
                confidence_threshold=g("confidence_threshold").get_parameter_value().double_value,
                max_detections=int(g("max_detections").get_parameter_value().integer_value),
                clahe_clip=g("clahe_clip").get_parameter_value().double_value,
                gaussian_ksize=int(g("gaussian_ksize").get_parameter_value().integer_value),
                white_s_max=int(g("white_s_max").get_parameter_value().integer_value),
                white_v_min=int(g("white_v_min").get_parameter_value().integer_value),
                blob_min_area=g("blob_min_area").get_parameter_value().double_value,
                blob_max_area=g("blob_max_area").get_parameter_value().double_value,
                blob_min_circularity=g("blob_min_circularity").get_parameter_value().double_value,
                blob_min_inertia=g("blob_min_inertia").get_parameter_value().double_value,
                blob_min_convexity=g("blob_min_convexity").get_parameter_value().double_value,
                blob_min_dist=g("blob_min_dist").get_parameter_value().double_value,
                blob_min_size=g("blob_min_size").get_parameter_value().double_value,
                blob_max_size=g("blob_max_size").get_parameter_value().double_value,
                mser_delta=int(g("mser_delta").get_parameter_value().integer_value),
                mser_min_area=int(g("mser_min_area").get_parameter_value().integer_value),
                mser_max_area=int(g("mser_max_area").get_parameter_value().integer_value),
                geom_min_circularity=g("geom_min_circularity").get_parameter_value().double_value,
                geom_min_convexity=g("geom_min_convexity").get_parameter_value().double_value,
                geom_min_inertia=g("geom_min_inertia").get_parameter_value().double_value,
                geom_min_aspect=g("geom_min_aspect").get_parameter_value().double_value,
                geom_max_aspect=g("geom_max_aspect").get_parameter_value().double_value,
                expand_width_scale=g("expand_width_scale").get_parameter_value().double_value,
                expand_height_scale=g("expand_height_scale").get_parameter_value().double_value,
                nms_iou=g("nms_iou").get_parameter_value().double_value,
            )
            self.detector = VialBlobDetector(params)
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
            if not self.using_vision:
                self.get_logger().warn(
                    "vision_msgs not installed — publishing detections only as "
                    f"std_msgs/String JSON on {self.detections_json_topic}"
                )

            if self.run_offline:
                self._start_offline()
            else:
                self.create_subscription(Image, self.image_topic, self._on_image, 10)
                self.get_logger().info(f"Subscribing to {self.image_topic}")

        def _declare(self) -> None:
            defaults = {
                "image_topic": "/camera/color/image_raw",
                "dataset_path": "",
                "run_offline": False,
                "confidence_threshold": 0.70,
                "max_detections": 2,
                "visualize": True,
                "annotated_topic": "/vial/annotated",
                "detections_topic": "/vial/detections",
                "detections_json_topic": "/vial/detections_json",
                "frame_id": "camera_color_optical_frame",
                "offline_rate_hz": 5.0,
                "offline_loop": False,
                "log_every_n": 10,
                "clahe_clip": 2.0,
                "gaussian_ksize": 5,
                "white_s_max": 70,
                "white_v_min": 170,
                "blob_min_area": 120.0,
                "blob_max_area": 25000.0,
                "blob_min_circularity": 0.55,
                "blob_min_inertia": 0.35,
                "blob_min_convexity": 0.75,
                "blob_min_dist": 20.0,
                "blob_min_size": 14.0,
                "blob_max_size": 220.0,
                "mser_delta": 4,
                "mser_min_area": 150,
                "mser_max_area": 20000,
                "geom_min_circularity": 0.50,
                "geom_min_convexity": 0.80,
                "geom_min_inertia": 0.30,
                "geom_min_aspect": 0.50,
                "geom_max_aspect": 2.00,
                "expand_width_scale": 2.4,
                "expand_height_scale": 3.6,
                "nms_iou": 0.35,
            }
            for name, value in defaults.items():
                self.declare_parameter(name, value)

        def _start_offline(self) -> None:
            folder = Path(self.dataset_path) if self.dataset_path else Path()
            self._offline_images = list_images(folder)
            if not self._offline_images:
                self.get_logger().error(f"No images in dataset_path={folder}")
                return
            self.get_logger().info(
                f"Offline mode: {len(self._offline_images)} images from {folder}"
            )
            period = 1.0 / float(self.offline_rate_hz) if self.offline_rate_hz > 0 else 0.001
            self._offline_timer = self.create_timer(period, self._offline_tick)

        def _offline_tick(self) -> None:
            if not self._offline_images:
                return
            if self._offline_idx >= len(self._offline_images):
                if self.offline_loop:
                    self._offline_idx = 0
                else:
                    self.get_logger().info(
                        "Offline dataset complete; cancelling timer "
                        "(node stays up — Ctrl+C to exit)."
                    )
                    if self._offline_timer is not None:
                        self._offline_timer.cancel()
                    return
            path = self._offline_images[self._offline_idx]
            self._offline_idx += 1
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                self.get_logger().warn(f"Failed to read {path}")
                return
            stamp = self.get_clock().now().to_msg()
            self._process(bgr, stamp, source_name=path.name)

        def _on_image(self, msg: Image) -> None:
            try:
                bgr = imgmsg_to_numpy(msg)
            except ValueError as exc:
                self.get_logger().warn(f"Bad image: {exc}")
                return
            self._process(bgr, msg.header.stamp, source_name=msg.header.frame_id)

        def _process(self, bgr, stamp, *, source_name: str = "") -> None:
            dets, stats = self.detector.predict(bgr)
            now = time.perf_counter()
            self._fps_times.append(now)
            fps = 0.0
            if len(self._fps_times) >= 2:
                dt = self._fps_times[-1] - self._fps_times[0]
                if dt > 0:
                    fps = (len(self._fps_times) - 1) / dt

            json_str = detections_to_json(
                dets,
                frame_id=self.frame_id,
                stamp_sec=int(getattr(stamp, "sec", 0)),
                stamp_nanosec=int(getattr(stamp, "nanosec", 0)),
                backend=stats.backend,
                inference_ms=stats.inference_ms,
            )
            jmsg = String()
            jmsg.data = json_str
            self.json_pub.publish(jmsg)

            if self.vision_pub is not None:
                arr = build_detection2d_array(dets, frame_id=self.frame_id, stamp=stamp)
                self.vision_pub.publish(arr)

            if self.visualize:
                annotated = self.detector.annotate(bgr, dets)
                cv2.putText(
                    annotated,
                    f"FPS {fps:.1f}  n={len(dets)}  blob/mser/cc="
                    f"{stats.n_blob}/{stats.n_mser}/{stats.n_cc}",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                self.annotated_pub.publish(
                    bgr8_to_imgmsg(annotated, frame_id=self.frame_id, stamp=stamp)
                )

            self._frame_i += 1
            if self.log_every_n <= 1 or (self._frame_i % self.log_every_n) == 0:
                self.get_logger().info(
                    f"frame={self._frame_i} src={source_name!r} dets={len(dets)} "
                    f"infer_ms={stats.inference_ms:.1f} fps={fps:.1f} backend={stats.backend}"
                )

    rclpy.init(args=args)
    node = VialDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
