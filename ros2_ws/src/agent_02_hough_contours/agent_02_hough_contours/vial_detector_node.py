"""ROS2 vial detector node — offline dataset OR live Image subscription.

Classical HoughCircles + contour/ellipse geometry (Agent #02).

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


def main(args: Optional[List[str]] = None) -> None:
    try:
        import rclpy
        from rclpy.node import Node
    except ImportError as exc:
        print(
            "rclpy is not available in this Python. On Linux: source /opt/ros/$ROS_DISTRO/setup.bash. "
            "On Windows RoboStack: activate the ros env, then source local_setup. "
            f"Import error: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)

    import cv2
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    from agent_02_hough_contours.detector import (
        HoughContourParams,
        VialHoughContourDetector,
        list_images,
        params_from_ros,
    )
    from agent_02_hough_contours.image_codec import bgr8_to_imgmsg, imgmsg_to_numpy
    from agent_02_hough_contours.msgs_compat import (
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

            algo = params_from_ros(self.get_parameter)
            self.detector = VialHoughContourDetector(algo)
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
                f"backend={self.detector.backend} "
                f"dp={algo.dp} min_circ={algo.min_circularity} "
                f"canny={algo.canny} use_hough={algo.use_hough} "
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
            # Shared contract
            self.declare_parameter("image_topic", "/camera/color/image_raw")
            self.declare_parameter("dataset_path", "")
            self.declare_parameter("run_offline", False)
            self.declare_parameter("visualize", True)
            self.declare_parameter("annotated_topic", "/vial/annotated")
            self.declare_parameter("detections_topic", "/vial/detections")
            self.declare_parameter("detections_json_topic", "/vial/detections_json")
            self.declare_parameter("frame_id", "camera_color_optical_frame")
            self.declare_parameter("offline_rate_hz", 5.0)
            self.declare_parameter("offline_loop", False)
            self.declare_parameter("log_every_n", 10)
            # Algorithm-specific
            defaults = HoughContourParams()
            self.declare_parameter("v_min", defaults.v_min)
            self.declare_parameter("s_max", defaults.s_max)
            self.declare_parameter("min_circularity", defaults.min_circularity)
            self.declare_parameter("max_circularity", defaults.max_circularity)
            self.declare_parameter("min_ellipse_ar", defaults.min_ellipse_ar)
            self.declare_parameter("min_area_frac", defaults.min_area_frac)
            self.declare_parameter("max_area_frac", defaults.max_area_frac)
            self.declare_parameter("cap_expand_down", defaults.cap_expand_down)
            self.declare_parameter("cap_expand_side", defaults.cap_expand_side)
            self.declare_parameter("min_box_aspect", defaults.min_box_aspect)
            self.declare_parameter("max_box_aspect", defaults.max_box_aspect)
            self.declare_parameter("use_hough", defaults.use_hough)
            self.declare_parameter("dp", defaults.dp)
            self.declare_parameter("min_dist_frac", defaults.min_dist_frac)
            self.declare_parameter("canny", defaults.canny)
            self.declare_parameter("accum", defaults.accum)
            self.declare_parameter("min_radius_frac", defaults.min_radius_frac)
            self.declare_parameter("max_radius_frac", defaults.max_radius_frac)
            self.declare_parameter("hough_white_support", defaults.hough_white_support)
            self.declare_parameter("score_thresh", defaults.score_thresh)
            self.declare_parameter("nms_iou", defaults.nms_iou)
            self.declare_parameter("max_dets", defaults.max_dets)

        def _start_offline(self) -> None:
            folder = Path(self.dataset_path) if self.dataset_path else Path()
            self._offline_images = list_images(folder)
            if not self._offline_images:
                self.get_logger().error(f"No images in dataset_path={folder}")
                return
            self.get_logger().info(
                f"Offline mode: {len(self._offline_images)} images from {folder}"
            )
            if self.offline_rate_hz and self.offline_rate_hz > 0:
                period = 1.0 / float(self.offline_rate_hz)
            else:
                period = 0.001
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
                    f"FPS {fps:.1f}  n={len(dets)}  {stats.backend}",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
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
