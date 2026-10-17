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

    from agent_04_hsv_morph.detector import (
        HsvMorphParams,
        VialHsvMorphDetector,
        list_images,
    )
    from agent_04_hsv_morph.image_codec import bgr8_to_imgmsg, imgmsg_to_numpy
    from agent_04_hsv_morph.msgs_compat import (
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

            params = HsvMorphParams(
                hsv_h_min=int(g("hsv_h_min").value),
                hsv_h_max=int(g("hsv_h_max").value),
                hsv_s_min=int(g("hsv_s_min").value),
                hsv_s_max=int(g("hsv_s_max").value),
                hsv_v_min=int(g("hsv_v_min").value),
                hsv_v_max=int(g("hsv_v_max").value),
                lab_l_min=int(g("lab_l_min").value),
                lab_l_max=int(g("lab_l_max").value),
                lab_a_center=int(g("lab_a_center").value),
                lab_a_tol=int(g("lab_a_tol").value),
                lab_b_center=int(g("lab_b_center").value),
                lab_b_tol=int(g("lab_b_tol").value),
                morph_open_ksize=int(g("morph_open_ksize").value),
                morph_close_ksize=int(g("morph_close_ksize").value),
                use_edge_assist=bool(g("use_edge_assist").value),
                canny_low=int(g("canny_low").value),
                canny_high=int(g("canny_high").value),
                body_expand_cap_h_mult=float(g("body_expand_cap_h_mult").value),
                body_expand_cap_w_mult=float(g("body_expand_cap_w_mult").value),
                min_area_frac=float(g("min_area_frac").value),
                max_area_frac=float(g("max_area_frac").value),
                min_box_aspect=float(g("min_box_aspect").value),
                max_box_aspect=float(g("max_box_aspect").value),
                max_box_area_frac=float(g("max_box_area_frac").value),
                min_cap_fill=float(g("min_cap_fill").value),
                min_score=float(g("min_score").value),
                max_dets=int(g("max_dets").value),
                nms_iou=float(g("nms_iou").value),
            )
            self.detector = VialHsvMorphDetector(params)

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
                f"backend=hsv_morph S<={params.hsv_s_max} V>={params.hsv_v_min} "
                f"edge_assist={params.use_edge_assist} vision_msgs={self.using_vision} "
                f"(available={vision_msgs_available()})"
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
            self.declare_parameter("image_topic", "/camera/color/image_raw")
            self.declare_parameter("dataset_path", "")
            self.declare_parameter("run_offline", False)
            self.declare_parameter("hsv_h_min", 0)
            self.declare_parameter("hsv_h_max", 179)
            self.declare_parameter("hsv_s_min", 0)
            self.declare_parameter("hsv_s_max", 40)
            self.declare_parameter("hsv_v_min", 195)
            self.declare_parameter("hsv_v_max", 255)
            self.declare_parameter("lab_l_min", 190)
            self.declare_parameter("lab_l_max", 255)
            self.declare_parameter("lab_a_center", 128)
            self.declare_parameter("lab_a_tol", 12)
            self.declare_parameter("lab_b_center", 128)
            self.declare_parameter("lab_b_tol", 12)
            self.declare_parameter("morph_open_ksize", 5)
            self.declare_parameter("morph_close_ksize", 5)
            self.declare_parameter("use_edge_assist", True)
            self.declare_parameter("canny_low", 50)
            self.declare_parameter("canny_high", 120)
            self.declare_parameter("body_expand_cap_h_mult", 2.6)
            self.declare_parameter("body_expand_cap_w_mult", 1.35)
            self.declare_parameter("min_area_frac", 0.0006)
            self.declare_parameter("max_area_frac", 0.08)
            self.declare_parameter("min_box_aspect", 0.75)
            self.declare_parameter("max_box_aspect", 2.6)
            self.declare_parameter("max_box_area_frac", 0.28)
            self.declare_parameter("min_cap_fill", 0.25)
            self.declare_parameter("min_score", 0.42)
            self.declare_parameter("max_dets", 3)
            self.declare_parameter("nms_iou", 0.35)
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
                    f"FPS {fps:.1f}  n={len(dets)}  hsv_morph",
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
                    f"infer_ms={stats.inference_ms:.1f} fps={fps:.1f} "
                    f"mask_px={stats.mask_pixels}"
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
