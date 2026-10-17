"""ROS2 vial detector node — offline dataset OR live Image subscription.

Publishes oriented detections (theta = vial axis) via vision_msgs when available,
plus JSON with explicit cap/base keypoints.
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

    from r2_09_keypoint_axis.detector import AxisParams, KeypointAxisDetector
    from r2_09_keypoint_axis.image_codec import bgr8_to_imgmsg, imgmsg_to_numpy
    from r2_09_keypoint_axis.msgs_compat import (
        build_detection2d_array,
        create_detection_publishers,
        detections_to_json,
        vision_msgs_available,
    )
    from r2_09_keypoint_axis.types import list_images

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

            params = AxisParams(
                confidence_threshold=g("confidence_threshold").get_parameter_value().double_value,
                max_detections=int(g("max_detections").get_parameter_value().integer_value),
                white_s_max=int(g("white_s_max").get_parameter_value().integer_value),
                white_v_min=int(g("white_v_min").get_parameter_value().integer_value),
                min_cap_area=int(g("min_cap_area").get_parameter_value().integer_value),
                nms_iou=g("nms_iou").get_parameter_value().double_value,
                expected_aspect=g("expected_aspect").get_parameter_value().double_value,
                use_yolo_seed=g("use_yolo_seed").get_parameter_value().bool_value,
                yolo_weights=g("yolo_weights").get_parameter_value().string_value,
                yolo_conf=g("yolo_conf").get_parameter_value().double_value,
                min_base_score=g("min_base_score").get_parameter_value().double_value,
            )
            self.detector = KeypointAxisDetector(params)
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
                "dataset_path": "E:/Desktop/Jessica/robotics/sideways_race/synth",
                "run_offline": True,
                "confidence_threshold": 0.35,
                "max_detections": 3,
                "visualize": True,
                "annotated_topic": "/vial/annotated",
                "detections_topic": "/vial/detections",
                "detections_json_topic": "/vial/detections_json",
                "frame_id": "camera_color_optical_frame",
                "offline_rate_hz": 5.0,
                "offline_loop": False,
                "log_every_n": 10,
                "white_s_max": 55,
                "white_v_min": 200,
                "min_cap_area": 35,
                "nms_iou": 0.40,
                "expected_aspect": 1.34,
                "use_yolo_seed": True,
                "yolo_weights": "",
                "yolo_conf": 0.15,
                "min_base_score": 0.85,
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
            depth = None
            depth_path = path.with_name(path.name.replace("_color.png", "_depth.npy"))
            if depth_path.is_file():
                try:
                    import numpy as np

                    depth = np.load(str(depth_path))
                except Exception:  # noqa: BLE001
                    depth = None
            stamp = self.get_clock().now().to_msg()
            self._process(bgr, stamp, source_name=path.name, depth_mm=depth)

        def _on_image(self, msg: Image) -> None:
            try:
                bgr = imgmsg_to_numpy(msg)
            except ValueError as exc:
                self.get_logger().warn(f"Bad image: {exc}")
                return
            self._process(bgr, msg.header.stamp, source_name=msg.header.frame_id)

        def _process(self, bgr, stamp, *, source_name: str = "", depth_mm=None) -> None:
            dets, stats = self.detector.predict(bgr, depth_mm=depth_mm)
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
                    f"FPS {fps:.1f}  n={len(dets)}  caps={stats.n_caps}",
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
                angles = ",".join(f"{d.angle_deg:.0f}" for d in dets) or "-"
                self.get_logger().info(
                    f"frame={self._frame_i} src={source_name!r} dets={len(dets)} "
                    f"angles=[{angles}] infer_ms={stats.inference_ms:.1f} fps={fps:.1f} "
                    f"backend={stats.backend}"
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
