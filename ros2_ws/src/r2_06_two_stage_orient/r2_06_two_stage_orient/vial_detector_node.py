"""ROS2 two-stage vial detector node (offline dataset OR live Image).

Publishes:
  - vision_msgs/Detection2DArray on detections_topic (if available)
  - std_msgs/String JSON on detections_json_topic (always; includes orientation)
  - sensor_msgs/Image annotated frame on annotated_topic
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import cv2
import numpy as np


def main(args: Optional[List[str]] = None) -> None:
    try:
        import rclpy
        from rclpy.node import Node
    except ImportError as exc:
        print(
            "rclpy is not available in this Python. Activate the RoboStack/ROS env first. "
            f"Import error: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)

    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    from r2_06_two_stage_orient.image_codec import (
        bgr8_to_imgmsg,
        depth_imgmsg_to_mm,
        imgmsg_to_numpy,
    )
    from r2_06_two_stage_orient.msgs_compat import (
        build_detection2d_array,
        create_detection_publishers,
        detections_to_json,
    )
    from r2_06_two_stage_orient.pipeline import TwoStageOrientDetector

    class VialDetectorNode(Node):
        def __init__(self) -> None:
            super().__init__("vial_detector_node")
            self._declare()
            g = self.get_parameter
            self.image_topic = g("image_topic").get_parameter_value().string_value
            self.depth_topic = g("depth_topic").get_parameter_value().string_value
            self.dataset_path = g("dataset_path").get_parameter_value().string_value
            self.run_offline = g("run_offline").get_parameter_value().bool_value
            self.model_path = g("model_path").get_parameter_value().string_value
            self.camera_root = g("camera_root").get_parameter_value().string_value
            self.conf = g("confidence_threshold").get_parameter_value().double_value
            self.iou = g("iou_threshold").get_parameter_value().double_value
            self.imgsz = int(g("imgsz").get_parameter_value().integer_value)
            self.use_yolo = g("use_yolo").get_parameter_value().bool_value
            self.use_rotation_tta = g("use_rotation_tta").get_parameter_value().bool_value
            self.use_cap_proposals = g("use_cap_proposals").get_parameter_value().bool_value
            self.use_sideways_method = g("use_sideways_method").get_parameter_value().bool_value
            self.min_conf = g("min_det_conf").get_parameter_value().double_value
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

            self.detector = TwoStageOrientDetector(
                camera_root=self.camera_root or None,
                model_path=self.model_path or None,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                use_yolo=self.use_yolo,
                use_rotation_tta=self.use_rotation_tta,
                use_cap_proposals=self.use_cap_proposals,
                use_sideways_method=self.use_sideways_method,
                min_conf=self.min_conf,
            )
            self.detector.load()
            self.vision_pub, self.json_pub, self.using_vision = create_detection_publishers(
                self, self.detections_topic, self.detections_json_topic
            )
            self.annotated_pub = self.create_publisher(Image, self.annotated_topic, 10)

            self._fps_times: Deque[float] = deque(maxlen=60)
            self._frame_i = 0
            self._offline_idx = 0
            self._offline_images: List[Path] = []
            self._latest_depth: Optional[np.ndarray] = None

            self.get_logger().info(
                f"backend={self.detector.backend} weights={self.detector.model_path!r} "
                f"vision_msgs={self.using_vision} offline={self.run_offline}"
            )

            if self.depth_topic:
                self.create_subscription(Image, self.depth_topic, self._on_depth, 10)

            if self.run_offline:
                root = Path(self.dataset_path)
                exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                self._offline_images = sorted(
                    p for p in root.rglob("*") if p.suffix.lower() in exts
                )
                if not self._offline_images:
                    self.get_logger().error(f"No images under {root}")
                period = 1.0 / max(0.1, self.offline_rate_hz)
                self.create_timer(period, self._offline_tick)
            else:
                self.create_subscription(Image, self.image_topic, self._on_image, 10)

        def _declare(self) -> None:
            defaults = {
                "image_topic": "/orbbec/cam0/color/image_raw",
                "depth_topic": "/orbbec/cam0/depth/image_raw",
                "dataset_path": "E:/Desktop/Jessica/robotics/vial_dataset/raw",
                "run_offline": True,
                "model_path": "",
                "camera_root": "E:/Desktop/Jessica/robotics",
                "confidence_threshold": 0.12,
                "iou_threshold": 0.45,
                "imgsz": 640,
                "use_yolo": True,
                "use_rotation_tta": True,
                "use_cap_proposals": True,
                "use_sideways_method": True,
                "min_det_conf": 0.18,
                "visualize": True,
                "annotated_topic": "/vial/annotated",
                "detections_topic": "/vial/detections",
                "detections_json_topic": "/vial/detections_json",
                "frame_id": "camera_color_optical_frame",
                "offline_rate_hz": 5.0,
                "offline_loop": False,
                "log_every_n": 10,
            }
            for k, v in defaults.items():
                self.declare_parameter(k, v)

        def _on_depth(self, msg: Image) -> None:
            try:
                self._latest_depth = depth_imgmsg_to_mm(msg)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"depth decode failed: {exc}")

        def _on_image(self, msg: Image) -> None:
            try:
                bgr = imgmsg_to_numpy(msg)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"image decode failed: {exc}")
                return
            self._process(bgr, msg.header.stamp)

        def _offline_tick(self) -> None:
            if not self._offline_images:
                return
            if self._offline_idx >= len(self._offline_images):
                if self.offline_loop:
                    self._offline_idx = 0
                else:
                    self.get_logger().info("Offline dataset complete.")
                    return
            path = self._offline_images[self._offline_idx]
            self._offline_idx += 1
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                return
            stamp = self.get_clock().now().to_msg()
            # Pair synth depth when filename matches *_color.png
            depth = None
            if path.name.endswith("_color.png"):
                dpath = path.with_name(path.name.replace("_color.png", "_depth.npy"))
                if dpath.is_file():
                    depth = np.load(dpath).astype(np.float32)
            self._latest_depth = depth
            self._process(bgr, stamp)

        def _process(self, bgr: np.ndarray, stamp) -> None:
            t0 = time.perf_counter()
            dets, stats = self.detector.detect(bgr, depth_mm=self._latest_depth)
            self._fps_times.append(time.perf_counter() - t0)
            self._frame_i += 1

            sec = int(getattr(stamp, "sec", 0))
            nsec = int(getattr(stamp, "nanosec", 0))
            payload = detections_to_json(
                dets,
                frame_id=self.frame_id,
                stamp_sec=sec,
                stamp_nanosec=nsec,
                backend=stats.backend,
                inference_ms=stats.inference_ms,
            )
            msg_s = String()
            msg_s.data = payload
            self.json_pub.publish(msg_s)

            if self.vision_pub is not None:
                try:
                    arr = build_detection2d_array(
                        dets, frame_id=self.frame_id, stamp=stamp
                    )
                    self.vision_pub.publish(arr)
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(f"vision_msgs publish failed: {exc}")

            if self.visualize:
                ann = self.detector.annotate(bgr, dets)
                self.annotated_pub.publish(
                    bgr8_to_imgmsg(ann, frame_id=self.frame_id, stamp=stamp)
                )

            if self.log_every_n > 0 and self._frame_i % self.log_every_n == 0:
                fps = (
                    len(self._fps_times) / max(1e-6, sum(self._fps_times))
                    if self._fps_times
                    else 0.0
                )
                orients = {d.orientation: 0 for d in dets}
                for d in dets:
                    orients[d.orientation] = orients.get(d.orientation, 0) + 1
                self.get_logger().info(
                    f"frame={self._frame_i} dets={len(dets)} props={stats.n_proposals} "
                    f"orients={orients} {stats.inference_ms:.1f}ms ~{fps:.1f}FPS"
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
