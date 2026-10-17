"""ROS2 multi-orientation vial fusion node."""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


def main(args: Optional[List[str]] = None) -> None:
    try:
        import rclpy
        from rclpy.node import Node
    except ImportError as exc:
        print(
            "rclpy is not available in this Python. "
            f"Import error: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)

    from sensor_msgs.msg import Image
    from rclpy.qos import qos_profile_sensor_data

    from r2_10_fusion_multiorient.fusion import FusionConfig, FusionVialDetector
    from r2_10_fusion_multiorient.image_codec import bgr8_to_imgmsg, imgmsg_to_numpy
    from r2_10_fusion_multiorient.msgs_compat import (
        build_detection2d_array,
        create_detection_publishers,
        detections_to_json,
        vision_msgs_available,
    )
    from r2_10_fusion_multiorient.preview_gui import PreviewWindow

    class VialDetectorNode(Node):
        def __init__(self) -> None:
            super().__init__("vial_detector_node")
            self._declare()
            g = self.get_parameter
            self.image_topic = g("image_topic").get_parameter_value().string_value
            self.depth_topic = g("depth_topic").get_parameter_value().string_value
            self.image_topic_2 = g("image_topic_2").get_parameter_value().string_value
            self.depth_topic_2 = g("depth_topic_2").get_parameter_value().string_value
            self.dataset_path = g("dataset_path").get_parameter_value().string_value
            self.run_offline = g("run_offline").get_parameter_value().bool_value
            self.model_path = g("model_path").get_parameter_value().string_value
            offline_use_yolo = g("use_yolo").get_parameter_value().bool_value
            live_use_yolo = g("use_yolo_live").get_parameter_value().bool_value
            # Keep audited offline evaluation unchanged.  Live Windows ROS
            # defaults classical-only because loading Torch/Ultralytics inside
            # a launch child can natively abort with exit code 3.
            self.use_yolo = offline_use_yolo if self.run_offline else live_use_yolo
            self.camera_root = g("camera_root").get_parameter_value().string_value
            self.conf = g("confidence_threshold").get_parameter_value().double_value
            self.iou = g("iou_threshold").get_parameter_value().double_value
            self.imgsz = int(g("imgsz").get_parameter_value().integer_value)
            self.use_tta = g("use_yolo_tta").get_parameter_value().bool_value
            self.use_classical = g("use_classical").get_parameter_value().bool_value
            self.use_classical_sideways = (
                g("use_classical_sideways").get_parameter_value().bool_value
            )
            self.enable_rgb_classical_sideways = (
                g("enable_rgb_classical_sideways").get_parameter_value().bool_value
            )
            self.use_refine = g("use_cap_refine").get_parameter_value().bool_value
            self.extra_yolo_conf_after_method_i = (
                g("extra_yolo_conf_after_method_i").get_parameter_value().double_value
            )
            self.empty_scene_yolo_floor = (
                g("empty_scene_yolo_floor").get_parameter_value().double_value
            )
            self.empty_method_i_keep_conf = (
                g("empty_method_i_keep_conf").get_parameter_value().double_value
            )
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

            cfg = FusionConfig(
                camera_root=self.camera_root or None,
                model_path=self.model_path or None,
                yolo_conf=self.conf,
                yolo_iou=self.iou,
                imgsz=self.imgsz,
                use_yolo=self.use_yolo,
                use_yolo_tta=self.use_tta,
                use_classical_upright=self.use_classical,
                use_classical_sideways=self.use_classical_sideways
                if self.use_classical
                else False,
                enable_rgb_classical_sideways=self.enable_rgb_classical_sideways,
                use_cap_refine=self.use_refine,
                extra_yolo_conf_after_method_i=self.extra_yolo_conf_after_method_i,
                empty_scene_yolo_floor=self.empty_scene_yolo_floor,
                empty_method_i_keep_conf=self.empty_method_i_keep_conf,
            )
            if self.use_yolo:
                self.get_logger().warn(
                    f"YOLO opt-in: loading model={self.model_path or 'auto'} "
                    "in-process. On Windows RoboStack this may natively abort "
                    "with exit code 3; leave use_yolo_live:=false there."
                )
            else:
                self.get_logger().info(
                    "Live YOLO disabled (Windows ROS crash hardening); "
                    "starting classical/Method-I detector without Torch."
                )
            self.detector = FusionVialDetector(cfg)
            self._depths: dict[str, np.ndarray] = {}
            self._preview: Optional[PreviewWindow] = None
            self._preview_failed = False
            self._annotated_frames: dict[str, np.ndarray] = {}
            self._live_frames: dict[str, int] = {}
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
                f"backend={self.detector.backend} model={self.detector.model_path} "
                f"yolo={self.use_yolo} tta={self.use_tta} "
                f"classical={self.use_classical} "
                f"vision_msgs={self.using_vision}"
            )
            if not self.using_vision:
                self.get_logger().warn(
                    "vision_msgs not installed — JSON only on "
                    f"{self.detections_json_topic}"
                )

            if self.run_offline:
                self._start_offline()
            else:
                self._subscribe_camera("cam0", self.image_topic, self.depth_topic)
                if self.image_topic_2:
                    self._subscribe_camera("cam1", self.image_topic_2, self.depth_topic_2)
                if self.visualize:
                    self.get_logger().info(
                        "Annotated preview uses one side-by-side r2_10 window; "
                        "live YOLO remains controlled separately by use_yolo_live."
                    )
                self.create_timer(15.0, self._live_health_check)

        def _declare(self) -> None:
            self.declare_parameter(
                "image_topic", "/orbbec/cam0/color/image_raw"
            )
            self.declare_parameter(
                "depth_topic", "/orbbec/cam0/depth/image_raw"
            )
            self.declare_parameter(
                "image_topic_2", "/orbbec/cam1/color/image_raw"
            )
            self.declare_parameter(
                "depth_topic_2", "/orbbec/cam1/depth/image_raw"
            )
            self.declare_parameter("dataset_path", "")
            self.declare_parameter("run_offline", False)
            self.declare_parameter("model_path", "")
            self.declare_parameter("use_yolo", True)
            self.declare_parameter("use_yolo_live", False)
            self.declare_parameter("camera_root", "E:/Desktop/Jessica/robotics")
            self.declare_parameter("confidence_threshold", 0.15)
            self.declare_parameter("iou_threshold", 0.45)
            self.declare_parameter("imgsz", 640)
            self.declare_parameter("use_yolo_tta", True)
            self.declare_parameter("use_classical", True)
            self.declare_parameter("use_classical_sideways", True)
            self.declare_parameter("enable_rgb_classical_sideways", False)
            self.declare_parameter("use_cap_refine", True)
            self.declare_parameter("extra_yolo_conf_after_method_i", 0.60)
            self.declare_parameter("empty_scene_yolo_floor", 0.35)
            self.declare_parameter("empty_method_i_keep_conf", 0.50)
            self.declare_parameter("visualize", True)
            self.declare_parameter("annotated_topic", "/vial/annotated")
            self.declare_parameter("detections_topic", "/vial/detections")
            self.declare_parameter("detections_json_topic", "/vial/detections_json")
            self.declare_parameter("frame_id", "camera_color_optical_frame")
            self.declare_parameter("offline_rate_hz", 5.0)
            self.declare_parameter("offline_loop", False)
            self.declare_parameter("log_every_n", 10)

        def _subscribe_camera(
            self, source_id: str, image_topic: str, depth_topic: str
        ) -> None:
            self._live_frames[source_id] = 0
            self.create_subscription(
                Image,
                image_topic,
                lambda msg, source=source_id: self._on_image(msg, source),
                qos_profile_sensor_data,
            )
            if depth_topic:
                self.create_subscription(
                    Image,
                    depth_topic,
                    lambda msg, source=source_id: self._on_depth(msg, source),
                    qos_profile_sensor_data,
                )
            self.get_logger().info(
                f"{source_id}: subscribing color={image_topic} "
                f"depth={depth_topic or 'disabled'} (sensor-data QoS)"
            )

        def _live_health_check(self) -> None:
            missing = [name for name, count in self._live_frames.items() if count == 0]
            if missing:
                self.get_logger().error(
                    "No color frames received from "
                    + ", ".join(missing)
                    + ". Check Orbbec publisher logs, USB3 connections, and topic names."
                )

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
                    self.get_logger().info("Offline dataset complete.")
                    if self._offline_timer is not None:
                        self._offline_timer.cancel()
                    return
            path = self._offline_images[self._offline_idx]
            self._offline_idx += 1
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                return
            # Optional sibling depth .npy
            depth = None
            npy = path.with_name(path.stem.replace("_color", "") + "_depth.npy")
            if not npy.exists():
                npy = path.with_suffix(".npy")
            if npy.exists():
                try:
                    depth = np.load(str(npy)).astype(np.float32)
                except Exception:  # noqa: BLE001
                    depth = None
            stamp = self.get_clock().now().to_msg()
            self._process(
                bgr,
                stamp,
                depth=depth,
                source_name=path.name,
                source_id="offline",
            )

        def _on_depth(self, msg: Image, source_id: str) -> None:
            try:
                enc = (msg.encoding or "").lower()
                h, w = int(msg.height), int(msg.width)
                if enc in ("16uc1", "mono16"):
                    raw = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(h, w)
                    self._depths[source_id] = raw.astype(np.float32)
                elif enc in ("32fc1",):
                    raw = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(h, w)
                    self._depths[source_id] = raw.copy()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"{source_id}: bad depth: {exc}")

        def _on_image(self, msg: Image, source_id: str) -> None:
            try:
                bgr = imgmsg_to_numpy(msg)
            except ValueError as exc:
                self.get_logger().warn(f"{source_id}: bad image: {exc}")
                return
            self._live_frames[source_id] += 1
            frame_id = msg.header.frame_id or self.frame_id
            self._process(
                bgr,
                msg.header.stamp,
                depth=self._depths.get(source_id),
                source_name=source_id,
                source_id=source_id,
                frame_id=frame_id,
            )

        def _process(
            self,
            bgr,
            stamp,
            depth=None,
            source_name: str = "",
            source_id: str = "camera",
            frame_id: str = "",
        ) -> None:
            output_frame_id = frame_id or self.frame_id
            t0 = time.perf_counter()
            dets = self.detector.detect(bgr, depth)
            ms = (time.perf_counter() - t0) * 1000.0
            self._fps_times.append(time.perf_counter())
            self._frame_i += 1

            sec = int(getattr(stamp, "sec", 0))
            nsec = int(getattr(stamp, "nanosec", 0))
            payload = detections_to_json(
                dets,
                frame_id=output_frame_id,
                stamp_sec=sec,
                stamp_nanosec=nsec,
                backend=self.detector.backend,
                inference_ms=ms,
            )
            from std_msgs.msg import String as StrMsg

            jmsg = StrMsg()
            jmsg.data = payload
            self.json_pub.publish(jmsg)
            if self.using_vision and self.vision_pub is not None:
                try:
                    self.vision_pub.publish(
                        build_detection2d_array(
                            dets, frame_id=output_frame_id, stamp=stamp
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(f"vision_msgs publish failed: {exc}")

            if self.visualize:
                ann = self.detector.annotate(bgr, dets)
                self.annotated_pub.publish(
                    bgr8_to_imgmsg(ann, frame_id=output_frame_id, stamp=stamp)
                )
                self._show_annotated_preview(source_id, ann)

            if self._frame_i % max(1, self.log_every_n) == 0:
                fps = 0.0
                if len(self._fps_times) >= 2:
                    dt = self._fps_times[-1] - self._fps_times[0]
                    if dt > 0:
                        fps = (len(self._fps_times) - 1) / dt
                orients = ",".join(d.orientation[0] for d in dets) or "-"
                self.get_logger().info(
                    f"n={len(dets)} orient={orients} {ms:.1f}ms ~{fps:.1f}fps {source_name}"
                )

        def _show_annotated_preview(
            self, source_id: str, annotated: np.ndarray
        ) -> None:
            if self._preview_failed:
                return
            self._annotated_frames[source_id] = annotated.copy()
            frame = self._compose_annotated_preview(source_id)
            if self._preview is None:
                title = (
                    "r2_10 detections - side by side"
                    if not self.run_offline and self.image_topic_2
                    else f"r2_10 detections - {source_id}"
                )
                self._preview = PreviewWindow(title)
            if self._preview.enabled and not self._preview.show(frame):
                if self._preview.user_closed:
                    self.get_logger().info(
                        "Annotated preview closed; detection continues"
                    )
                else:
                    self.get_logger().error(
                        "Annotated preview unavailable: "
                        f"{self._preview.last_error}. "
                        "Install tkinter/Pillow or set visualize:=false."
                    )
                self._preview_failed = True
                self._preview = None

        def _compose_annotated_preview(self, source_id: str) -> np.ndarray:
            if self.run_offline or not self.image_topic_2:
                return self._labeled_frame(
                    self._annotated_frames[source_id], source_id
                )

            available = self._annotated_frames[source_id]
            panels = []
            for camera in ("cam0", "cam1"):
                frame = self._annotated_frames.get(camera)
                if frame is None:
                    frame = np.zeros_like(available)
                    cv2.putText(
                        frame,
                        f"Waiting for {camera}...",
                        (20, max(40, frame.shape[0] // 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )
                panels.append(self._labeled_frame(frame, camera))

            target_h = min(panel.shape[0] for panel in panels)
            resized = []
            for panel in panels:
                if panel.shape[0] != target_h:
                    scale = target_h / float(panel.shape[0])
                    panel = cv2.resize(
                        panel, (max(1, int(panel.shape[1] * scale)), target_h)
                    )
                resized.append(panel)
            return cv2.hconcat(resized)

        @staticmethod
        def _labeled_frame(frame: np.ndarray, label: str) -> np.ndarray:
            labeled = frame.copy()
            cv2.putText(
                labeled,
                label,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            return labeled

        def destroy_node(self) -> bool:
            if self._preview is not None:
                self._preview.close()
                self._preview = None
            self._annotated_frames.clear()
            return super().destroy_node()

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
