#!/usr/bin/env python3
"""ROS2 node: HOG + Linear SVM vial detector (dataset offline or live Image)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
except ImportError as exc:  # pragma: no cover - documented Windows fallback
    rclpy = None
    Node = object  # type: ignore
    Image = None  # type: ignore
    String = None  # type: ignore
    _RCLPY_IMPORT_ERROR = exc
else:
    _RCLPY_IMPORT_ERROR = None

try:
    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
    _HAS_VISION_MSGS = True
except ImportError:
    Detection2D = Detection2DArray = ObjectHypothesisWithPose = None  # type: ignore
    _HAS_VISION_MSGS = False

from agent_07_hog_svm.hog_detector import Detection, HogSvmDetector, resolve_default_model


def _bgr_to_image_msg(node: "VialDetectorNode", bgr: np.ndarray, frame_id: str = "camera") -> "Image":
    msg = Image()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = frame_id
    msg.height, msg.width = bgr.shape[:2]
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = bgr.tobytes()
    return msg


def _image_msg_to_bgr(msg: "Image") -> np.ndarray:
    if msg.encoding in ("bgr8", "rgb8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr.copy()
    if msg.encoding in ("mono8",):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported encoding: {msg.encoding}")


def detections_to_json(dets: List[Detection]) -> str:
    return json.dumps({"detections": [d.as_dict() for d in dets], "count": len(dets)})


class VialDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("vial_detector_node")

        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("dataset_path", "")
        self.declare_parameter("model_path", str(resolve_default_model()))
        self.declare_parameter("confidence_threshold", 0.0)
        self.declare_parameter("visualize", True)
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("scale_factor", 1.25)
        self.declare_parameter("step_size", 8)
        self.declare_parameter("max_image_side", 480)
        self.declare_parameter("nms_iou", 0.3)
        self.declare_parameter("frame_id", "camera")

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        dataset_path = self.get_parameter("dataset_path").get_parameter_value().string_value
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        conf = self.get_parameter("confidence_threshold").get_parameter_value().double_value
        self.visualize = self.get_parameter("visualize").get_parameter_value().bool_value
        rate_hz = self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        scale_factor = self.get_parameter("scale_factor").get_parameter_value().double_value
        step_size = int(self.get_parameter("step_size").get_parameter_value().integer_value)
        max_side = int(self.get_parameter("max_image_side").get_parameter_value().integer_value)
        nms_iou = self.get_parameter("nms_iou").get_parameter_value().double_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value

        model_file = Path(model_path) if model_path else Path()
        if not model_path or not model_file.is_file():
            fallback = resolve_default_model()
            if model_path:
                self.get_logger().warn(f"model_path={model_path!r} missing; using {fallback}")
            model_path = str(fallback)

        self.detector = HogSvmDetector(
            model_path=model_path,
            confidence_threshold=conf,
            scale_factor=scale_factor,
            step_size=step_size,
            max_image_side=max_side,
            nms_iou=nms_iou,
        )
        self.get_logger().info(f"Loaded HOG-SVM model: {model_path}")

        self.pub_annot = self.create_publisher(Image, "vial/annotated", 10)
        self.pub_json = self.create_publisher(String, "vial/detections_json", 10)
        self.pub_det = None
        if _HAS_VISION_MSGS:
            self.pub_det = self.create_publisher(Detection2DArray, "vial/detections", 10)
            self.get_logger().info("Publishing vision_msgs/Detection2DArray on vial/detections")
        else:
            self.get_logger().warn(
                "vision_msgs not available — publishing JSON on vial/detections_json only"
            )

        self._dataset_images: List[Path] = []
        self._dataset_idx = 0
        if dataset_path:
            root = Path(dataset_path)
            if root.is_file():
                self._dataset_images = [root]
            else:
                for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                    self._dataset_images.extend(sorted(root.rglob(pattern)))
            if not self._dataset_images:
                self.get_logger().error(f"No images under dataset_path={dataset_path}")
            else:
                self.get_logger().info(
                    f"Offline dataset mode: {len(self._dataset_images)} images from {dataset_path}"
                )
                period = 1.0 / max(rate_hz, 0.1)
                self.create_timer(period, self._dataset_tick)
        else:
            self.create_subscription(Image, image_topic, self._image_cb, 10)
            self.get_logger().info(f"Subscribing to Image topic: {image_topic}")

        self._frames = 0
        self._t0 = time.time()

    def _publish(self, bgr: np.ndarray, dets: List[Detection]) -> None:
        self.pub_json.publish(String(data=detections_to_json(dets)))
        if self.pub_det is not None:
            self.pub_det.publish(self._to_detection2d_array(dets))
        if self.visualize:
            annot = self.detector.annotate(bgr, dets)
        else:
            annot = bgr
        self.pub_annot.publish(_bgr_to_image_msg(self, annot, self.frame_id))

        self._frames += 1
        elapsed = max(1e-6, time.time() - self._t0)
        fps = self._frames / elapsed
        now = time.time()
        if now - getattr(self, "_last_log_t", 0.0) >= 1.0:
            self._last_log_t = now
            self.get_logger().info(f"detections={len(dets)} fps={fps:.2f}")

    def _to_detection2d_array(self, dets: List[Detection]) -> "Detection2DArray":
        arr = Detection2DArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.header.frame_id = self.frame_id
        for d in dets:
            det = Detection2D()
            det.header = arr.header
            det.bbox.center.position.x = float(d.x + d.w / 2.0)
            det.bbox.center.position.y = float(d.y + d.h / 2.0)
            det.bbox.size_x = float(d.w)
            det.bbox.size_y = float(d.h)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = "vial"
            hyp.hypothesis.score = float(d.score)
            det.results.append(hyp)
            arr.detections.append(det)
        return arr

    def _image_cb(self, msg: "Image") -> None:
        try:
            bgr = _image_msg_to_bgr(msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Image decode failed: {exc}")
            return
        t1 = time.time()
        dets = self.detector.detect(bgr)
        _ = time.time() - t1
        self._publish(bgr, dets)

    def _dataset_tick(self) -> None:
        if not self._dataset_images:
            return
        path = self._dataset_images[self._dataset_idx % len(self._dataset_images)]
        self._dataset_idx += 1
        bgr = cv2.imread(str(path))
        if bgr is None:
            self.get_logger().warn(f"Failed to read {path}")
            return
        dets = self.detector.detect(bgr)
        self._publish(bgr, dets)


def main(args=None) -> None:
    if rclpy is None:
        raise SystemExit(
            "rclpy is not installed in this Python environment. "
            "Source your ROS2 setup (or use WSL2/Linux) then: "
            "ros2 run agent_07_hog_svm vial_detector_node. "
            f"Import error: {_RCLPY_IMPORT_ERROR}"
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
