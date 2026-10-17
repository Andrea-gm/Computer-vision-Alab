"""ROS2 node: multi-stage ensemble vial detector.

Subscribes to sensor_msgs/Image, publishes:
  - vision_msgs/Detection2DArray (or std_msgs/String JSON fallback)
  - sensor_msgs/Image annotated
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .ensemble_pipeline import EnsembleConfig, EnsembleVialDetector


def _bgr_from_image_msg(msg) -> np.ndarray:
    """Decode sensor_msgs/Image without cv_bridge (Robostack-friendly)."""
    h = int(msg.height)
    w = int(msg.width)
    encoding = (msg.encoding or "").lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if encoding in ("bgr8", "rgb8"):
        img = data.reshape(h, w, 3).copy()
        if encoding == "rgb8":
            img = img[:, :, ::-1].copy()
        return img
    if encoding in ("mono8", "8uc1"):
        gray = data.reshape(h, w)
        return np.stack([gray, gray, gray], axis=-1)
    # Best-effort: assume tightly packed BGR.
    if data.size >= h * w * 3:
        return data[: h * w * 3].reshape(h, w, 3).copy()
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def _image_msg_from_bgr(bgr: np.ndarray, header, ImageCls):
    msg = ImageCls()
    msg.header = header
    msg.height = int(bgr.shape[0])
    msg.width = int(bgr.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = int(bgr.shape[1] * 3)
    msg.data = bgr.tobytes()
    return msg


def _detections_to_vision_msgs(detections, header, stamp, Detection2DArray, Detection2D, ObjectHypothesisWithPose, Pose2D=None):
    arr = Detection2DArray()
    arr.header = header
    for det in detections:
        d = Detection2D()
        d.header = header
        x1, y1, x2, y2 = det.xyxy
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        # Humble+/vision_msgs: bbox.center is Pose2D or BoundingBox2D with center.position
        try:
            d.bbox.center.position.x = float(cx)
            d.bbox.center.position.y = float(cy)
            d.bbox.center.theta = 0.0
        except Exception:
            d.bbox.center.x = float(cx)
            d.bbox.center.y = float(cy)
            d.bbox.center.theta = 0.0
        d.bbox.size_x = float(bw)
        d.bbox.size_y = float(bh)
        hyp = ObjectHypothesisWithPose()
        try:
            hyp.hypothesis.class_id = "vial"
            hyp.hypothesis.score = float(det.score)
        except Exception:
            # Older vision_msgs layout
            hyp.id = "vial"
            hyp.score = float(det.score)
        d.results.append(hyp)
        arr.detections.append(d)
    return arr


def _detections_to_json(detections, frame_id: str) -> str:
    payload = {
        "frame_id": frame_id,
        "detections": [
            {
                "class_id": "vial",
                "score": float(d.score),
                "source": d.source,
                "bbox_xyxy": [float(x) for x in d.xyxy],
            }
            for d in detections
        ],
    }
    return json.dumps(payload)


class VialDetectorNode:
    """rclpy Node wrapper (instantiated in main so import works offline)."""

    def __init__(self, node):
        self.node = node

        # Declare parameters
        defaults = {
            "image_topic": "/camera/color/image_raw",
            "detections_topic": "/vial/detections",
            "annotated_topic": "/vial/annotated",
            "use_cap": True,
            "use_edge": True,
            "use_yolo": True,
            "yolo_weights": "",
            "yolo_conf": 0.10,
            "verifier_path": "",
            "verifier_thresh": 0.42,
            "nms_iou": 0.45,
            "use_soft_nms": True,
            "max_det": 5,
            "camera_root": "",
            "frame_id": "camera_color_optical_frame",
        }
        for k, v in defaults.items():
            node.declare_parameter(k, v)

        gp = node.get_parameter
        yolo_w = str(gp("yolo_weights").value or "").strip() or None
        vpath = str(gp("verifier_path").value or "").strip() or None
        cam_root = str(gp("camera_root").value or "").strip() or None

        cfg = EnsembleConfig(
            use_cap=bool(gp("use_cap").value),
            use_edge=bool(gp("use_edge").value),
            use_yolo=bool(gp("use_yolo").value),
            yolo_weights=yolo_w,
            yolo_conf=float(gp("yolo_conf").value),
            verifier_path=vpath,
            verifier_thresh=float(gp("verifier_thresh").value),
            nms_iou=float(gp("nms_iou").value),
            use_soft_nms=bool(gp("use_soft_nms").value),
            max_det=int(gp("max_det").value),
            camera_root=cam_root,
        )
        self.detector = EnsembleVialDetector(cfg)
        self.frame_id = str(gp("frame_id").value)
        self._image_topic = str(gp("image_topic").value)
        self._det_topic = str(gp("detections_topic").value)
        self._ann_topic = str(gp("annotated_topic").value)

        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        self._Image = Image
        self._String = String
        self._vision = None
        try:
            from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

            self._vision = {
                "Detection2DArray": Detection2DArray,
                "Detection2D": Detection2D,
                "ObjectHypothesisWithPose": ObjectHypothesisWithPose,
            }
            self.det_pub = node.create_publisher(Detection2DArray, self._det_topic, 10)
            node.get_logger().info("Publishing vision_msgs/Detection2DArray")
        except Exception as exc:
            self.det_pub = node.create_publisher(String, self._det_topic, 10)
            node.get_logger().warn(
                f"vision_msgs unavailable ({exc}); publishing JSON on {self._det_topic}"
            )

        self.ann_pub = node.create_publisher(Image, self._ann_topic, 10)
        self.json_pub = node.create_publisher(String, self._det_topic + "_json", 10)
        self.sub = node.create_subscription(Image, self._image_topic, self.on_image, 10)

        node.get_logger().info(
            f"Ensemble vial detector ready | yolo={self.detector.yolo_weights or 'off'} | "
            f"verifier={self.detector.verifier.model_path or 'heuristic'} | "
            f"sub={self._image_topic}"
        )

    def on_image(self, msg) -> None:
        try:
            bgr = _bgr_from_image_msg(msg)
        except Exception as exc:
            self.node.get_logger().error(f"image decode failed: {exc}")
            return
        dets = self.detector.detect(bgr)
        header = msg.header
        if not header.frame_id:
            header.frame_id = self.frame_id

        if self._vision is not None:
            arr = _detections_to_vision_msgs(
                dets,
                header,
                header.stamp,
                self._vision["Detection2DArray"],
                self._vision["Detection2D"],
                self._vision["ObjectHypothesisWithPose"],
            )
            self.det_pub.publish(arr)
        else:
            from std_msgs.msg import String

            m = String()
            m.data = _detections_to_json(dets, header.frame_id)
            self.det_pub.publish(m)

        j = self._String()
        j.data = _detections_to_json(dets, header.frame_id)
        self.json_pub.publish(j)

        ann = self.detector.annotate(bgr, dets)
        self.ann_pub.publish(_image_msg_from_bgr(ann, header, self._Image))


def main(args: Any = None) -> None:
    import rclpy
    from rclpy.node import Node

    rclpy.init(args=args)
    node = Node("vial_detector_node")
    _ = VialDetectorNode(node)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
