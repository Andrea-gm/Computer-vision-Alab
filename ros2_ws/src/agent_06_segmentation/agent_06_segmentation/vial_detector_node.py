"""ROS2 node: instance-segmentation vial detector.

Publishes:
  - detections (vision_msgs/Detection2DArray when available, else skipped)
  - detections_json (std_msgs/String JSON)
  - annotated / mask overlay (sensor_msgs/Image, optional)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from agent_06_segmentation.image_codec import bgr8_to_imgmsg, imgmsg_to_numpy
from agent_06_segmentation.segmenter import (
    CAMERA_ROOT_DEFAULT,
    InstanceDet,
    VialSegmenter,
    detections_to_json,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class VialDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("vial_detector_node")

        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter(
            "dataset_path", str(CAMERA_ROOT_DEFAULT / "vial_dataset" / "raw")
        )
        self.declare_parameter("run_offline", True)
        self.declare_parameter("model_path", "")
        self.declare_parameter("camera_root", str(CAMERA_ROOT_DEFAULT))
        self.declare_parameter("confidence_threshold", 0.25)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("mask_fallback", "ellipse")  # ellipse|rect|none
        self.declare_parameter("visualize", True)
        self.declare_parameter("publish_mask_overlay", True)
        self.declare_parameter("annotated_topic", "/vial/annotated")
        self.declare_parameter("mask_overlay_topic", "/vial/mask_overlay")
        self.declare_parameter("detections_topic", "/vial/detections")
        self.declare_parameter("detections_json_topic", "/vial/detections_json")
        self.declare_parameter("frame_id", "camera_color_optical_frame")
        self.declare_parameter("offline_rate_hz", 5.0)
        self.declare_parameter("offline_loop", False)
        self.declare_parameter("log_every_n", 10)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.dataset_path = Path(str(self.get_parameter("dataset_path").value))
        self.run_offline = bool(self.get_parameter("run_offline").value)
        model_path = str(self.get_parameter("model_path").value).strip()
        camera_root = Path(str(self.get_parameter("camera_root").value))
        conf = float(self.get_parameter("confidence_threshold").value)
        iou = float(self.get_parameter("iou_threshold").value)
        imgsz = int(self.get_parameter("imgsz").value)
        mask_fallback = str(self.get_parameter("mask_fallback").value)
        self.visualize = bool(self.get_parameter("visualize").value)
        self.publish_mask_overlay = bool(
            self.get_parameter("publish_mask_overlay").value
        )
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.offline_rate_hz = float(self.get_parameter("offline_rate_hz").value)
        self.offline_loop = bool(self.get_parameter("offline_loop").value)
        self.log_every_n = max(1, int(self.get_parameter("log_every_n").value))

        self._frame_i = 0
        self._offline_files: List[Path] = []
        self._offline_idx = 0

        self.segmenter = VialSegmenter(
            model_path=model_path or None,
            camera_root=camera_root,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            mask_fallback=mask_fallback,
        )
        try:
            self.segmenter.load()
            self.get_logger().info(
                f"Loaded model={self.segmenter.resolved_path} "
                f"kind={self.segmenter.model_kind} "
                f"mask_head={self.segmenter._has_mask_head} "
                f"fallback={mask_fallback}"
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to load segmentation model: {exc}")
            raise

        self.json_pub = self.create_publisher(
            String, str(self.get_parameter("detections_json_topic").value), 10
        )
        self.annot_pub = None
        self.mask_pub = None
        if self.visualize:
            self.annot_pub = self.create_publisher(
                Image, str(self.get_parameter("annotated_topic").value), 10
            )
        if self.publish_mask_overlay:
            self.mask_pub = self.create_publisher(
                Image, str(self.get_parameter("mask_overlay_topic").value), 10
            )

        self._det_array_pub = None
        self._Detection2DArray = None
        self._try_vision_msgs(
            str(self.get_parameter("detections_topic").value)
        )

        if self.run_offline:
            self._offline_files = self._list_images(self.dataset_path)
            if not self._offline_files:
                self.get_logger().error(
                    f"No images under dataset_path={self.dataset_path}"
                )
            else:
                self.get_logger().info(
                    f"Offline mode: {len(self._offline_files)} images in "
                    f"{self.dataset_path}"
                )
            period = (
                1.0 / self.offline_rate_hz if self.offline_rate_hz > 0 else 0.001
            )
            self._timer = self.create_timer(period, self._offline_tick)
        else:
            self.create_subscription(
                Image, self.image_topic, self._on_image, 10
            )
            self.get_logger().info(f"Subscribed to {self.image_topic}")

    def _try_vision_msgs(self, topic: str) -> None:
        try:
            from vision_msgs.msg import Detection2DArray  # type: ignore

            self._Detection2DArray = Detection2DArray
            self._det_array_pub = self.create_publisher(Detection2DArray, topic, 10)
            self.get_logger().info(f"Publishing Detection2DArray on {topic}")
        except Exception:  # noqa: BLE001
            self.get_logger().warn(
                "vision_msgs not available; using JSON detections only"
            )

    @staticmethod
    def _list_images(folder: Path) -> List[Path]:
        if not folder.is_dir():
            return []
        files = [
            p
            for p in sorted(folder.iterdir())
            if p.suffix.lower() in IMAGE_EXTS and p.is_file()
        ]
        return files

    def _offline_tick(self) -> None:
        if not self._offline_files:
            return
        if self._offline_idx >= len(self._offline_files):
            if self.offline_loop:
                self._offline_idx = 0
            else:
                self.get_logger().info("Offline dataset complete; shutting down.")
                raise SystemExit(0)
        path = self._offline_files[self._offline_idx]
        self._offline_idx += 1
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn(f"Failed to read {path}")
            return
        stamp = self.get_clock().now().to_msg()
        self._process(bgr, stamp, source=str(path.name))

    def _on_image(self, msg: Image) -> None:
        try:
            bgr = imgmsg_to_numpy(msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Bad image: {exc}")
            return
        self._process(bgr, msg.header.stamp, source=msg.header.frame_id)

    def _process(self, bgr, stamp, *, source: str) -> None:
        t0 = time.perf_counter()
        dets = self.segmenter.predict(bgr)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._frame_i += 1

        meta = {
            "source": source,
            "model": str(self.segmenter.resolved_path),
            "model_kind": self.segmenter.model_kind,
            "infer_ms": round(dt_ms, 2),
            "agent": "agent_06_segmentation",
        }
        self.json_pub.publish(String(data=detections_to_json(dets, **meta)))

        if self._det_array_pub is not None and self._Detection2DArray is not None:
            self._det_array_pub.publish(self._to_detection2d_array(dets, stamp))

        if self.annot_pub is not None:
            annotated = self.segmenter.annotate(bgr, dets)
            self.annot_pub.publish(
                bgr8_to_imgmsg(annotated, frame_id=self.frame_id, stamp=stamp)
            )

        if self.mask_pub is not None:
            overlay = self._mask_only_overlay(bgr, dets)
            self.mask_pub.publish(
                bgr8_to_imgmsg(overlay, frame_id=self.frame_id, stamp=stamp)
            )

        if self._frame_i % self.log_every_n == 0:
            sources = {d.mask_source for d in dets}
            self.get_logger().info(
                f"[{self._frame_i}] n={len(dets)} {dt_ms:.1f}ms "
                f"mask_src={sources or {'none'}} src={source}"
            )

    def _mask_only_overlay(
        self, bgr, dets: List[InstanceDet]
    ):
        canvas = bgr.copy()
        tint = canvas.copy()
        for i, d in enumerate(dets):
            if d.mask is None:
                continue
            color = (40 + 40 * (i % 5), 180 - 20 * (i % 4), 80 + 30 * (i % 3))
            tint[d.mask.astype(bool)] = color
        cv2.addWeighted(tint, 0.5, canvas, 0.5, 0, canvas)
        return canvas

    def _to_detection2d_array(self, dets: List[InstanceDet], stamp):
        from vision_msgs.msg import BoundingBox2D, Detection2D, ObjectHypothesisWithPose

        arr = self._Detection2DArray()
        arr.header.stamp = stamp
        arr.header.frame_id = self.frame_id
        for d in dets:
            det = Detection2D()
            det.header = arr.header
            hyp = ObjectHypothesisWithPose()
            # Humble+: hypothesis nested; older stacks may use hyp.id / hyp.score
            if hasattr(hyp, "hypothesis"):
                hyp.hypothesis.class_id = str(d.class_name)
                hyp.hypothesis.score = float(d.conf)
            else:
                hyp.id = str(d.class_name)
                hyp.score = float(d.conf)
            det.results.append(hyp)
            bb = BoundingBox2D()
            # Pose2D location differs across vision_msgs versions
            center = bb.center
            if hasattr(center, "position"):
                center.position.x = float(d.x + d.w / 2.0)
                center.position.y = float(d.y + d.h / 2.0)
                if hasattr(center, "theta"):
                    center.theta = 0.0
            else:
                center.x = float(d.x + d.w / 2.0)
                center.y = float(d.y + d.h / 2.0)
                if hasattr(center, "theta"):
                    center.theta = 0.0
            bb.size_x = float(d.w)
            bb.size_y = float(d.h)
            det.bbox = bb
            arr.detections.append(det)
        return arr


def main(args=None) -> None:
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
