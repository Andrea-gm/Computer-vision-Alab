"""Subscribe to cam1 color and show full-rack OPEN/CLOSED overlay.

Does NOT open CAP_OBSENSOR — safe while ROS publishers own the Orbbec USB.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image

from .camera_path import ensure_camera_on_path
from .image_codec import imgmsg_to_numpy
from .preview_gui import PreviewWindow
from .rack_overlay_draw import (
    classify_whole_rack,
    draw_rack_overlay,
    load_rack_rois,
)


def _default_rack_json() -> Path:
    root = ensure_camera_on_path()
    return root / "rack_lid_state" / "rack.json"


class RackOverlayNode(Node):
    def __init__(self) -> None:
        super().__init__("orbbec_rack_overlay")
        self.declare_parameter("color_topic", "/orbbec/cam1/color/image_raw")
        self.declare_parameter("rack_json", str(_default_rack_json()))
        self.declare_parameter("show_preview", True)
        # force_closed | force_open | classify — default classify (real CV).
        self.declare_parameter("state_mode", "classify")
        self.declare_parameter("force_state", "CLOSED")
        self.declare_parameter("save_dir", "")
        self.declare_parameter("save_every_n", 0)

        topic = str(self.get_parameter("color_topic").value)
        rack_raw = str(self.get_parameter("rack_json").value).strip()
        rack_path = Path(rack_raw) if rack_raw else _default_rack_json()
        self.show_preview = bool(self.get_parameter("show_preview").value)
        self.state_mode = str(self.get_parameter("state_mode").value).strip().lower()
        self.force_state = str(self.get_parameter("force_state").value).strip().upper()
        self.save_dir = str(self.get_parameter("save_dir").value).strip()
        self.save_every_n = int(self.get_parameter("save_every_n").value)
        self._frame_i = 0
        self._saved = 0
        self._robotics = ensure_camera_on_path()

        if not rack_path.is_file():
            raise FileNotFoundError(f"rack.json not found: {rack_path}")
        self.classify_xywh, self.display_xywh, self.holes = load_rack_rois(rack_path)
        self.get_logger().info(
            f"Rack overlay on {topic}; classify={self.classify_xywh} "
            f"display={self.display_xywh} mode={self.state_mode}"
        )

        self._preview = (
            PreviewWindow("orbbec_rack_overlay", max_height=720)
            if self.show_preview
            else None
        )
        if self._preview is not None:
            self.get_logger().info(
                "Preview ON — full rack box + OPEN/CLOSED. "
                "Q / close hides UI only; Ctrl+C stops launch."
            )
        if self.save_dir:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Image, topic, self._on_image, image_qos)

    def _resolve_state(self, bgr) -> str:
        if self.state_mode == "force_open":
            return "OPEN"
        if self.state_mode == "force_closed":
            return "CLOSED"
        if self.state_mode in ("force", "fixed"):
            return self.force_state or "CLOSED"
        try:
            return classify_whole_rack(
                bgr,
                self.classify_xywh,
                self.holes,
                robotics_root=self._robotics,
            )
        except Exception as exc:
            self.get_logger().warn(f"classify failed, using CLOSED: {exc}")
            return "CLOSED"

    def _on_image(self, msg: Image) -> None:
        try:
            bgr = imgmsg_to_numpy(msg)
            if bgr.ndim != 3:
                return
            state = self._resolve_state(bgr)
            view = draw_rack_overlay(
                bgr,
                self.display_xywh,
                state,
                classify_xywh=self.classify_xywh,
            )
            self._frame_i += 1
            if self._preview is not None:
                self._preview.show(view)
            if self.save_dir and self.save_every_n > 0:
                if self._frame_i % self.save_every_n == 0:
                    out = Path(self.save_dir) / f"rack_overlay_{self._saved:04d}.png"
                    cv2.imwrite(str(out), view)
                    self._saved += 1
                    self.get_logger().info(f"Saved {out}")
        except Exception as exc:
            self.get_logger().error(
                f"rack overlay callback failed: {exc}\n{traceback.format_exc()}"
            )

    def destroy_node(self) -> bool:
        if self._preview is not None:
            try:
                self._preview.close()
            except Exception:
                pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = RackOverlayNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
