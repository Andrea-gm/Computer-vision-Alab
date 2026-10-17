"""Subscribe to cam0 color and show furnace slot overlay.

Does NOT open CAP_OBSENSOR — safe while ROS publishers own the Orbbec USB.
Uses robotics/furnace_state/furnace.json fixed slots (keep boxes when occluded).
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
from .furnace_overlay_draw import (
    default_furnace_json,
    paint_furnace_overlay,
)
from .image_codec import imgmsg_to_numpy
from .preview_gui import PreviewWindow


class FurnaceOverlayNode(Node):
    def __init__(self) -> None:
        super().__init__("orbbec_furnace_overlay")
        self.declare_parameter("color_topic", "/orbbec/cam0/color/image_raw")
        self.declare_parameter("furnace_json", "")
        self.declare_parameter("show_preview", True)
        self.declare_parameter("match_live", True)
        self.declare_parameter("match_every_n", 2)
        self.declare_parameter("save_dir", "")
        self.declare_parameter("save_every_n", 0)

        topic = str(self.get_parameter("color_topic").value)
        raw = str(self.get_parameter("furnace_json").value).strip()
        self._robotics = ensure_camera_on_path()
        self.furnace_json = Path(raw) if raw else default_furnace_json(self._robotics)
        self.show_preview = bool(self.get_parameter("show_preview").value)
        self.match_live = bool(self.get_parameter("match_live").value)
        self.match_every_n = max(1, int(self.get_parameter("match_every_n").value))
        self.save_dir = str(self.get_parameter("save_dir").value).strip()
        self.save_every_n = int(self.get_parameter("save_every_n").value)
        self._frame_i = 0
        self._saved = 0
        self._last_view = None

        if not self.furnace_json.is_file():
            raise FileNotFoundError(f"furnace.json not found: {self.furnace_json}")
        self.get_logger().info(
            f"Furnace overlay on {topic}; json={self.furnace_json} "
            f"match_live={self.match_live}"
        )

        self._preview = (
            PreviewWindow("orbbec_furnace_overlay", max_height=720)
            if self.show_preview
            else None
        )
        if self._preview is not None:
            self.get_logger().info(
                "Preview ON — furnace boxes + labels on cam0. "
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

    def _on_image(self, msg: Image) -> None:
        try:
            bgr = imgmsg_to_numpy(msg)
            if bgr.ndim != 3:
                return
            self._frame_i += 1
            do_match = (
                self.match_live
                and (
                    self._frame_i % self.match_every_n == 0
                    or self._last_view is None
                )
            )
            if do_match or self._last_view is None:
                view = paint_furnace_overlay(
                    bgr,
                    robotics_root=self._robotics,
                    furnace_json=self.furnace_json,
                    match_live=self.match_live,
                    corner_tag="cam0 furnace overlay",
                )
                self._last_view = view
            else:
                # Re-draw fixed slots without re-matching (keeps boxes stable).
                view = paint_furnace_overlay(
                    bgr,
                    robotics_root=self._robotics,
                    furnace_json=self.furnace_json,
                    match_live=False,
                    corner_tag="cam0 furnace overlay",
                )
                self._last_view = view
            if self._preview is not None:
                self._preview.show(view)
            if self.save_dir and self.save_every_n > 0:
                if self._frame_i % self.save_every_n == 0:
                    out = Path(self.save_dir) / f"furnace_overlay_{self._saved:04d}.png"
                    cv2.imwrite(str(out), view)
                    self._saved += 1
                    self.get_logger().info(f"Saved {out}")
        except Exception as exc:
            self.get_logger().error(
                f"furnace overlay callback failed: {exc}\n{traceback.format_exc()}"
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
        node = FurnaceOverlayNode()
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
