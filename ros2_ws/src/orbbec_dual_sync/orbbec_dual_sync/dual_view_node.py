"""Live side-by-side preview: cam0 furnace CV | cam1 rack OPEN/CLOSED CV.

Subscribes only — does not open CAP_OBSENSOR. Furnace overlay on the left pane
(same helpers as furnace_agents / orbbec_furnace_overlay). Rack overlay on the
right pane (same as orbbec_rack_overlay).
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
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
from .rack_overlay_draw import (
    classify_whole_rack,
    draw_rack_overlay,
    load_rack_rois,
)


def _label(bgr: np.ndarray, text: str) -> np.ndarray:
    out = bgr.copy()
    cv2.putText(
        out,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _placeholder(lines: list[str], h: int = 480, w: int = 640) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    y0 = h // 2 - 12 * (len(lines) - 1)
    for i, text in enumerate(lines):
        cv2.putText(
            img,
            text,
            (24, y0 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (80, 80, 255),
            2,
            cv2.LINE_AA,
        )
    return img


def compose_side_by_side(
    left: Optional[np.ndarray],
    right: Optional[np.ndarray],
    *,
    max_height: int = 480,
    left_status: str = "",
    right_status: str = "",
    left_already_labeled: bool = False,
    right_already_labeled: bool = False,
) -> np.ndarray:
    if left is not None:
        a = left if left_already_labeled else _label(left, "cam0")
    else:
        a = _placeholder(["waiting cam0…", left_status] if left_status else ["waiting cam0…"])
    if right is not None:
        b = right if right_already_labeled else _label(right, "cam1")
    else:
        b = _placeholder(
            ["waiting cam1…", right_status] if right_status else ["waiting cam1…"]
        )
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = min(ha, hb, max_height) if max_height > 0 else min(ha, hb)
    a_s = cv2.resize(a, (max(1, int(wa * (h / float(ha)))), h))
    b_s = cv2.resize(b, (max(1, int(wb * (h / float(hb)))), h))
    return cv2.hconcat([a_s, b_s])


def _default_rack_json() -> Path:
    root = ensure_camera_on_path()
    return root / "rack_lid_state" / "rack.json"


class DualViewNode(Node):
    def __init__(self) -> None:
        super().__init__("orbbec_dual_view")
        self.declare_parameter("cam0_color_topic", "/orbbec/cam0/color/image_raw")
        self.declare_parameter("cam1_color_topic", "/orbbec/cam1/color/image_raw")
        self.declare_parameter("cam0_depth_topic", "/orbbec/cam0/depth/image_raw")
        self.declare_parameter("cam1_depth_topic", "/orbbec/cam1/depth/image_raw")
        self.declare_parameter("max_height", 480)
        self.declare_parameter("refresh_hz", 15.0)
        # Cam1 rack OPEN/CLOSED CV (same as orbbec_rack_overlay).
        self.declare_parameter("enable_rack_cv", True)
        self.declare_parameter("rack_json", "")
        self.declare_parameter("state_mode", "classify")
        self.declare_parameter("force_state", "CLOSED")
        self.declare_parameter("classify_every_n", 2)
        # Cam0 furnace CV (same helpers as furnace_agents / orbbec_furnace_overlay).
        self.declare_parameter("enable_furnace_cv", True)
        self.declare_parameter("furnace_json", "")
        self.declare_parameter("furnace_match_live", True)
        self.declare_parameter("furnace_match_every_n", 2)
        # YOLO arm detector (arm_state/runs/arm_detect/weights/best.pt)
        self.declare_parameter("enable_arm_cv", True)
        self.declare_parameter("arm_weights", "")
        self.declare_parameter("arm_conf", 0.25)
        self.declare_parameter("arm_every_n", 3)
        # Pale baby-blue color mask tracker (no NN) — UR joint plastic.
        self.declare_parameter("enable_blue_cv", True)
        self.declare_parameter("blue_show_mask", True)
        self.declare_parameter("blue_min_area", 280)
        self.declare_parameter("blue_every_n", 1)
        self.declare_parameter("blue_proximity_mm", 100.0)  # 10 cm red alert

        t0 = str(self.get_parameter("cam0_color_topic").value)
        t1 = str(self.get_parameter("cam1_color_topic").value)
        self.cam0_depth_topic = str(self.get_parameter("cam0_depth_topic").value)
        self.cam1_depth_topic = str(self.get_parameter("cam1_depth_topic").value)
        self.max_height = int(self.get_parameter("max_height").value)
        hz = float(self.get_parameter("refresh_hz").value)
        period = 1.0 / max(hz, 1.0)

        self.enable_rack_cv = bool(self.get_parameter("enable_rack_cv").value)
        self.state_mode = str(self.get_parameter("state_mode").value).strip().lower()
        self.force_state = str(self.get_parameter("force_state").value).strip().upper()
        self.classify_every_n = max(1, int(self.get_parameter("classify_every_n").value))

        self.enable_furnace_cv = bool(self.get_parameter("enable_furnace_cv").value)
        self.furnace_match_live = bool(self.get_parameter("furnace_match_live").value)
        self.furnace_match_every_n = max(
            1, int(self.get_parameter("furnace_match_every_n").value)
        )
        self.furnace_json: Optional[Path] = None

        self.enable_arm_cv = bool(self.get_parameter("enable_arm_cv").value)
        arm_w = str(self.get_parameter("arm_weights").value).strip()
        self.arm_weights: Optional[Path] = Path(arm_w) if arm_w else None
        self.arm_conf = float(self.get_parameter("arm_conf").value)
        self.arm_every_n = max(1, int(self.get_parameter("arm_every_n").value))
        self._arm_cache0: Optional[np.ndarray] = None
        self._arm_cache1: Optional[np.ndarray] = None

        self.enable_blue_cv = bool(self.get_parameter("enable_blue_cv").value)
        self.blue_show_mask = bool(self.get_parameter("blue_show_mask").value)
        self.blue_min_area = max(20, int(self.get_parameter("blue_min_area").value))
        self.blue_every_n = max(1, int(self.get_parameter("blue_every_n").value))
        self.blue_proximity_mm = max(
            10.0, float(self.get_parameter("blue_proximity_mm").value)
        )
        self._blue_boxes0 = None
        self._blue_boxes1 = None
        self._blue_alert0 = False
        self._blue_alert1 = False
        self._furnace_slots: list[tuple[int, int, int, int, int]] = []
        self._latest_depth0: Optional[np.ndarray] = None
        self._latest_depth1: Optional[np.ndarray] = None

        self._latest0: Optional[np.ndarray] = None
        self._latest0_cv: Optional[np.ndarray] = None
        self._latest1: Optional[np.ndarray] = None
        self._latest1_cv: Optional[np.ndarray] = None
        self._recv0 = 0
        self._recv1 = 0
        self._cam0_seq = 0
        self._cam1_seq = 0
        self._last_state = "CLOSED"
        self._started = time.monotonic()
        self._warned_empty = False
        self._preview_error_logged = False
        self._should_exit = False
        self._robotics: Optional[Path] = None
        self.classify_xywh = (226, 55, 137, 152)
        self.display_xywh = (186, 30, 220, 260)
        self.holes: list = []

        need_robotics = (
            self.enable_rack_cv or self.enable_furnace_cv or self.enable_arm_cv
        )
        if need_robotics:
            try:
                self._robotics = ensure_camera_on_path()
            except Exception as exc:
                self.get_logger().error(f"Could not resolve robotics root: {exc}")
                self._robotics = None

        if self.enable_arm_cv:
            if self._robotics is None:
                self.enable_arm_cv = False
                self.get_logger().error("Arm CV disabled — robotics root unavailable")
            else:
                w = self.arm_weights or (
                    self._robotics
                    / "arm_state"
                    / "runs"
                    / "arm_detect"
                    / "weights"
                    / "best.pt"
                )
                if not w.is_file():
                    self.enable_arm_cv = False
                    self.get_logger().error(f"Arm CV disabled — missing weights {w}")
                else:
                    self.arm_weights = w
                    self.get_logger().info(
                        f"Arm YOLO ON both panes: conf={self.arm_conf} "
                        f"every_n={self.arm_every_n} weights={w}"
                    )

        if self.enable_blue_cv:
            self.get_logger().info(
                f"Baby-blue color mask ON both panes: min_area={self.blue_min_area} "
                f"show_mask={self.blue_show_mask}; RED alert if blue < "
                f"{self.blue_proximity_mm:.0f}mm of black/silver equipment "
                f"(mask hidden; 3D/depth; base joint ignored)"
            )

        if self.enable_rack_cv:
            try:
                if self._robotics is None:
                    raise FileNotFoundError("robotics root unavailable")
                rack_raw = str(self.get_parameter("rack_json").value).strip()
                rack_path = Path(rack_raw) if rack_raw else _default_rack_json()
                if not rack_path.is_file():
                    raise FileNotFoundError(f"rack.json not found: {rack_path}")
                self.classify_xywh, self.display_xywh, self.holes = load_rack_rois(
                    rack_path
                )
                self.get_logger().info(
                    f"Rack CV ON cam1 pane: mode={self.state_mode} "
                    f"classify={self.classify_xywh} display={self.display_xywh} "
                    f"json={rack_path}"
                )
            except Exception as exc:
                self.enable_rack_cv = False
                self.get_logger().error(
                    f"Rack CV disabled (could not load ROI/templates): {exc}"
                )

        if self.enable_furnace_cv:
            try:
                if self._robotics is None:
                    raise FileNotFoundError("robotics root unavailable")
                furnace_raw = str(self.get_parameter("furnace_json").value).strip()
                self.furnace_json = (
                    Path(furnace_raw) if furnace_raw else default_furnace_json(self._robotics)
                )
                if not self.furnace_json.is_file():
                    raise FileNotFoundError(
                        f"furnace.json not found: {self.furnace_json}"
                    )
                self.get_logger().info(
                    f"Furnace CV ON cam0 pane: match_live={self.furnace_match_live} "
                    f"json={self.furnace_json}"
                )
            except Exception as exc:
                self.enable_furnace_cv = False
                self.furnace_json = None
                self.get_logger().error(
                    f"Furnace CV disabled (could not load furnace.json): {exc}"
                )

        title_bits = ["cam0"]
        if self.enable_furnace_cv:
            title_bits[0] = "cam0 + furnace CV"
        title_bits.append("cam1 + rack CV" if self.enable_rack_cv else "cam1")
        self._preview = PreviewWindow(
            f"orbbec_dual_view ({' | '.join(title_bits)})",
            max_height=max(self.max_height, 480),
        )
        extras = []
        if self.enable_furnace_cv:
            extras.append("furnace boxes on cam0")
        if self.enable_rack_cv:
            extras.append("rack OPEN/CLOSED on cam1")
        if self.enable_arm_cv:
            extras.append("arm YOLO on both panes")
        if self.enable_blue_cv:
            extras.append(
                f"baby-blue vs black/silver equipment ({self.blue_proximity_mm:.0f}mm 3D alert)"
            )

        extra = (" + " + "; ".join(extras) + ".") if extras else "."
        self.get_logger().info(
            f"Dual view: {t0} | {t1}{extra} Q / Esc / close exits."
        )

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Image, t0, self._on_cam0, image_qos)
        self.create_subscription(Image, t1, self._on_cam1, image_qos)
        self.create_subscription(
            Image, self.cam0_depth_topic, self._on_depth0, image_qos
        )
        self.create_subscription(
            Image, self.cam1_depth_topic, self._on_depth1, image_qos
        )
        self.create_timer(period, self._on_timer)

    def _resolve_state(self, bgr: np.ndarray) -> str:
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
            self.get_logger().warn(f"classify failed, using last={self._last_state}: {exc}")
            return self._last_state

    def _paint_cam0_cv(self, bgr: np.ndarray) -> np.ndarray:
        self._cam0_seq += 1
        if self.enable_furnace_cv:
            do_match = self.furnace_match_live and (
                self._cam0_seq % self.furnace_match_every_n == 0
                or self._latest0_cv is None
            )
            try:
                from .furnace_overlay_draw import (
                    draw_furnace_overlay_on_frame,
                    resolve_slots_for_frame,
                )

                slots = resolve_slots_for_frame(
                    bgr,
                    robotics_root=self._robotics,
                    furnace_json=self.furnace_json,
                    match_live=do_match,
                )
                self._furnace_slots = [
                    (int(s.id), int(s.x), int(s.y), int(s.w), int(s.h)) for s in slots
                ]
                view = draw_furnace_overlay_on_frame(
                    bgr,
                    slots,
                    robotics_root=self._robotics,
                    corner_tag="cam0 + furnace + arm",
                )
            except Exception as exc:
                self.get_logger().warn(f"furnace resolve failed: {exc}")
                self._furnace_slots = []
                view = paint_furnace_overlay(
                    bgr,
                    robotics_root=self._robotics,
                    furnace_json=self.furnace_json,
                    match_live=do_match,
                    corner_tag="cam0 + furnace + arm",
                )
        else:
            self._furnace_slots = []
            view = _label(bgr, "cam0")
        view = self._overlay_arm_on(view, bgr, cam="cam0")
        return self._overlay_blue_on(view, bgr, cam="cam0")

    def _overlay_blue_on(
        self, decorated: np.ndarray, raw: np.ndarray, *, cam: str
    ) -> np.ndarray:
        """HSV baby-blue boxes labeled with Orbbec depth distance (not confidence)."""
        alert_attr = "_blue_alert0" if cam == "cam0" else "_blue_alert1"
        if not self.enable_blue_cv:
            setattr(self, alert_attr, False)
            return decorated
        seq = self._cam0_seq if cam == "cam0" else self._cam1_seq
        cache_attr = "_blue_boxes0" if cam == "cam0" else "_blue_boxes1"
        depth = self._latest_depth0 if cam == "cam0" else self._latest_depth1
        if seq % self.blue_every_n == 0 or getattr(self, cache_attr) is None:
            try:
                from .baby_blue_draw import find_baby_blue_boxes

                setattr(
                    self,
                    cache_attr,
                    find_baby_blue_boxes(
                        raw, min_area=self.blue_min_area, depth_mm=depth
                    ),
                )
            except Exception as exc:
                self.get_logger().warn(f"blue mask {cam}: {exc}")
                setattr(self, cache_attr, [])
        try:
            from .baby_blue_draw import (
                baby_blue_mask,
                draw_baby_blue_overlay,
                draw_proximity_error,
                proximity_alerts,
            )

            boxes = getattr(self, cache_attr) or []
            out = draw_baby_blue_overlay(
                decorated,
                min_area=self.blue_min_area,
                show_mask=self.blue_show_mask,
                boxes=boxes,
                depth_mm=depth,
            )
            alerts = []
            if depth is not None and boxes:
                mask = baby_blue_mask(raw)
                if mask.shape[:2] != depth.shape[:2]:
                    mask = cv2.resize(
                        mask,
                        (depth.shape[1], depth.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    raw_for_prox = cv2.resize(
                        raw,
                        (depth.shape[1], depth.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    raw_for_prox = raw
                # Boxes are in color-frame coords; scale to depth if needed
                dh, dw = depth.shape[:2]
                rh, rw = raw.shape[:2]
                if (dh, dw) != (rh, rw) and rh > 0 and rw > 0:
                    sx, sy = dw / float(rw), dh / float(rh)
                    boxes_d = [
                        (
                            int(x * sx),
                            int(y * sy),
                            max(1, int(w * sx)),
                            max(1, int(h * sy)),
                            d,
                        )
                        for x, y, w, h, d in boxes
                    ]
                else:
                    boxes_d = boxes
                # Hidden black equipment mask vs blue joints (3D / depth).
                alerts = proximity_alerts(
                    depth,
                    boxes_d,
                    mask,
                    bgr=raw_for_prox,
                    thresh_mm=self.blue_proximity_mm,
                    skip_base=True,
                )
                if alerts and (dh, dw) != out.shape[:2]:
                    ox, oy = out.shape[1] / float(dw), out.shape[0] / float(dh)
                    alerts = [
                        (
                            int(a[0] * ox),
                            int(a[1] * oy),
                            max(1, int(a[2] * ox)),
                            max(1, int(a[3] * oy)),
                            a[4],
                            a[5] if len(a) > 5 else "",
                        )
                        for a in alerts
                    ]
            setattr(self, alert_attr, bool(alerts))
            if alerts:
                out = draw_proximity_error(
                    out, alerts, thresh_cm=self.blue_proximity_mm / 10.0
                )
            return out
        except Exception as exc:
            self.get_logger().warn(f"blue draw {cam}: {exc}")
            setattr(self, alert_attr, False)
            return decorated

    def _overlay_arm_on(
        self, decorated: np.ndarray, raw: np.ndarray, *, cam: str
    ) -> np.ndarray:
        """Detect on raw frame, then draw boxes onto the decorated overlay."""
        if not self.enable_arm_cv or self._robotics is None:
            return decorated
        seq = self._cam0_seq if cam == "cam0" else self._cam1_seq
        cache_attr = "_arm_boxes0" if cam == "cam0" else "_arm_boxes1"
        if not hasattr(self, "_arm_boxes0"):
            self._arm_boxes0 = None
            self._arm_boxes1 = None
        if seq % self.arm_every_n == 0 or getattr(self, cache_attr) is None:
            try:
                from .arm_overlay_draw import default_arm_weights, get_worker

                wpath = self.arm_weights or default_arm_weights(self._robotics)
                boxes = get_worker(wpath, conf=self.arm_conf).predict(
                    raw, conf=self.arm_conf, cam=cam, refine=True
                )
                setattr(self, cache_attr, boxes)
            except Exception as exc:
                self.get_logger().warn(f"arm detect {cam}: {exc}")
                setattr(self, cache_attr, [])
        out = decorated.copy()
        for xyxy, conf in getattr(self, cache_attr) or []:
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 220, 0), 2)
            cv2.putText(
                out,
                f"arm {conf:.2f}",
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 220, 0),
                2,
                cv2.LINE_AA,
            )
        return out

    def _paint_cam1_cv(self, bgr: np.ndarray) -> np.ndarray:
        self._cam1_seq += 1
        if self.enable_rack_cv:
            if self._cam1_seq % self.classify_every_n == 0 or self._latest1_cv is None:
                self._last_state = self._resolve_state(bgr)
            view = draw_rack_overlay(
                bgr,
                self.display_xywh,
                self._last_state,
                classify_xywh=self.classify_xywh,
            )
            cv2.putText(
                view,
                "cam1 + rack + arm",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            view = _label(bgr, "cam1")
        view = self._overlay_arm_on(view, bgr, cam="cam1")
        return self._overlay_blue_on(view, bgr, cam="cam1")

    def _on_depth0(self, msg: Image) -> None:
        try:
            d = imgmsg_to_numpy(msg)
            if d.ndim == 2:
                self._latest_depth0 = d
        except Exception as exc:
            self.get_logger().warn(f"cam0 depth decode failed: {exc}")

    def _on_depth1(self, msg: Image) -> None:
        try:
            d = imgmsg_to_numpy(msg)
            if d.ndim == 2:
                self._latest_depth1 = d
        except Exception as exc:
            self.get_logger().warn(f"cam1 depth decode failed: {exc}")

    def _on_cam0(self, msg: Image) -> None:
        try:
            img = imgmsg_to_numpy(msg)
            if img.ndim == 3:
                self._latest0 = img
                self._latest0_cv = self._paint_cam0_cv(img)
                self._recv0 += 1
                if self._recv0 == 1:
                    self.get_logger().info(
                        f"cam0 first frame {img.shape[1]}x{img.shape[0]} {msg.encoding}"
                    )
        except Exception as exc:
            self.get_logger().warn(f"cam0 decode/CV failed: {exc}")

    def _on_cam1(self, msg: Image) -> None:
        try:
            img = imgmsg_to_numpy(msg)
            if img.ndim == 3:
                self._latest1 = img
                self._latest1_cv = self._paint_cam1_cv(img)
                self._recv1 += 1
                if self._recv1 == 1:
                    self.get_logger().info(
                        f"cam1 first frame {img.shape[1]}x{img.shape[0]} {msg.encoding}"
                    )
        except Exception as exc:
            self.get_logger().warn(f"cam1 decode/CV failed: {exc}")

    def _status_line(self, recv: int) -> str:
        age = int(time.monotonic() - self._started)
        if recv > 0:
            return f"ok ({recv})"
        return f"no ROS frames {age}s — is publisher up?"

    def _on_timer(self) -> None:
        if self._should_exit:
            return
        age = time.monotonic() - self._started
        if (
            not self._warned_empty
            and age >= 8.0
            and (self._recv0 == 0 or self._recv1 == 0)
        ):
            self._warned_empty = True
            self.get_logger().error(
                f"Still empty after {age:.0f}s (cam0={self._recv0} cam1={self._recv1}). "
                "Publishers must be streaming; close USB dual viewers first."
            )
        try:
            left = self._latest0_cv if self._latest0_cv is not None else self._latest0
            right = self._latest1_cv if self._latest1_cv is not None else self._latest1
            frame = compose_side_by_side(
                left,
                right,
                max_height=self.max_height,
                left_status=self._status_line(self._recv0),
                right_status=self._status_line(self._recv1),
                left_already_labeled=self._latest0_cv is not None,
                right_already_labeled=self._latest1_cv is not None,
            )
            if self._blue_alert0 or self._blue_alert1:
                # Extra full-window red signal so it's impossible to miss
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 18)
                cv2.putText(
                    frame,
                    "ERROR: BLUE WITHIN 10cm",
                    (24, max(40, h // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (0, 0, 255),
                    4,
                    cv2.LINE_AA,
                )
            ok = self._preview.show(frame)
            if not ok and self._preview.user_closed:
                self.get_logger().info("Preview closed — exiting dual view.")
                self._should_exit = True
            elif (
                not ok
                and self._preview.last_error
                and not self._preview_error_logged
            ):
                self._preview_error_logged = True
                self.get_logger().error(
                    f"Preview backend failed: {self._preview.last_error}"
                )
        except Exception as exc:
            self.get_logger().error(
                f"dual view refresh failed: {exc}\n{traceback.format_exc()}"
            )

    def destroy_node(self) -> bool:
        try:
            self._preview.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = DualViewNode()
        # Single-threaded: tkinter preview must run on the main thread.
        while rclpy.ok() and not node._should_exit:
            rclpy.spin_once(node, timeout_sec=0.05)
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
