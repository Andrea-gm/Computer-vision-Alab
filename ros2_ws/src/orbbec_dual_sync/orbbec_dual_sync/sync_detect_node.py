"""Sync dual Orbbec topics with ApproximateTime, then run locate_standalone.

Option A: ROS sync + import depthcamera helpers. No soft-peer invent.
Reports agree / one-miss rates so you can compare against unsynced dual-independent.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import cv2
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .approx_time import stamp_msg_to_seconds
from .detect_bridge import DetectionWorkerClient, PairDetectResult, PairDetectStats
from .image_codec import imgmsg_to_numpy
from .preview_gui import PreviewWindow
from .preview_labels import draw_vial_box


class SyncDetectNode(Node):
    def __init__(self) -> None:
        super().__init__("orbbec_sync_detect")
        self.declare_parameter("slop_s", 0.05)
        self.declare_parameter("queue_size", 10)
        self.declare_parameter("cam0_color_topic", "/orbbec/cam0/color/image_raw")
        self.declare_parameter("cam1_color_topic", "/orbbec/cam1/color/image_raw")
        self.declare_parameter("cam0_depth_topic", "/orbbec/cam0/depth/image_raw")
        self.declare_parameter("cam1_depth_topic", "/orbbec/cam1/depth/image_raw")
        self.declare_parameter("show_preview", True)
        self.declare_parameter("save_synced_dir", "")
        self.declare_parameter("corroborate_abs_cm", 3.0)
        self.declare_parameter("corroborate_rel", 0.15)
        self.declare_parameter("weak_camera", 0)
        self.declare_parameter("model", "")
        self.declare_parameter("use_yolo", False)
        self.declare_parameter("enable_detect", False)
        self.declare_parameter("symmetric_yolo_gates", True)
        self.declare_parameter("stats_topic", "/orbbec/sync_detect/stats")

        slop = float(self.get_parameter("slop_s").value)
        qsize = int(self.get_parameter("queue_size").value)
        self.show_preview = bool(self.get_parameter("show_preview").value)
        self.save_dir = str(self.get_parameter("save_synced_dir").value).strip()
        self.abs_cm = float(self.get_parameter("corroborate_abs_cm").value)
        self.rel = float(self.get_parameter("corroborate_rel").value)
        self.enable_detect = bool(self.get_parameter("enable_detect").value)
        self._first_pair_decoded = False
        self._first_detection = True
        self._sequence = 0
        self._pending = None
        self._busy_drops = 0
        self._drops_since_result = 0
        self._first_pair_at = None
        self._last_result_at = None
        self._last_box_at = None
        self._consecutive_no_box = 0
        self._last_health_alert: dict[str, float] = {}
        weak = int(self.get_parameter("weak_camera").value)
        model_s = str(self.get_parameter("model").value).strip()
        use_yolo = bool(self.get_parameter("use_yolo").value)
        symmetric_yolo_gates = bool(
            self.get_parameter("symmetric_yolo_gates").value
        )
        model = Path(model_s) if model_s else None

        c0 = str(self.get_parameter("cam0_color_topic").value)
        c1 = str(self.get_parameter("cam1_color_topic").value)
        d0 = str(self.get_parameter("cam0_depth_topic").value)
        d1 = str(self.get_parameter("cam1_depth_topic").value)

        self.stats = PairDetectStats()
        self.stats_pub = self.create_publisher(
            String, str(self.get_parameter("stats_topic").value), 10
        )
        self._detect_worker = None
        self._worker_timer = None
        self._health_timer = None
        if not self.enable_detect:
            self.get_logger().warn(
                "VIAL DETECT DISABLED (enable_detect:=false). "
                "Cameras publish / preview only — no YOLO, no locate, no vial boxes. "
                "Use rack_lid_state.py focus for open/closed."
            )
            print(
                "[orbbec_sync_detect] VIAL DETECT DISABLED — cameras only. "
                "Rack: python rack_lid_state.py focus",
                flush=True,
            )
        else:
            self._detect_worker = DetectionWorkerClient(
                model=model,
                use_yolo=use_yolo,
                weak_camera=weak,
                symmetric_yolo_gates=symmetric_yolo_gates,
            )
            strong = 1 if weak == 0 else 0
            if symmetric_yolo_gates:
                self.get_logger().info(
                    "Stability gates: both cameras use the permissive "
                    "independent-camera profile (no peer assist/invent)."
                )
            else:
                self.get_logger().info(
                    f"Asymmetric gates: weak=cam{weak} (permissive), "
                    f"strong=cam{strong} (tighter). Override: weak_camera"
                )
            detector_mode = "yolo+local-fallbacks" if use_yolo else "local-fallbacks"
            self.get_logger().info(
                f"Detection=depthcamera.locate_standalone "
                f"({detector_mode}, isolated worker); sideways fallback="
                "I_hard_robust; upright-first; "
                "live 3D size-score/reproject off; soft-peer invent off; "
                f"ApproximateTime slop={slop*1000:.0f}ms"
            )
            self._worker_timer = self.create_timer(0.05, self._poll_detect_worker)
            self._health_timer = self.create_timer(2.0, self._check_detector_health)

        self._preview = (
            PreviewWindow("orbbec_sync_detect") if self.show_preview else None
        )
        if self._preview is not None:
            self.get_logger().info(
                "Preview ON (launch-safe tkinter; set ORBBEC_PREVIEW_CV2=1 "
                "only for a known-good OpenCV HighGUI build). "
                "Press Q or close the window to hide preview; Ctrl+C stops launch."
            )
        else:
            self.get_logger().info("Preview OFF (show_preview:=false).")

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._sub_c0 = Subscriber(self, Image, c0, qos_profile=image_qos)
        self._sub_d0 = Subscriber(self, Image, d0, qos_profile=image_qos)
        self._sub_c1 = Subscriber(self, Image, c1, qos_profile=image_qos)
        self._sub_d1 = Subscriber(self, Image, d1, qos_profile=image_qos)
        self._sync = ApproximateTimeSynchronizer(
            [self._sub_c0, self._sub_d0, self._sub_c1, self._sub_d1],
            qsize,
            slop,
        )
        self._sync.registerCallback(self._on_synced)
        self.get_logger().info(
            "Waiting for ApproximateTime-synced RGB-D pairs "
            f"(cam0={c0}, cam1={c1})..."
        )

        if self.save_dir:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def _on_synced(
        self, color0: Image, depth0: Image, color1: Image, depth1: Image
    ) -> None:
        try:
            self._handle_synced(color0, depth0, color1, depth1)
        except Exception as exc:
            self.get_logger().error(
                f"synced callback failed (node stays up): {exc}\n"
                f"{traceback.format_exc()}"
            )

    def _handle_synced(
        self, color0: Image, depth0: Image, color1: Image, depth1: Image
    ) -> None:
        try:
            c0 = imgmsg_to_numpy(color0)
            d0 = imgmsg_to_numpy(depth0)
            c1 = imgmsg_to_numpy(color1)
            d1 = imgmsg_to_numpy(depth1)
        except Exception as exc:
            self.get_logger().warn(f"decode failed: {exc}")
            return

        if not self._first_pair_decoded:
            self._first_pair_at = time.monotonic()
            print(
                "[orbbec_sync_detect] first synchronized RGB-D pair decoded",
                flush=True,
            )
            self._first_pair_decoded = True
        if not self.enable_detect or self._detect_worker is None:
            # Cameras-only: optional raw preview, no locate / boxes / YOLO.
            if self._preview is not None:
                view = self._compose_cameras_only(c0, c1)
                if not self._preview.show(view):
                    self._preview = None
                    self.get_logger().info(
                        "Preview closed; publishers still running. Ctrl+C to stop."
                    )
            return
        sequence = self._sequence
        self._sequence += 1
        if not self._detect_worker.submit(sequence, c0, d0, c1, d1):
            self._busy_drops += 1
            self._drops_since_result += 1
            if self._drops_since_result == 30 or self._drops_since_result % 100 == 0:
                self._loud(
                    "warn",
                    "DETECTION DEGRADED: detector is busy; "
                    f"{self._drops_since_result} synchronized pairs skipped "
                    "since the last result. Preview may be stale.",
                )
            return
        self._pending = (sequence, c0, c1, color0.header.stamp, color1.header.stamp)

    def _poll_detect_worker(self) -> None:
        for event in self._detect_worker.poll():
            if event.kind == "ready":
                print(
                    f"[orbbec_sync_detect] detection worker ready: {event.detail}",
                    flush=True,
                )
                self.get_logger().info(f"Detection worker ready ({event.detail})")
                continue
            if event.kind in ("error", "startup_error"):
                self._pending = None
                self._loud(
                    "error",
                    f"DETECTOR WORKER {event.kind.upper()}: detection stopped "
                    f"for this pair; node stays up.\n{event.detail}",
                )
                continue
            if event.kind == "crash":
                self._pending = None
                self._loud(
                    "error",
                    f"DETECTOR WORKER DIED: {event.detail}. "
                    "ROS sync stays up; preview is stale until a new result arrives.",
                )
                continue
            if event.kind != "result" or event.result is None:
                continue
            pending = self._pending
            self._pending = None
            if pending is None or pending[0] != event.sequence:
                self.get_logger().warn(
                    f"Ignoring stale detection result sequence={event.sequence}"
                )
                continue
            _, c0, c1, stamp0, stamp1 = pending
            self._handle_detection_result(event.result, c0, c1, stamp0, stamp1)

    def _handle_detection_result(
        self,
        result: PairDetectResult,
        c0,
        c1,
        stamp0,
        stamp1,
    ) -> None:
        r0 = result.cam0
        r1 = result.cam1
        now = time.monotonic()
        previous_no_box = self._consecutive_no_box
        self._last_result_at = now
        self._drops_since_result = 0
        if r0.has_vial or r1.has_vial:
            self._last_box_at = now
            self._consecutive_no_box = 0
            if previous_no_box >= 15:
                self._loud(
                    "info",
                    "DETECTION RECOVERED: a camera is finding the vial again.",
                )
        else:
            self._consecutive_no_box += 1
            if (
                self._consecutive_no_box == 15
                or self._consecutive_no_box % 60 == 0
            ):
                self._loud(
                    "warn",
                    "DETECTION DEGRADED: both cameras returned no vial for "
                    f"{self._consecutive_no_box} consecutive processed pairs.",
                )
        if self._first_detection:
            print("[orbbec_sync_detect] cam0 detection completed", flush=True)
            print("[orbbec_sync_detect] cam1 detection completed", flush=True)
            self._first_detection = False
        label = self.stats.note(r0, r1, abs_cm=self.abs_cm, rel=self.rel)
        dt_ms = (
            abs(
                stamp_msg_to_seconds(stamp0)
                - stamp_msg_to_seconds(stamp1)
            )
            * 1000.0
        )

        line = (
            f"{self.stats.summary_line()} | last={label} dt={dt_ms:.1f}ms "
            f"cam0={self._result_text(r0)} "
            f"cam1={self._result_text(r1)} busy_drops={self._busy_drops}"
        )
        if self.stats.pairs % 15 == 0:
            self.get_logger().info(line)
        msg = String()
        msg.data = line
        self.stats_pub.publish(msg)

        if self._preview is not None and self._preview.enabled:
            ok = self._preview.show(self._compose_preview(c0, c1, r0, r1, label, dt_ms))
            if not ok:
                if self._preview.user_closed:
                    self.get_logger().info(
                        "Preview closed; sync_detect still running. "
                        "Ctrl+C to stop launch."
                    )
                else:
                    self.get_logger().error(
                        "Preview could not open; sync_detect remains alive. "
                        f"Detail: {self._preview.last_error or 'unknown UI error'}"
                    )
                self._preview = None
        if self.save_dir and self.stats.pairs % 5 == 0:
            stem = Path(self.save_dir) / f"pair_{self.stats.pairs:06d}"
            cv2.imwrite(str(stem) + "_cam0.jpg", c0)
            cv2.imwrite(str(stem) + "_cam1.jpg", c1)

    def _check_detector_health(self) -> None:
        if self._first_pair_at is None:
            return
        now = time.monotonic()
        result_anchor = self._last_result_at or self._first_pair_at
        stale_s = now - result_anchor
        if stale_s >= 12.0 and self._alert_due("stale", now, 10.0):
            self._loud(
                "error",
                "DETECTION STALLED: no completed detector result for "
                f"{stale_s:.0f}s; preview is stale. The worker watchdog will "
                "restart it and disable YOLO after repeated failures.",
            )
        box_anchor = self._last_box_at or self._first_pair_at
        no_box_s = now - box_anchor
        if (
            self._last_result_at is not None
            and self._consecutive_no_box > 0
            and no_box_s >= 10.0
            and self._alert_due("no_box", now, 15.0)
        ):
            self._loud(
                "warn",
                "DETECTION DEGRADED: neither camera has produced a vial box "
                f"for {no_box_s:.0f}s ({self._consecutive_no_box} processed pairs).",
            )

    def _alert_due(self, key: str, now: float, interval_s: float) -> bool:
        previous = self._last_health_alert.get(key)
        if previous is not None and now - previous < interval_s:
            return False
        self._last_health_alert[key] = now
        return True

    def _loud(self, level: str, message: str) -> None:
        print(f"[orbbec_sync_detect] !!! {message}", flush=True)
        getattr(self.get_logger(), level)(message)

    @staticmethod
    def _result_text(result) -> str:
        gate = f"{result.gate_profile or 'gate'}@{result.conf_gate:.2f}"
        if not result.has_vial:
            return f"{result.miss_reason}[{gate}]"
        distance = (
            f" {result.distance_cm:.1f}cm"
            if result.distance_cm is not None
            else ""
        )
        return f"vial[{result.detection_source},{gate}]{distance}"

    @staticmethod
    def _preview_status(result) -> str:
        """On-screen status without a vial class caption."""
        if not result.has_vial:
            return "no box"
        if result.distance_cm is not None:
            return f"{result.distance_cm:.1f}cm"
        return "box"

    def _compose_cameras_only(self, c0, c1):
        """Side-by-side raw cams — no vial boxes."""
        a = c0.copy()
        b = c1.copy()
        cv2.putText(
            a,
            "cam0 (detect OFF)",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            b,
            "cam1 (detect OFF)",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        ha, wa = a.shape[:2]
        hb, wb = b.shape[:2]
        h = min(ha, hb, 480)
        a_s = cv2.resize(a, (int(wa * (h / float(ha))), h))
        b_s = cv2.resize(b, (int(wb * (h / float(hb))), h))
        return cv2.hconcat([a_s, b_s])

    def _compose_preview(self, c0, c1, r0, r1, label: str, dt_ms: float):
        del label  # console stats only; the preview does not print vial captions
        a = c0.copy()
        b = c1.copy()
        if r0.has_vial and r0.box is not None:
            draw_vial_box(
                a,
                r0.box,
                detection_source=r0.detection_source,
                distance_cm=r0.distance_cm,
            )
        if r1.has_vial and r1.box is not None:
            draw_vial_box(
                b,
                r1.box,
                detection_source=r1.detection_source,
                distance_cm=r1.distance_cm,
            )
        cv2.putText(
            a,
            f"cam0 {self._preview_status(r0)} dt={dt_ms:.0f}ms",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            b,
            f"cam1 {self._preview_status(r1)} synced",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        ha, wa = a.shape[:2]
        hb, wb = b.shape[:2]
        h = min(ha, hb, 480)
        scale_a = h / float(ha)
        scale_b = h / float(hb)
        a_s = cv2.resize(a, (int(wa * scale_a), h))
        b_s = cv2.resize(b, (int(wb * scale_b), h))
        return cv2.hconcat([a_s, b_s])

    def destroy_node(self) -> bool:
        try:
            if self._detect_worker is not None:
                self._detect_worker.close()
        except Exception:
            pass
        if self._preview is not None:
            try:
                self._preview.close()
            except Exception:
                pass
            self._preview = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = SyncDetectNode()
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
