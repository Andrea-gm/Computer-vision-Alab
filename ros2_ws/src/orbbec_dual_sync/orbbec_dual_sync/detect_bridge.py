"""Call existing depthcamera locate helpers on synced RGB-D pairs.

Integration option A: keep detection logic in depthcamera.py; this module only
builds cameras / runs locate_standalone. Soft-peer invent stays off. Live 3D
size-score stays off; reproject is not enabled here.
"""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import queue
import time
import traceback
from typing import Any, Optional

import numpy as np

from .camera_path import ensure_camera_on_path


@dataclass
class CamDetectResult:
    camera_id: int
    has_vial: bool
    box: Optional[tuple[int, int, int, int]]
    distance_cm: Optional[float]
    conf_gate: float
    gate_profile: str = ""
    miss_reason: str = ""
    detection_source: str = ""
    held: bool = False


@dataclass
class PairDetectResult:
    sequence: int
    cam0: CamDetectResult
    cam1: CamDetectResult


@dataclass
class DetectWorkerEvent:
    kind: str
    sequence: int = -1
    result: Optional[PairDetectResult] = None
    detail: str = ""


@dataclass
class PairDetectStats:
    pairs: int = 0
    both_vial: int = 0
    one_vial: int = 0
    neither: int = 0
    agree: int = 0
    disagree: int = 0
    incomplete_cm: int = 0

    def note(
        self,
        a: CamDetectResult,
        b: CamDetectResult,
        *,
        abs_cm: float,
        rel: float,
    ) -> str:
        self.pairs += 1
        if a.has_vial and b.has_vial:
            self.both_vial += 1
            if a.distance_cm is None or b.distance_cm is None:
                self.incomplete_cm += 1
                return "both vial (incomplete cm)"
            delta = abs(float(a.distance_cm) - float(b.distance_cm))
            scale = max(abs(float(a.distance_cm)), abs(float(b.distance_cm)), 1e-6)
            if delta <= abs_cm or (delta / scale) <= rel:
                self.agree += 1
                return f"agree Δ{delta:.1f}cm"
            self.disagree += 1
            return f"disagree Δ{delta:.1f}cm"
        if a.has_vial or b.has_vial:
            self.one_vial += 1
            return "one-cam vial / other miss"
        self.neither += 1
        return "neither"

    def summary_line(self) -> str:
        if self.pairs <= 0:
            return "sync-detect: no pairs yet"
        both_rate = 100.0 * self.both_vial / self.pairs
        agree_den = max(self.agree + self.disagree, 1)
        agree_rate = 100.0 * self.agree / agree_den
        one_rate = 100.0 * self.one_vial / self.pairs
        return (
            f"pairs={self.pairs} both={self.both_vial} ({both_rate:.0f}%) "
            f"one-miss={self.one_vial} ({one_rate:.0f}%) "
            f"agree={self.agree} disagree={self.disagree} "
            f"(agree_rate={agree_rate:.0f}% of both-with-cm)"
        )


class DepthcameraDetectSession:
    """Owns two OrbbecDepthCamera shells (no OpenCV capture) for locate only."""

    def __init__(
        self,
        *,
        model: Optional[Path] = None,
        use_yolo: bool = False,
        weak_camera: int = 0,
        cam0_conf: Optional[float] = None,
        cam1_conf: Optional[float] = None,
        base_conf: Optional[float] = None,
        symmetric_yolo_gates: bool = False,
    ) -> None:
        ensure_camera_on_path()
        import depthcamera as dc

        self.dc = dc
        try:
            from sideways_race.methods.method_i_hard_robust import (
                detect_sideways as winner,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Could not import required sideways winner "
                "sideways_race.methods.method_i_hard_robust. "
                "Check ORBBEC_CAMERA_ROOT and the Camera/sideways_race tree."
            ) from exc
        # Pin the ROS contract directly. Standalone depthcamera experiments may
        # select a different winner without silently changing this launch.
        dc._detect_sideways_race_winner = winner
        # Make this process's contract explicit. locate_standalone remains
        # upright-first and invokes the promoted winner only after YOLO/cap miss.
        dc.set_allow_sideways(True)
        dc.set_sideways_assist(True)
        self.sideways_detector = "I_hard_robust"
        conf = float(base_conf if base_conf is not None else dc.DEFAULT_YOLO_CONF)
        self.gate_map = dc.build_camera_gate_map(
            weak_camera=int(weak_camera),
            base_conf=conf,
            cam0_conf=cam0_conf,
            cam1_conf=cam1_conf,
        )
        # The ROS YOLO path favors stable recall in both independent views.
        # Keep the asymmetric map available for local/non-YOLO experiments,
        # but do not make one synchronized camera pass materially tighter
        # structure/depth gates than its sibling.
        if symmetric_yolo_gates:
            explicit_conf = {0: cam0_conf, 1: cam1_conf}
            self.gate_map = {
                idx: dc.weak_gate_profile(
                    yolo_conf=explicit_conf[idx],
                    base_conf=conf,
                )
                for idx in (0, 1)
            }
        shared_conf = min(g.yolo_conf for g in self.gate_map.values())
        # Torch/Ultralytics can abort the Windows ros2-launch child natively
        # (exit code 3) on first inference when its OpenMP runtime conflicts
        # with the OpenCV/RoboStack runtime. Keep the robust local cap, edge,
        # and promoted sideways paths active without importing Torch by default.
        self.detector = (
            dc.build_shared_detector(model, shared_conf)
            if use_yolo
            else dc.VialDetector(None, shared_conf)
        )
        self.detector_mode = "yolo+local-fallbacks" if use_yolo else "local-fallbacks"
        self.cams: dict[int, Any] = {}
        for idx in (0, 1):
            gates = dc.gate_profile_for(idx, self.gate_map)
            self.cams[idx] = dc.OrbbecDepthCamera(
                model,
                idx,
                gates.yolo_conf,
                detector=self.detector,
                window=f"ros_sync_cam{idx}",
                gates=gates,
                use_3d_model_score=False,
                use_edge_assist=True,
                use_orb_assist=False,
            )
            # No VideoCapture — frames arrive from ROS / local sync.
            self.cams[idx].capture = None

    def locate(self, camera_id: int, color: np.ndarray, depth_mm: np.ndarray) -> CamDetectResult:
        cam = self.cams[int(camera_id)]
        fix = self.dc.locate_standalone(cam, color, depth_mm)
        if fix is None:
            return CamDetectResult(
                camera_id=int(camera_id),
                has_vial=False,
                box=None,
                distance_cm=None,
                conf_gate=float(cam.gates.yolo_conf),
                gate_profile=str(cam.gates.name),
                miss_reason=str(getattr(cam, "miss_reason", "no_vial") or "no_vial"),
                detection_source="",
            )
        dist_cm = None if fix.distance_mm is None else float(fix.distance_mm) / 10.0
        held = bool(getattr(fix, "held", False))
        return CamDetectResult(
            camera_id=int(camera_id),
            has_vial=True,
            box=tuple(int(v) for v in fix.box),
            distance_cm=dist_cm,
            conf_gate=float(cam.gates.yolo_conf),
            gate_profile=str(cam.gates.name),
            detection_source=(
                "temporal-hold"
                if held
                else str(getattr(cam, "last_detect_source", "") or "primary")
            ),
            held=held,
        )


def _detect_worker_main(
    request_queue,
    response_queue,
    *,
    model: Optional[Path],
    use_yolo: bool,
    weak_camera: int,
    symmetric_yolo_gates: bool,
) -> None:
    """Run native OpenCV detection outside the ROS process.

    A Windows access violation/heap abort cannot be caught in Python. Keeping
    locate_standalone in this child lets the ROS subscriber survive and replace
    only the failed detector.
    """
    try:
        session = DepthcameraDetectSession(
            model=model,
            use_yolo=use_yolo,
            weak_camera=weak_camera,
            symmetric_yolo_gates=symmetric_yolo_gates,
        )
        gates = ", ".join(
            f"cam{idx}={gate.name}/conf{gate.yolo_conf:.2f}"
            for idx, gate in sorted(session.gate_map.items())
        )
        response_queue.put(("ready", -1, f"{session.detector_mode}; {gates}"))
    except Exception:
        response_queue.put(("startup_error", -1, traceback.format_exc()))
        return

    while True:
        request = request_queue.get()
        if request is None:
            return
        sequence, color0, depth0, color1, depth1 = request
        try:
            result = PairDetectResult(
                sequence=int(sequence),
                cam0=session.locate(0, color0, depth0),
                cam1=session.locate(1, color1, depth1),
            )
            response_queue.put(("result", int(sequence), result))
        except Exception:
            # Ordinary detector failures are reported per pair. Native faults
            # terminate this worker and are handled by the client watchdog.
            response_queue.put(("error", int(sequence), traceback.format_exc()))


class DetectionWorkerClient:
    """Single-slot, restartable detector worker for a ROS node."""

    def __init__(
        self,
        *,
        model: Optional[Path],
        use_yolo: bool,
        weak_camera: int,
        symmetric_yolo_gates: bool = False,
        detection_timeout_s: float = 20.0,
        yolo_fallback_after_crashes: int = 2,
    ) -> None:
        self._model = model
        self._use_yolo = bool(use_yolo)
        self._active_use_yolo = bool(use_yolo)
        self._weak_camera = int(weak_camera)
        self._symmetric_yolo_gates = bool(symmetric_yolo_gates)
        self._detection_timeout_s = max(1.0, float(detection_timeout_s))
        self._yolo_fallback_after_crashes = max(
            1, int(yolo_fallback_after_crashes)
        )
        self._yolo_crashes = 0
        self._ctx = mp.get_context("spawn")
        self._request_queue = None
        self._response_queue = None
        self._process = None
        self._outstanding = False
        self._submitted_at = None
        self._stopping = False
        self.start()

    @property
    def busy(self) -> bool:
        return self._outstanding

    def start(self) -> None:
        self._request_queue = self._ctx.Queue(maxsize=1)
        self._response_queue = self._ctx.Queue(maxsize=4)
        self._process = self._ctx.Process(
            target=_detect_worker_main,
            kwargs={
                "request_queue": self._request_queue,
                "response_queue": self._response_queue,
                "model": self._model,
                "use_yolo": self._active_use_yolo,
                "weak_camera": self._weak_camera,
                "symmetric_yolo_gates": self._symmetric_yolo_gates,
            },
            name="orbbec_ros_detect_worker",
            daemon=True,
        )
        self._process.start()
        self._outstanding = False
        self._submitted_at = None

    def submit(
        self,
        sequence: int,
        color0: np.ndarray,
        depth0: np.ndarray,
        color1: np.ndarray,
        depth1: np.ndarray,
    ) -> bool:
        if self._outstanding or self._process is None or not self._process.is_alive():
            return False
        request = (
            int(sequence),
            np.ascontiguousarray(color0),
            np.ascontiguousarray(depth0),
            np.ascontiguousarray(color1),
            np.ascontiguousarray(depth1),
        )
        try:
            self._request_queue.put_nowait(request)
        except queue.Full:
            return False
        self._outstanding = True
        self._submitted_at = time.monotonic()
        return True

    def poll(self) -> list[DetectWorkerEvent]:
        events: list[DetectWorkerEvent] = []
        if self._response_queue is not None:
            while True:
                try:
                    kind, sequence, payload = self._response_queue.get_nowait()
                except queue.Empty:
                    break
                if kind == "result":
                    self._outstanding = False
                    self._submitted_at = None
                    events.append(
                        DetectWorkerEvent(
                            kind=kind,
                            sequence=int(sequence),
                            result=payload,
                        )
                    )
                else:
                    if kind in ("error", "startup_error"):
                        self._outstanding = False
                        self._submitted_at = None
                    events.append(
                        DetectWorkerEvent(
                            kind=kind,
                            sequence=int(sequence),
                            detail=str(payload),
                        )
                    )

        timed_out = (
            self._outstanding
            and self._submitted_at is not None
            and (time.monotonic() - self._submitted_at) > self._detection_timeout_s
        )
        if (
            timed_out
            and self._process is not None
            and self._process.is_alive()
        ):
            self._process.terminate()
            self._process.join(timeout=2.0)

        if (
            not self._stopping
            and self._process is not None
            and not self._process.is_alive()
        ):
            exitcode = self._process.exitcode
            detail = (
                f"detector worker timed out after {self._detection_timeout_s:.0f}s"
                if timed_out
                else f"detector worker exited with code {exitcode}"
            )
            if self._active_use_yolo:
                self._yolo_crashes += 1
                if self._yolo_crashes >= self._yolo_fallback_after_crashes:
                    self._active_use_yolo = False
                    detail += (
                        "; disabling YOLO and falling back to local cap/edge/"
                        "sideways detection"
                    )
                else:
                    detail += "; restarting YOLO worker"
            else:
                detail += "; restarting local-fallback worker"
            events.append(
                DetectWorkerEvent(
                    kind="crash",
                    detail=detail,
                )
            )
            self._close_queues()
            self.start()
        return events

    def close(self) -> None:
        self._stopping = True
        process = self._process
        if process is not None and process.is_alive():
            try:
                self._request_queue.put_nowait(None)
            except (queue.Full, ValueError):
                pass
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        self._close_queues()
        self._process = None

    def _close_queues(self) -> None:
        for work_queue in (self._request_queue, self._response_queue):
            if work_queue is not None:
                try:
                    work_queue.cancel_join_thread()
                    work_queue.close()
                except (OSError, ValueError):
                    pass
        self._request_queue = None
        self._response_queue = None
