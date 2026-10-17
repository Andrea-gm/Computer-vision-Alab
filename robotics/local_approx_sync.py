"""NOT ROS. Optional debug helper only — not the supported multi-cam sync path.

This is a plain Python process. It is **not** a ROS2 Node, does not use
``rclpy``, and does **not** replace ``colcon build`` of ``ros2_ws``.

Supported path (real ROS2 Nodes + launch)::

    cd /path/to/Jessica/ros2_ws
    source /opt/ros/$ROS_DISTRO/setup.bash
    ./build.sh
    source install/setup.bash
    export ORBBEC_CAMERA_ROOT=/path/to/Jessica/robotics
    ros2 launch orbbec_dual_sync dual_sync_detect.launch.py

See ``ROS_SYNC.md``.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
ROS_WS = ROOT.parent / "ros2_ws"
PKG = ROS_WS / "src" / "orbbec_dual_sync"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orbbec_dual_sync.approx_time import ApproximateTimeSynchronizer2  # noqa: E402
from orbbec_dual_sync.detect_bridge import (  # noqa: E402
    DepthcameraDetectSession,
    PairDetectStats,
)


def _log(*args, **kwargs) -> None:
    """Print with flush so Windows consoles show progress immediately."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="local_approx_sync.py",
        description=(
            "Local dual-Orbbec ApproximateTime sync + vial detect (no ROS). "
            "Compare agree_rate / one-miss vs unsynced dual-independent."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cam0", type=int, default=0, help="Orbbec index for camera A")
    p.add_argument("--cam1", type=int, default=1, help="Orbbec index for camera B")
    p.add_argument("--slop-ms", type=float, default=50.0, help="Max |t0-t1| in ms")
    p.add_argument("--queue-size", type=int, default=10, help="ApproxTime queue depth")
    p.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Stop after N seconds (0 = run until Ctrl+C / q)",
    )
    p.add_argument("--weak-camera", type=int, default=0, help="Camera index with weaker gates")
    p.add_argument("--model", type=Path, default=None, help="Optional YOLO weights path")
    p.add_argument("--abs-cm", type=float, default=3.0, help="Agree absolute cm threshold")
    p.add_argument("--rel", type=float, default=0.15, help="Agree relative threshold")
    p.add_argument("--no-preview", action="store_true", help="Disable OpenCV preview window")
    p.add_argument(
        "--baseline-unsynced",
        action="store_true",
        help="Ignore ApproximateTime; detect on whatever latest frames are each tick",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional folder for synced color pairs (every 5th pair)",
    )
    return p


def _open_capture(dc, index: int):
    cam = dc.OrbbecDepthCamera(
        None,
        index,
        dc.DEFAULT_YOLO_CONF,
        window=f"local_sync_cap{index}",
    )
    cam.open(require_frame=True)
    return cam


def _print_final(
    stats: PairDetectStats,
    *,
    baseline: bool,
    sync: ApproximateTimeSynchronizer2 | None,
    slop_ms: float,
    frames0: int,
    frames1: int,
    elapsed_s: float,
) -> None:
    _log("")
    _log("--- final ---")
    _log(f"elapsed={elapsed_s:.1f}s frames_cam0={frames0} frames_cam1={frames1}")
    _log(stats.summary_line())
    if not baseline and sync is not None:
        _log(
            f"sync emitted={sync.emitted} dropped_stale={sync.dropped_stale} "
            f"slop={slop_ms:.0f}ms"
        )
    if stats.pairs <= 0:
        _log(
            "WARNING: zero pairs processed. Cameras may not be streaming, "
            "or ApproximateTime never matched within slop. "
            "Retry with --baseline-unsynced or check USB."
        )
    _log(
        "How to read: if synced mode cuts one-miss% and raises agree_rate vs "
        "--baseline-unsynced / --dual-independent, time skew was a contributor. "
        "ROS/sync does not raise YOLO accuracy by itself."
    )


def main() -> int:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    args = build_parser().parse_args()

    mode = (
        "baseline unsynced (latest frames)"
        if args.baseline_unsynced
        else f"ApproximateTime slop={args.slop_ms:.0f}ms queue={args.queue_size}"
    )
    secs = "forever" if args.seconds <= 0 else f"{args.seconds:g}s"
    _log("=" * 64)
    _log("local_approx_sync — NOT a ROS2 process (plain Python debug only)")
    _log("Supported path: colcon build ../ros2_ws → source install/setup.bash")
    _log("  → ros2 launch orbbec_dual_sync dual_sync_detect.launch.py")
    _log(f"cams={args.cam0},{args.cam1}  duration={secs}  preview={'off' if args.no_preview else 'on'}")
    _log(f"mode: {mode}")
    _log("Live 3D size-score: off | soft-peer invent: off | reproject: off")
    _log("=" * 64)

    _log("[1/4] Importing depthcamera (OpenCV / ultralytics)...")
    try:
        import depthcamera as dc
    except Exception as exc:
        _log(f"ERROR: failed to import depthcamera: {exc}")
        traceback.print_exc()
        return 1
    _log("[1/4] depthcamera imported OK")

    _log("[2/4] Probing Orbbec USB indexes (can take several seconds)...")
    try:
        found = dc.list_orbbec_indexes()
    except Exception as exc:
        _log(f"ERROR: camera probe crashed: {exc}")
        traceback.print_exc()
        return 1
    _log(f"[2/4] detected Orbbec indexes: {found if found else 'none'}")
    if args.cam0 not in found or args.cam1 not in found:
        _log(
            f"ERROR: need cams {args.cam0} and {args.cam1}; "
            f"detected: {found if found else 'none'}"
        )
        _log(dc.USB_TIP)
        return 1
    if args.cam0 == args.cam1:
        _log("ERROR: --cam0 and --cam1 must differ")
        return 2

    _log("[3/4] Loading YOLO detect session (may take a while)...")
    try:
        session = DepthcameraDetectSession(
            model=args.model,
            weak_camera=args.weak_camera,
        )
    except Exception as exc:
        _log(f"ERROR: detect session failed: {exc}")
        traceback.print_exc()
        return 1
    _log(dc.format_gate_assignment_line(session.gate_map))
    _log("[3/4] detect session ready")

    _log(f"[4/4] Opening cameras {args.cam0} and {args.cam1} (waiting for first frames)...")
    cap0 = cap1 = None
    try:
        cap0 = _open_capture(dc, args.cam0)
        _log(f"[4/4] camera {args.cam0} OPEN")
        cap1 = _open_capture(dc, args.cam1)
        _log(f"[4/4] camera {args.cam1} OPEN")
    except Exception as exc:
        _log(f"ERROR: camera open failed: {exc}")
        traceback.print_exc()
        if cap0 is not None:
            try:
                cap0.close()
            except Exception:
                pass
        return 1

    sync = ApproximateTimeSynchronizer2(
        slop_s=float(args.slop_ms) / 1000.0,
        queue_size=int(args.queue_size),
    )
    stats = PairDetectStats()
    latest0 = latest1 = None
    frames0 = frames1 = 0
    t0 = time.time()
    last_status = t0
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        _log(f"Saving every 5th pair under: {args.save_dir}")

    _log("Running. Status line every ~1s. Ctrl+C or q in preview to stop.")
    _log("")

    def _process_pair(color0, depth0, color1, depth1, dt_ms: float) -> bool:
        """Run locate on one pair; return False if user asked to quit preview."""
        r0 = session.locate(0, color0, depth0)
        r1 = session.locate(1, color1, depth1)
        label = stats.note(r0, r1, abs_cm=args.abs_cm, rel=args.rel)

        if args.save_dir is not None and stats.pairs % 5 == 0:
            stem = args.save_dir / f"pair_{stats.pairs:06d}"
            cv2.imwrite(str(stem) + "_cam0.jpg", color0)
            cv2.imwrite(str(stem) + "_cam1.jpg", color1)

        if args.no_preview:
            return True
        a = color0.copy()
        b = color1.copy()
        if r0.has_vial and r0.box is not None:
            x, y, w, h = r0.box
            cv2.rectangle(a, (x, y), (x + w, y + h), (0, 220, 0), 2)
        if r1.has_vial and r1.box is not None:
            x, y, w, h = r1.box
            cv2.rectangle(b, (x, y), (x + w, y + h), (0, 220, 0), 2)
        cv2.putText(
            a,
            f"{label} dt={dt_ms:.0f}ms",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        h = min(a.shape[0], b.shape[0], 480)
        a_s = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
        b_s = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
        cv2.imshow("local_approx_sync", np.hstack([a_s, b_s]))
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)

    def _status_line() -> None:
        elapsed = time.time() - t0
        sync_part = (
            f" emitted={sync.emitted} dropped_stale={sync.dropped_stale}"
            if not args.baseline_unsynced
            else ""
        )
        _log(
            f"[status] t={elapsed:5.1f}s frames0={frames0} frames1={frames1} "
            f"{stats.summary_line()}{sync_part}"
        )

    try:
        while True:
            now_loop = time.time()
            if args.seconds > 0 and (now_loop - t0) >= args.seconds:
                _log(f"Reached --seconds {args.seconds:g}; stopping.")
                break

            ready = []
            now = time.time()
            c0, d0 = cap0.read_images()
            if c0 is not None and d0 is not None:
                frames0 += 1
                latest0 = (now, c0, d0)
                if not args.baseline_unsynced:
                    paired = sync.add_a(now, (c0, d0))
                    if paired is not None:
                        ready.append(paired)

            now = time.time()
            c1, d1 = cap1.read_images()
            if c1 is not None and d1 is not None:
                frames1 += 1
                latest1 = (now, c1, d1)
                if not args.baseline_unsynced:
                    paired = sync.add_b(now, (c1, d1))
                    if paired is not None:
                        ready.append(paired)

            if args.baseline_unsynced:
                if latest0 is not None and latest1 is not None:
                    (ts0, color0, depth0) = latest0
                    (ts1, color1, depth1) = latest1
                    if not _process_pair(
                        color0, depth0, color1, depth1, abs(ts0 - ts1) * 1000.0
                    ):
                        _log("Preview quit (q/Esc).")
                        break
            elif ready:
                stop = False
                for pair in ready:
                    color0, depth0 = pair.a.data
                    color1, depth1 = pair.b.data
                    if not _process_pair(
                        color0, depth0, color1, depth1, pair.dt_s * 1000.0
                    ):
                        stop = True
                        break
                if stop:
                    _log("Preview quit (q/Esc).")
                    break
            elif not args.no_preview:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    _log("Preview quit (q/Esc).")
                    break

            if time.time() - last_status >= 1.0:
                _status_line()
                last_status = time.time()
    except KeyboardInterrupt:
        _log("\nStopped (Ctrl+C).")
    except Exception as exc:
        _log(f"ERROR: run loop crashed: {exc}")
        traceback.print_exc()
        _print_final(
            stats,
            baseline=args.baseline_unsynced,
            sync=None if args.baseline_unsynced else sync,
            slop_ms=args.slop_ms,
            frames0=frames0,
            frames1=frames1,
            elapsed_s=time.time() - t0,
        )
        return 1
    finally:
        try:
            if cap0 is not None:
                cap0.close()
            if cap1 is not None:
                cap1.close()
        except Exception:
            pass
        try:
            cv2.destroyWindow("local_approx_sync")
        except cv2.error:
            pass

    _print_final(
        stats,
        baseline=args.baseline_unsynced,
        sync=None if args.baseline_unsynced else sync,
        slop_ms=args.slop_ms,
        frames0=frames0,
        frames1=frames1,
        elapsed_s=time.time() - t0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

