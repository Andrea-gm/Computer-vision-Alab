"""Standalone CaptureAgent worker: grab N BGR frames from a ROS Image topic."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
ROS_WS = ROOT.parent / "ros2_ws"
for p in (
    ROS_WS / "install" / "Lib" / "site-packages",
    ROS_WS / "src" / "orbbec_dual_sync",
):
    s = str(p)
    if p.is_dir() and s not in sys.path:
        sys.path.insert(0, s)

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from orbbec_dual_sync.image_codec import imgmsg_to_numpy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/orbbec/cam1/color/image_raw")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--result-json", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames: list = []
    errors: list[str] = []

    rclpy.init()
    node = Node("rack_cv_capture_ros_worker")

    def cb(msg: Image) -> None:
        if len(frames) >= args.n:
            return
        try:
            img = imgmsg_to_numpy(msg)
            if img.ndim == 3:
                frames.append(img.copy())
        except Exception as exc:
            errors.append(str(exc))

    node.create_subscription(Image, args.topic, cb, qos_profile_sensor_data)
    t0 = time.time()
    while len(frames) < args.n and time.time() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.02)
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass

    paths: list[str] = []
    for i, frame in enumerate(frames, start=1):
        path = args.out_dir / f"raw_{i:03d}.png"
        cv2.imwrite(str(path), frame)
        paths.append(str(path))

    payload = {
        "ok": bool(paths),
        "agent": "CaptureAgent",
        "paths": paths,
        "source": "ros",
        "n": len(paths),
        "topic": args.topic,
        "errors": errors[:5],
    }
    if not paths:
        payload["error"] = "no frames"
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": payload["ok"], "n": payload["n"]}), flush=True)
    return 0 if paths else 1


if __name__ == "__main__":
    raise SystemExit(main())
