"""Match a user-provided rack crop into live cam1 and update rack.json."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

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

STATE = Path(__file__).resolve().parent
RACK = STATE / "rack.json"
USER_CROP = Path(
    r"C:\Users\Ceder-ALAB\.cursor\projects\c-Users-Ceder-ALAB-Documents-GitHub-alab-one"
    r"\assets\c__Users_Ceder-ALAB_AppData_Roaming_Cursor_User_workspaceStorage_"
    r"3d6bbacd5f40cbd7d1aac7c702b53646_images_image-d828b964-350b-4966-bbe0-62087ec6ccd2.png"
)


class Grab(Node):
    def __init__(self) -> None:
        super().__init__("rack_relock_user")
        self.frame = None
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            Image, "/orbbec/cam1/color/image_raw", self._cb, qos
        )

    def _cb(self, msg: Image) -> None:
        if msg.encoding not in ("bgr8", "rgb8"):
            return
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )
        if msg.encoding == "rgb8":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        self.frame = arr.copy()


def strip_overlay_tint(bgr: np.ndarray) -> np.ndarray:
    """Drop bright cyan overlay pixels from the dual_view crop if present."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # cyan / baby-blue overlay text & boxes
    cyan = cv2.inRange(
        hsv, np.array([80, 80, 120], np.uint8), np.array([110, 255, 255], np.uint8)
    )
    if cv2.countNonZero(cyan) < 20:
        return bgr
    out = bgr.copy()
    # inpaint cyan overlay so match focuses on the rack
    out = cv2.inpaint(out, cyan, 3, cv2.INPAINT_TELEA)
    return out


def match_template(
    frame: np.ndarray, ref: np.ndarray
) -> tuple[float, int, int, int, int, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    # slight blur helps with compression / overlay residue
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    rgray = cv2.GaussianBlur(rgray, (3, 3), 0)
    rh, rw = rgray.shape[:2]
    best = (-1.0, 0, 0, rw, rh, 1.0)
    for scale in np.linspace(0.85, 1.25, 21):
        tw = int(round(rw * float(scale)))
        th = int(round(rh * float(scale)))
        if tw < 40 or th < 40 or tw >= gray.shape[1] or th >= gray.shape[0]:
            continue
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        templ = cv2.resize(rgray, (tw, th), interpolation=interp)
        score_map = cv2.matchTemplate(gray, templ, cv2.TM_CCOEFF_NORMED)
        _mn, mx, _ml, ml = cv2.minMaxLoc(score_map)
        if float(mx) > best[0]:
            best = (float(mx), int(ml[0]), int(ml[1]), tw, th, float(scale))
    return best


def main() -> None:
    if not USER_CROP.is_file():
        raise SystemExit(f"missing user crop: {USER_CROP}")

    rclpy.init()
    node = Grab()
    t0 = time.time()
    while node.frame is None and time.time() - t0 < 15:
        rclpy.spin_once(node, timeout_sec=0.2)
    frame = node.frame
    node.destroy_node()
    rclpy.shutdown()
    if frame is None:
        raise SystemExit("no cam1 frame")

    crop = cv2.imread(str(USER_CROP))
    if crop is None:
        raise SystemExit("failed to read user crop")
    crop = strip_overlay_tint(crop)

    score, x, y, w, h, sc = match_template(frame, crop)
    print(
        "frame",
        frame.shape,
        "crop",
        crop.shape,
        "match",
        {"score": round(score, 3), "x": x, "y": y, "w": w, "h": h, "scale": round(sc, 3)},
    )
    if score < 0.40:
        raise SystemExit(f"match too weak ({score:.3f})")

    fh, fw = frame.shape[:2]
    # display: expand to include silver frame around the black rack face
    pad_l, pad_t, pad_r, pad_b = 28, 18, 32, 55
    ndx = max(0, x - pad_l)
    ndy = max(0, y - pad_t)
    ndx2 = min(fw, x + w + pad_r)
    ndy2 = min(fh, y + h + pad_b)
    ndw, ndh = ndx2 - ndx, ndy2 - ndy

    data = json.loads(RACK.read_text(encoding="utf-8"))
    old = (int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"]))
    sx = w / max(1, old[2])
    sy = h / max(1, old[3])

    holes = []
    for hole in data.get("holes") or []:
        holes.append(
            {
                "cx": int(round(float(hole["cx"]) * sx)),
                "cy": int(round(float(hole["cy"]) * sy)),
                "rx": max(3, int(round(float(hole.get("rx", 8)) * sx))),
                "ry": max(3, int(round(float(hole.get("ry", 5)) * sy))),
            }
        )

    data["x"] = int(x)
    data["y"] = int(y)
    data["w"] = int(w)
    data["h"] = int(h)
    data["roi"] = {
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
        "comment": "Classify ROI from user closed-rack crop after camera move.",
    }
    data["display_roi"] = {
        "x": int(ndx),
        "y": int(ndy),
        "w": int(ndw),
        "h": int(ndh),
        "comment": "Silver-frame display box around new closed-rack place.",
    }
    data["holes"] = holes
    if holes:
        data["slot_rx"] = holes[0]["rx"]
        data["slot_ry"] = holes[0]["ry"]
    data["image_width"] = int(fw)
    data["image_height"] = int(fh)
    data["frame_size"] = [int(fw), int(fh)]
    data["state"] = "CLOSED"
    data["note"] = (
        f"User closed-rack crop re-lock. match={score:.3f} scale={sc:.3f}. "
        f"Old ROI was {old}."
    )
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    data["reference"] = "template_roi.png"
    data["reference_closed"] = "reference_closed.png"
    RACK.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Refresh closed templates from the matched live crop (clean, no overlay)
    live_crop = frame[y : y + h, x : x + w].copy()
    cv2.imwrite(str(STATE / "template_roi.png"), live_crop)
    cv2.imwrite(str(STATE / "reference_closed.png"), live_crop)
    # keep a copy of the user seed
    shutil.copy2(USER_CROP, STATE / "user_closed_seed.png")

    vis = frame.copy()
    cv2.rectangle(vis, (ndx, ndy), (ndx + ndw, ndy + ndh), (0, 255, 255), 2)
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)
    cv2.putText(
        vis,
        f"CLOSED {score:.2f}",
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 200, 0),
        2,
    )
    out = STATE / "_relock_preview.png"
    cv2.imwrite(str(out), vis)
    print("classify", x, y, w, h, "display", ndx, ndy, ndw, ndh)
    print("wrote", RACK)
    print("templates updated: template_roi.png, reference_closed.png")
    print("preview", out)


if __name__ == "__main__":
    main()
