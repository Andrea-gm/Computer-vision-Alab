"""Grab one cam1 frame, find the black jar rack, update rack.json ROIs."""

from __future__ import annotations

import json
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
REF_CLOSED = STATE / "template_roi.png"
REF_OPEN = STATE / "template_open_roi.png"
USER_CROP = Path(
    r"C:\Users\Ceder-ALAB\.cursor\projects\c-Users-Ceder-ALAB-Documents-GitHub-alab-one"
    r"\assets\c__Users_Ceder-ALAB_AppData_Roaming_Cursor_User_workspaceStorage_"
    r"3d6bbacd5f40cbd7d1aac7c702b53646_images_image-59168f68-7ea9-493c-aa61-8d6bd0c791a1.png"
)


class Grab(Node):
    def __init__(self) -> None:
        super().__init__("rack_relock")
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


def match_template(
    frame: np.ndarray, ref: np.ndarray, *, scales: np.ndarray
) -> tuple[float, int, int, int, int, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    rh, rw = rgray.shape[:2]
    best = (-1.0, 0, 0, rw, rh, 1.0)
    for scale in scales:
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


def find_black_rack(frame: np.ndarray) -> tuple[float, int, int, int, int]:
    """Score dark rectangular blobs by white-lid content (rack signature)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dark = cv2.inRange(
        hsv, np.array([0, 0, 0], np.uint8), np.array([180, 90, 70], np.uint8)
    )
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    white = cv2.inRange(
        hsv, np.array([0, 0, 170], np.uint8), np.array([180, 60, 255], np.uint8)
    )
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fh, fw = frame.shape[:2]
    best = (-1.0, 0, 0, 0, 0)
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < 2500:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 70 or h < 90:
            continue
        aspect = h / max(1.0, float(w))
        if aspect < 0.85 or aspect > 2.2:
            continue
        # Prefer mid-frame (avoid huge floor shadows)
        cy = y + 0.5 * h
        if cy < 0.12 * fh or cy > 0.88 * fh:
            continue
        roi_w = white[y : y + h, x : x + w]
        white_frac = float(cv2.countNonZero(roi_w)) / max(1.0, float(w * h))
        # Rack has many white lids; arm/table shadows do not
        if white_frac < 0.04:
            continue
        fill = area / max(1.0, float(w * h))
        score = 2.5 * white_frac + 0.4 * fill + 0.15 * min(aspect, 1.4)
        if score > best[0]:
            best = (score, int(x), int(y), int(w), int(h))
    return best


def main() -> None:
    rclpy.init()
    node = Grab()
    t0 = time.time()
    while node.frame is None and time.time() - t0 < 15:
        rclpy.spin_once(node, timeout_sec=0.2)
    frame = node.frame
    node.destroy_node()
    rclpy.shutdown()
    if frame is None:
        raise SystemExit("no cam1 frame in 15s")

    candidates: list[tuple[str, float, int, int, int, int, float]] = []
    scales = np.linspace(0.75, 1.35, 25)
    for name, path in (
        ("user_crop", USER_CROP),
        ("template_open", REF_OPEN),
        ("template_closed", REF_CLOSED),
    ):
        if not path.is_file():
            continue
        ref = cv2.imread(str(path))
        if ref is None:
            continue
        score, x, y, w, h, sc = match_template(frame, ref, scales=scales)
        candidates.append((name, score, x, y, w, h, sc))
        print(f"tmpl {name}: score={score:.3f} xywh=({x},{y},{w},{h}) scale={sc:.3f}")

    bscore, bx, by, bw, bh = find_black_rack(frame)
    print(f"black_rack: score={bscore:.3f} xywh=({bx},{by},{bw},{bh})")
    if bscore > 0:
        candidates.append(("black_rack", bscore, bx, by, bw, bh, 1.0))

    if not candidates:
        raise SystemExit("no rack candidates")

    # Prefer high template scores; if all templates weak, use black_rack
    tmpl_best = max(
        (c for c in candidates if c[0] != "black_rack"),
        key=lambda c: c[1],
        default=None,
    )
    black = next((c for c in candidates if c[0] == "black_rack"), None)
    if tmpl_best and tmpl_best[1] >= 0.62:
        chosen = tmpl_best
    elif black and black[1] > 0:
        # If template agrees roughly with black blob, use black geometry (more stable)
        if tmpl_best and tmpl_best[1] >= 0.45:
            tx, ty, tw, th = tmpl_best[2:6]
            ox = max(tx, black[2])
            oy = max(ty, black[3])
            ox2 = min(tx + tw, black[2] + black[4])
            oy2 = min(ty + th, black[3] + black[5])
            inter = max(0, ox2 - ox) * max(0, oy2 - oy)
            union = tw * th + black[4] * black[5] - inter
            iou = inter / max(1.0, float(union))
            print(f"iou(tmpl,black)={iou:.3f}")
            if iou >= 0.25:
                chosen = black
            else:
                chosen = black  # templates confused by arm — trust black+white lids
        else:
            chosen = black
    elif tmpl_best:
        chosen = tmpl_best
    else:
        raise SystemExit("no usable candidate")

    name, score, x, y, w, h, sc = chosen
    print("CHOSEN", name, score, x, y, w, h)

    # Tighten classify ROI slightly inside display-ish black box for OPEN/CLOSED
    # Keep similar aspect to historical 137x152 (~0.90)
    classify_w = int(round(w * 0.72))
    classify_h = int(round(h * 0.78))
    classify_x = int(x + (w - classify_w) // 2)
    classify_y = int(y + int(0.08 * h))
    # Clamp
    fh, fw = frame.shape[:2]
    classify_x = max(0, min(fw - classify_w, classify_x))
    classify_y = max(0, min(fh - classify_h, classify_y))

    data = json.loads(RACK.read_text(encoding="utf-8"))
    old_x, old_y, old_w, old_h = (
        int(data["x"]),
        int(data["y"]),
        int(data["w"]),
        int(data["h"]),
    )
    sx = classify_w / max(1, old_w)
    sy = classify_h / max(1, old_h)

    # display = black-rack box with a little silver-frame pad
    pad = 12
    ndx = max(0, x - pad)
    ndy = max(0, y - pad)
    ndx2 = min(fw, x + w + pad)
    ndy2 = min(fh, y + h + pad)
    ndw, ndh = ndx2 - ndx, ndy2 - ndy

    holes = data.get("holes") or []
    new_holes = []
    for hole in holes:
        new_holes.append(
            {
                "cx": int(round(float(hole["cx"]) * sx)),
                "cy": int(round(float(hole["cy"]) * sy)),
                "rx": max(3, int(round(float(hole.get("rx", 8)) * sx))),
                "ry": max(3, int(round(float(hole.get("ry", 5)) * sy))),
            }
        )

    data["x"] = int(classify_x)
    data["y"] = int(classify_y)
    data["w"] = int(classify_w)
    data["h"] = int(classify_h)
    data["roi"] = {
        "x": int(classify_x),
        "y": int(classify_y),
        "w": int(classify_w),
        "h": int(classify_h),
        "comment": "Classify ROI re-locked after cam1 move.",
    }
    data["display_roi"] = {
        "x": int(ndx),
        "y": int(ndy),
        "w": int(ndw),
        "h": int(ndh),
        "comment": "Full black rack (+pad) on cam1 after camera move.",
    }
    data["holes"] = new_holes
    if new_holes:
        data["slot_rx"] = new_holes[0]["rx"]
        data["slot_ry"] = new_holes[0]["ry"]
    data["image_width"] = int(fw)
    data["image_height"] = int(fh)
    data["frame_size"] = [int(fw), int(fh)]
    data["note"] = (
        f"Re-locked after cam1 move via {name} (score={score:.3f}). "
        f"Old ROI was ({old_x},{old_y},{old_w},{old_h})."
    )
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    RACK.write_text(json.dumps(data, indent=2), encoding="utf-8")

    vis = frame.copy()
    cv2.rectangle(vis, (ndx, ndy), (ndx + ndw, ndy + ndh), (0, 255, 255), 2)
    cv2.rectangle(
        vis,
        (classify_x, classify_y),
        (classify_x + classify_w, classify_y + classify_h),
        (0, 255, 0),
        2,
    )
    cv2.putText(
        vis,
        f"{name} {score:.2f}",
        (classify_x, max(20, classify_y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )
    out = STATE / "_relock_preview.png"
    cv2.imwrite(str(out), vis)
    print(
        "classify",
        classify_x,
        classify_y,
        classify_w,
        classify_h,
        "display",
        ndx,
        ndy,
        ndw,
        ndh,
    )
    print("wrote", RACK)
    print("preview", out)


if __name__ == "__main__":
    main()
