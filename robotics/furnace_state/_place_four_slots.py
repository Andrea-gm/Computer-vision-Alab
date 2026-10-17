"""Pin 4 fixed furnace slots on the furnace ROW only (same y, even x spacing)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import furnace_agents as fa  # noqa: E402

STATE = ROOT / "furnace_state"
OUT = STATE / "live_for_slots.png"
PREVIEW = STATE / "preview_fixed_slots.png"

# Furnace doors sit in this vertical band on cam0 640x480.
Y_LO, Y_HI = 95, 185
MIN_DX = 50


def peaks_in_band(gray, tmpl_gray, *, n=8, thr=0.35):
    th, tw = tmpl_gray.shape[:2]
    res = cv2.matchTemplate(gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    peaks = []
    work = res.copy()
    for _ in range(24):
        _mn, maxv, _ml, maxl = cv2.minMaxLoc(work)
        if maxv < thr:
            break
        x, y = int(maxl[0]), int(maxl[1])
        y0, y1 = max(0, y - th // 2), min(work.shape[0], y + th // 2)
        x0, x1 = max(0, x - tw // 2), min(work.shape[1], x + tw // 2)
        work[y0:y1, x0:x1] = 0
        if not (Y_LO <= y <= Y_HI):
            continue
        if any(abs(x - px) < MIN_DX for px, *_ in peaks):
            continue
        peaks.append((x, y, tw, th, float(maxv)))
        if len(peaks) >= n:
            break
    peaks.sort(key=lambda p: p[0])
    return peaks


def even_four(peaks, tw, th, frame_w: int):
    """Build exactly 4 slots on one horizontal line from in-band peaks."""
    if not peaks:
        return []
    ys = [p[1] for p in peaks]
    y = int(round(float(np.median(ys))))
    xs = sorted(p[0] for p in peaks)
    if len(xs) >= 2:
        gaps = np.diff(xs)
        # Prefer typical neighbor spacing (~60-90px for these doors)
        good = [int(g) for g in gaps if 45 <= g <= 110]
        dx = int(round(float(np.median(good if good else gaps))))
    else:
        dx = 70
    x0 = int(xs[0])
    # If we somehow started mid-row, shift left so 4 fit in frame
    while x0 + 3 * dx + tw > frame_w - 4 and x0 - dx >= 0:
        x0 -= dx
    x0 = max(0, x0)
    return [(x0 + i * dx, y, tw, th, 0.99) for i in range(4)]


def main() -> int:
    frames = fa.grab_frames_ros(
        4, topic="/orbbec/cam0/color/image_raw", timeout_s=25.0, interval_s=0.35
    )
    if not frames:
        print("ROS failed, trying USB")
        one, _ = fa.grab_one_frame(
            source="usb", camera=0, topic="/orbbec/cam0/color/image_raw"
        )
        frames = [one] if one is not None else []
    if not frames and OUT.is_file():
        print("using cached", OUT)
        frames = [cv2.imread(str(OUT))]
    if not frames or frames[-1] is None:
        print("ERROR: no live frame")
        return 1
    frame = frames[-1]
    cv2.imwrite(str(OUT), frame)
    frame_w = int(frame.shape[1])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    open_t = cv2.imread(str(STATE / "template_furnace.png"), cv2.IMREAD_GRAYSCALE)
    closed_t = cv2.imread(str(STATE / "template_closed.png"), cv2.IMREAD_GRAYSCALE)
    peaks_c = peaks_in_band(gray, closed_t, thr=0.30) if closed_t is not None else []
    peaks_o = peaks_in_band(gray, open_t, thr=0.40) if open_t is not None else []
    print("closed in-band", peaks_c)
    print("open in-band", peaks_o)

    # Prefer open-template hits for door ROIs (dark cavity / door face), else closed
    peaks = peaks_o if len(peaks_o) >= 2 else peaks_c
    if len(peaks) < 2:
        peaks = peaks_c + peaks_o
        peaks.sort(key=lambda p: p[0])
        # de-dupe x
        dedup = []
        for p in peaks:
            if any(abs(p[0] - d[0]) < MIN_DX for d in dedup):
                continue
            dedup.append(p)
        peaks = dedup

    tw = peaks[0][2] if peaks else (closed_t.shape[1] if closed_t is not None else 63)
    th = peaks[0][3] if peaks else (closed_t.shape[0] if closed_t is not None else 50)

    if len(peaks) < 1:
        # Last resort: known good row from earlier open peaks on this camera
        print("WARN: no in-band peaks; using manual row anchors")
        y, dx, x0 = 135, 70, 218
        peaks = [(x0 + i * dx, y, tw, th, 0.5) for i in range(4)]
    else:
        peaks = even_four(peaks, tw, th, frame_w)
        print("even four", peaks)

    slots = [
        {
            "id": i,
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "score": float(sc),
            "occluded": False,
            "label": "CLOSED",
        }
        for i, (x, y, w, h, sc) in enumerate(peaks[:4])
    ]

    cfg = fa.load_furnace_config()
    cfg["slots"] = slots
    cfg["n_slots"] = 4
    cfg["fixed_slots_only"] = True
    cfg["state_mode"] = "classify"
    cfg["keep_occluded"] = True
    cfg["note"] = "fixed 4 furnace slots on row — geometry pinned, no freestyle"
    cfg["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    fa.save_furnace_config(cfg)

    objs = fa.slots_from_config(cfg)
    labeled = fa.classify_slots_door_state(frame, objs, cfg=cfg)
    ov = fa.draw_furnace_overlay(frame, labeled)
    # Draw row guide so we can verify pin
    y_mid = slots[0]["y"] + slots[0]["h"] // 2
    cv2.line(ov, (0, y_mid), (frame.shape[1] - 1, y_mid), (255, 255, 0), 1)
    cv2.imwrite(str(PREVIEW), ov)
    cv2.imwrite(str(STATE / "preview_latest.png"), ov)
    for s in labeled:
        print(f"slot{s.id} @({s.x},{s.y}) {s.label} {s.score:.2f}")
    print("wrote", PREVIEW)
    print(json.dumps({"slots": slots}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
