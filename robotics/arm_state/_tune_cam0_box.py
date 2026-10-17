"""Debug cam0 arm box vs YOLO; write overlays."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(
    0, str(ROOT.parent / "ros2_ws" / "install" / "Lib" / "site-packages")
)

import arm_agents as aa  # noqa: E402
from orbbec_dual_sync.arm_overlay_draw import (  # noqa: E402
    default_arm_weights,
    get_worker,
)

STATE = Path(__file__).resolve().parent
img = None
caps = sorted(
    (STATE / "captures" / "cam0").glob("cam0_*.png"),
    key=lambda p: p.stat().st_mtime,
)
if caps:
    img = cv2.imread(str(caps[-1]))
    print("using capture", caps[-1].name)
if img is None:
    img = cv2.imread(str(ROOT / "furnace_state" / "live_for_slots.png"))
assert img is not None
cv2.imwrite(str(STATE / "live_cam0.png"), img)
hs, ws = img.shape[:2]
color = aa._arm_color_box(img, pad=48, cam="cam0")
print("color", color, "frame", ws, hs)
w = default_arm_weights(ROOT)
wrk = get_worker(w, 0.2)
boxes = wrk.predict(img, cam="cam0", refine=False)
print("yolo", boxes)
ref = wrk.predict(img, cam="cam0", refine=True)
print("refined", ref)

ov = img.copy()
if color:
    x, y, ww, hh = color
    cv2.rectangle(ov, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
    cv2.putText(
        ov, "color", (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
    )
for xyxy, c in boxes:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 128, 255), 2)
for xyxy, c in ref:
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(ov, (x1, y1), (x2, y2), (255, 220, 0), 2)
    cv2.putText(
        ov,
        f"arm {c:.2f}",
        (x1, max(16, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 220, 0),
        2,
    )
cv2.imwrite(str(STATE / "preview_cam0_compare.png"), ov)
print("wrote preview_cam0_compare.png")
