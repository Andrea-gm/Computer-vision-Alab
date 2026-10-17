from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import furnace_agents as fa

STATE = ROOT / "furnace_state"
frame = cv2.imread(str(STATE / "live_for_slots.png"))
open_t = cv2.imread(str(STATE / "template_furnace.png"))
closed_t = cv2.imread(str(STATE / "template_closed.png"))


def stats(name, bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    cy0, cy1 = h // 4, 3 * h // 4
    cx0, cx1 = w // 4, 3 * w // 4
    center = g[cy0:cy1, cx0:cx1]
    b, gch, r = cv2.split(bgr)
    cen_bgr = bgr[cy0:cy1, cx0:cx1]
    print(
        f"{name:12} shape={bgr.shape[:2]} mean={g.mean():.1f} "
        f"center={center.mean():.1f} std={center.std():.1f} "
        f"cenBGR=({cen_bgr[:,:,0].mean():.0f},{cen_bgr[:,:,1].mean():.0f},{cen_bgr[:,:,2].mean():.0f})"
    )


stats("OPEN_TMPL", open_t)
if closed_t is not None:
    stats("CLOSED_TMPL", closed_t)

cfg = fa.load_furnace_config()
for s in fa.slots_from_config(cfg):
    roi = frame[s.y : s.y + s.h, s.x : s.x + s.w]
    stats(f"slot{s.id}", roi)
