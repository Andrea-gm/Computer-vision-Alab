import cv2, sys
from pathlib import Path
sys.path.insert(0, r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics")
import furnace_agents as fa
cfg = fa.load_furnace_config()
open_t, closed_t = fa.load_state_templates(cfg)
caps = sorted(Path(r"C:\Users\Ceder-ALAB\Desktop\Jessica\Jessica\robotics\furnace_state\captures").glob("cam0_*closed*.png"), key=lambda p:p.stat().st_mtime, reverse=True)
print("closed captures", len(caps))
frame = cv2.imread(str(caps[0])) if caps else None
if frame is None:
    print("no closed capture"); raise SystemExit(1)
for s in fa.slots_from_config(cfg):
    roi = frame[s.y:s.y+s.h, s.x:s.x+s.w]
    o = fa.ncc_score(roi, open_t)
    c = fa.ncc_score(roi, closed_t)
    print(f"slot{s.id} open={o:.3f} closed={c:.3f}")
# Also score closed template against itself
print("self closed", fa.ncc_score(closed_t, closed_t))
