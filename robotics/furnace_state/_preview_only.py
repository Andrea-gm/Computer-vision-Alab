from pathlib import Path
import sys
import time
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import furnace_agents as fa

STATE = ROOT / "furnace_state"
cfg = fa.load_furnace_config()
cfg["state_mode"] = "classify"
cfg["fixed_slots_only"] = True
cfg["open_state_threshold"] = 0.48
cfg["open_cavity_delta"] = -999.0
cfg["classify_rule"] = "open_match_else_closed"
cfg["note"] = "4 fixed slots; OPEN = NCC match open template, else CLOSED"
cfg["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
fa.save_furnace_config(cfg)

frames = fa.grab_frames_ros(
    2, topic="/orbbec/cam0/color/image_raw", timeout_s=15.0, interval_s=0.3
)
frame = frames[-1] if frames else cv2.imread(str(STATE / "live_for_slots.png"))
if frame is None:
    raise SystemExit("no frame")
cv2.imwrite(str(STATE / "live_for_slots.png"), frame)
objs = fa.slots_from_config(cfg)
open_t, _ = fa.load_state_templates(cfg)
print("open tmpl self", fa.ncc_score(open_t, open_t), "delta", fa.cavity_darkness_delta(open_t))
labeled = fa.classify_slots_door_state(frame, objs, cfg=cfg, smooth=False)
for s in labeled:
    roi = frame[s.y : s.y + s.h, s.x : s.x + s.w]
    print(
        f"slot{s.id} @({s.x},{s.y}) {s.label} ncc={s.score:.3f} "
        f"delta={fa.cavity_darkness_delta(roi):.1f}"
    )
ov = fa.draw_furnace_overlay(frame, labeled)
cv2.imwrite(str(STATE / "preview_fixed_slots.png"), ov)
cv2.imwrite(str(STATE / "preview_latest.png"), ov)
print("wrote preview")
