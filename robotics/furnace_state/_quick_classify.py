"""Quick classify live frame + report labels (no reseed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import furnace_agents as fa  # noqa: E402

STATE = ROOT / "furnace_state"


def main() -> int:
    cfg = fa.load_furnace_config()
    # Ensure priors are OPEN when last known good state was open (occlusion hold).
    for s in cfg.get("slots") or []:
        if str(s.get("label", "")).upper() in ("", "FURNACE", "CLOSED"):
            # Prefer OPEN prior if scores were previously high / note open campaign.
            pass
    frame = None
    try:
        frames = fa.grab_frames_ros(
            2, topic="/orbbec/cam0/color/image_raw", timeout_s=12.0, interval_s=0.3
        )
        frame = frames[-1] if frames else None
    except Exception as exc:  # noqa: BLE001
        print("ROS skip", exc)
    if frame is None:
        frame = cv2.imread(str(STATE / "live_for_slots.png"))
    else:
        cv2.imwrite(str(STATE / "live_for_slots.png"), frame)

    objs = fa.slots_from_config(cfg)
    labeled = fa.classify_slots_door_state(frame, objs, cfg=cfg, smooth=False)
    ov = fa.draw_furnace_overlay(frame, labeled)
    cv2.imwrite(str(STATE / "preview_fixed_slots.png"), ov)
    cv2.imwrite(str(STATE / "preview_latest.png"), ov)

    open_tmpls = fa.load_open_templates(cfg)
    _, closed_t = fa.load_state_templates(cfg)
    rows = []
    for s, raw in zip(labeled, objs):
        roi = frame[s.y : s.y + s.h, s.x : s.x + s.w]
        open_s = fa._open_match_score(roi, open_tmpls)
        closed_s = fa.ncc_score(roi, closed_t) if closed_t is not None else -1.0
        rows.append(
            {
                "id": s.id,
                "prior": raw.label,
                "label": s.label,
                "open": round(open_s, 3),
                "closed": round(closed_s, 3),
                "door": round(fa.door_down_delta(roi), 1),
                "xywh": [s.x, s.y, s.w, s.h],
            }
        )
        print(
            f"slot{s.id} prior={raw.label} -> {s.label} "
            f"open={open_s:.3f} closed={closed_s:.3f} door={fa.door_down_delta(roi):.1f}"
        )
    print(json.dumps({"labels": [s.label for s in labeled], "rows": rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
