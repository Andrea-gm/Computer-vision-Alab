"""Seed OPEN cavity look, keep 4 pinned slots, preview OPEN-vs-not classify."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import furnace_agents as fa  # noqa: E402

STATE = ROOT / "furnace_state"
OPEN_SRC = Path(
    r"C:\Users\Ceder-ALAB\.cursor\projects"
    r"\c-Users-Ceder-ALAB-Documents-GitHub-alab-one\assets"
    r"\c__Users_Ceder-ALAB_AppData_Roaming_Cursor_User_workspaceStorage"
    r"_3d6bbacd5f40cbd7d1aac7c702b53646_images_image-d6341105-e4c6-420d-97a1-eb8484ac95e1.png"
)
PREVIEW = STATE / "preview_fixed_slots.png"


def main() -> int:
    dst = fa.seed_template(OPEN_SRC)
    print("seeded OPEN", dst, "alt", fa.DEFAULT_TEMPLATE_OPEN_ALT.is_file())

    cfg = fa.load_furnace_config()
    slots = cfg.get("slots") or []
    if len(slots) != 4:
        print("WARN expected 4 slots, have", len(slots))
    # Pin geometry; live classify overwrites labels.
    pinned = [
        {"id": 0, "x": 220, "y": 135, "w": 63, "h": 50, "score": 0.99, "occluded": False, "label": "CLOSED"},
        {"id": 1, "x": 287, "y": 135, "w": 63, "h": 50, "score": 0.99, "occluded": False, "label": "CLOSED"},
        {"id": 2, "x": 354, "y": 135, "w": 63, "h": 50, "score": 0.99, "occluded": False, "label": "CLOSED"},
        {"id": 3, "x": 421, "y": 135, "w": 63, "h": 50, "score": 0.99, "occluded": False, "label": "CLOSED"},
    ]
    if len(slots) == 4:
        for i, s in enumerate(slots):
            pinned[i]["x"] = int(s["x"])
            pinned[i]["y"] = int(s["y"])
            pinned[i]["w"] = int(s["w"])
            pinned[i]["h"] = int(s["h"])
    cfg["slots"] = pinned
    cfg["state_mode"] = "classify"
    cfg["fixed_slots_only"] = True
    cfg["keep_occluded"] = True
    cfg["open_state_threshold"] = 0.48
    cfg["open_cavity_delta"] = -999.0
    cfg["classify_rule"] = "open_match_else_closed"
    cfg["template_open_alt"] = "template_furnace_alt.png"
    cfg["note"] = (
        "4 fixed slots; OPEN = max NCC vs open templates (dark+cream) "
        "or door-down cue; else CLOSED; no cavity-delta gate"
    )
    cfg["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    fa.save_furnace_config(cfg)

    frame = None
    try:
        frames = fa.grab_frames_ros(
            3, topic="/orbbec/cam0/color/image_raw", timeout_s=20.0, interval_s=0.3
        )
        frame = frames[-1] if frames else None
    except Exception as exc:  # noqa: BLE001
        print("ROS grab skipped:", exc)
    if frame is None:
        cached = STATE / "live_for_slots.png"
        frame = cv2.imread(str(cached)) if cached.is_file() else None
        print("using cached", cached.name if frame is not None else None)
    if frame is None:
        print("ERROR: no frame")
        return 1
    cv2.imwrite(str(STATE / "live_for_slots.png"), frame)

    objs = fa.slots_from_config(cfg)
    open_tmpls = fa.load_open_templates(cfg)
    print("n_open_tmpls", len(open_tmpls), "shapes", [t.shape for t in open_tmpls])
    for s in objs:
        roi = frame[s.y : s.y + s.h, s.x : s.x + s.w]
        open_s = fa._open_match_score(roi, open_tmpls)
        delta = fa.cavity_darkness_delta(roi)
        door = fa.door_down_delta(roi)
        print(
            f"slot{s.id} @({s.x},{s.y}) open_ncc={open_s:.3f} "
            f"delta={delta:.1f} door_down={door:.1f}"
        )

    fa.reset_door_smoother(hold_hits=1)
    labeled = fa.classify_slots_door_state(frame, objs, cfg=cfg, smooth=False)
    ov = fa.draw_furnace_overlay(frame, labeled)
    cv2.imwrite(str(PREVIEW), ov)
    cv2.imwrite(str(STATE / "preview_latest.png"), ov)
    for s in labeled:
        print(f"  -> {s.label} {s.score:.2f}")
    print("wrote", PREVIEW)
    print(
        json.dumps(
            {
                "n_slots": len(labeled),
                "rule": cfg["classify_rule"],
                "open_thr": cfg["open_state_threshold"],
                "open_cavity_delta": cfg["open_cavity_delta"],
                "labels": [s.label for s in labeled],
                "xywh": [[s.x, s.y, s.w, s.h] for s in labeled],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
