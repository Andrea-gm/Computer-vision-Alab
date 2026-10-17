"""Capture cam0 frames, template-match CLOSED, write keeps + overlays."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import furnace_agents as fa  # noqa: E402


def main() -> int:
    fa.ensure_dirs()
    cfg = fa.load_furnace_config()
    cfg["state_mode"] = "classify"
    cfg["template_closed"] = "template_closed.png"
    # Slightly friendlier closed threshold for live cam noise
    cfg.setdefault("closed_state_threshold", 0.62)
    fa.save_furnace_config(cfg)

    topic = str(cfg.get("topic", fa.DEFAULT_TOPIC))
    print(f"CaptureAgent: grabbing frames topic={topic}")
    frames = fa.grab_frames_ros(24, topic=topic, timeout_s=30.0, interval_s=0.35)
    src = "ros"
    if not frames:
        print("ROS empty — trying USB cam0")
        one, src = fa.grab_one_frame(source="usb", camera=0, topic=topic)
        frames = [one] if one is not None else []
    if not frames:
        print("ERROR: no frames from ROS or USB")
        return 1

    stamp = time.strftime("%Y%m%d_%H%M%S")
    cap_dir = fa.CAPTURES_DIR
    keep_dir = fa.KEEPS_DIR / "closed"
    ov_dir = fa.STATE_DIR / "overlays" / "closed"
    keep_dir.mkdir(parents=True, exist_ok=True)
    ov_dir.mkdir(parents=True, exist_ok=True)

    fixed = fa.slots_from_config(cfg)
    n_closed = 0
    last_slots = fixed
    for i, fr in enumerate(frames, start=1):
        cap_path = cap_dir / f"cam0_{stamp}_closed{i:03d}.png"
        cv2.imwrite(str(cap_path), fr)
        slots = fa.classify_slots_door_state(fr, fixed, cfg=cfg)
        last_slots = slots
        closed_hits = sum(1 for s in slots if s.label == "CLOSED")
        if closed_hits > 0:
            n_closed += 1
            keep = keep_dir / f"closed_{stamp}_{i:03d}.png"
            cv2.imwrite(str(keep), fr)
            ov = fa.draw_furnace_overlay(fr, slots)
            cv2.imwrite(str(ov_dir / f"closed_{stamp}_{i:03d}.png"), ov)

        # Always write a few overlays for QA even if still OPEN
        if i <= 3 or i == len(frames) or closed_hits > 0:
            ov = fa.draw_furnace_overlay(fr, slots)
            cv2.imwrite(str(ov_dir / f"frame_{stamp}_{i:03d}.png"), ov)
            scores = ", ".join(f"{s.id}:{s.label}:{s.score:.2f}" for s in slots)
            print(f"  frame {i}: {scores}")

    # Persist majority label on slots
    maj_closed = sum(1 for s in last_slots if s.label == "CLOSED") >= max(
        1, len(last_slots) // 2
    )
    state = "CLOSED" if maj_closed else "OPEN"
    cfg["state"] = state
    cfg["slots"] = [
        {
            "id": s.id,
            "x": s.x,
            "y": s.y,
            "w": s.w,
            "h": s.h,
            "score": float(s.score),
            "occluded": bool(s.occluded),
            "label": s.label,
        }
        for s in last_slots
    ]
    cfg["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    cfg["note"] = (
        f"closed capture+classify via {src}: frames={len(frames)} "
        f"frames_with_closed={n_closed} majority={state}"
    )
    fa.save_furnace_config(cfg)

    preview = fa.draw_furnace_overlay(frames[-1], last_slots)
    cv2.imwrite(str(fa.STATE_DIR / "preview_latest.png"), preview)
    cv2.imwrite(str(fa.STATE_DIR / "preview_closed.png"), preview)
    print(
        f"Done via {src}: captures={len(frames)} closed_keeps~{n_closed} "
        f"majority={state}"
    )
    print(f"Overlays: {ov_dir}")
    print(f"Preview:  {fa.STATE_DIR / 'preview_closed.png'}")
    print(f"Config:   {fa.DEFAULT_FURNACE_JSON}")
    return 0 if n_closed > 0 or state == "CLOSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
