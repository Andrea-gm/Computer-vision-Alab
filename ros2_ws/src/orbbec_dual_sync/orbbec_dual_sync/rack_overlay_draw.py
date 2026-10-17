"""Pure drawing / ROI helpers for cam1 rack overlay (no rclpy)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def load_rack_rois(
    path: Path,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], list[dict]]:
    """Return (classify_xywh, display_xywh, holes)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if all(k in data for k in ("x", "y", "w", "h")):
        classify = (int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"]))
    else:
        nested = data.get("roi") or {}
        classify = (
            int(nested["x"]),
            int(nested["y"]),
            int(nested["w"]),
            int(nested["h"]),
        )
    disp = data.get("display_roi") or {}
    if all(k in disp for k in ("x", "y", "w", "h")):
        display = (int(disp["x"]), int(disp["y"]), int(disp["w"]), int(disp["h"]))
    else:
        x, y, w, h = classify
        display = (max(0, x - 40), max(0, y - 25), w + 80, h + 110)
    holes = list(data.get("holes") or [])
    return classify, display, holes


def classify_whole_rack(
    bgr: np.ndarray,
    classify_xywh: tuple[int, int, int, int],
    holes: list[dict],
    *,
    robotics_root: Optional[Path] = None,
) -> str:
    """Return OPEN or CLOSED via RackStateJudge (strict closed, looser open).

    Falls back to jar-hole majority vote only if the judge cannot load
    (missing templates / exemplars).
    """
    import sys

    if robotics_root is not None:
        root_s = str(robotics_root)
        if root_s not in sys.path:
            sys.path.insert(0, root_s)

    try:
        from rack_capture_agents import (  # noqa: PLC0415
            RackStateJudge,
            load_judge_thresholds,
            load_rack_config,
        )

        cfg = load_rack_config()
        # Prefer the live classify ROI from rack.json / node params.
        if classify_xywh and len(classify_xywh) == 4:
            cfg.roi = tuple(int(v) for v in classify_xywh)  # type: ignore[misc]
        th = load_judge_thresholds(blur_min=cfg.blur_min)
        judge = RackStateJudge(cfg, thresholds=th)
        result = judge.judge_bgr(bgr, already_roi=False)
        label = (result.label or "").strip().upper()
        if label in ("OPEN", "CLOSED"):
            return label
        # UNKNOWN (two-state UI): prefer CLOSED when closed looks as good or
        # better; only call OPEN when clearly away from the home template.
        if result.closed_score >= result.open_score:
            return "CLOSED"
        if result.closed_score < th.closed_not_closed_max:
            return "OPEN"
        return "CLOSED"
    except Exception:
        pass

    import rack_lid_state as rls  # noqa: PLC0415

    x, y, w, h = classify_xywh
    height, width = bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0 or not holes:
        return "CLOSED"
    rack_holes = [
        rls.RackHole(
            cx=int(item["cx"]),
            cy=int(item["cy"]),
            rx=int(item.get("rx", 8)),
            ry=int(item.get("ry", 5)),
        )
        for item in holes
    ]
    hits = rls.classify_holes(crop, rack_holes)
    closed = sum(1 for hit in hits if hit.state == "closed")
    opened = sum(1 for hit in hits if hit.state == "open")
    if opened == 0 and closed == 0:
        return "CLOSED"
    return "OPEN" if opened >= closed else "CLOSED"


def draw_rack_overlay(
    bgr: np.ndarray,
    display_xywh: tuple[int, int, int, int],
    state: str,
    *,
    classify_xywh: Optional[tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """Full-frame view with large rack rectangle + one OPEN/CLOSED label."""
    view = bgr.copy()
    dx, dy, dw, dh = display_xywh
    height, width = view.shape[:2]
    x0, y0 = max(0, dx), max(0, dy)
    x1, y1 = min(width - 1, dx + dw), min(height - 1, dy + dh)
    state_u = (state or "OPEN").strip().upper()
    if state_u not in ("OPEN", "CLOSED"):
        state_u = "OPEN"
    box_color = (0, 180, 255) if state_u == "OPEN" else (0, 220, 0)
    cv2.rectangle(view, (x0, y0), (x1, y1), box_color, 3)
    if classify_xywh is not None:
        cx, cy, cw, ch = classify_xywh
        cv2.rectangle(
            view,
            (max(0, cx), max(0, cy)),
            (min(width - 1, cx + cw), min(height - 1, cy + ch)),
            (80, 80, 80),
            1,
        )
    label = f"RACK {state_u}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 3
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    tx = x0 + 8
    ty = max(th + 12, y0 + th + 12)
    cv2.rectangle(
        view,
        (tx - 4, ty - th - 8),
        (tx + tw + 4, ty + 8),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        view,
        label,
        (tx, ty),
        font,
        scale,
        box_color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        view,
        "cam1 rack overlay (Q closes preview)",
        (12, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return view
