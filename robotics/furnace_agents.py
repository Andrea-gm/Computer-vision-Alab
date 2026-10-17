"""Multi-agent Orbbec furnace CV pipeline (camera 0 only).

Furnaces appear only on cam0 as nested rectangles (dark frame / light border /
dark cavity). Up to 4 fixed slots; keep drawing boxes when the robot occludes.

Agents (separate processes):
  CaptureAgent — grab cam0 frames on an interval -> furnace_state/captures/
  TrainAgent   — multi-scale NCC vs seeded template; write furnace.json slots
  PreviewAgent — overlay boxes + "furnace" labels (incl. occluded fixed slots)

Frame source:
  Prefer USB CAP_OBSENSOR index 0, unless ROS cam0 publisher is already up
  (then /orbbec/cam0/color/image_raw BEST_EFFORT). Does NOT launch dual ROS.

Examples:
  cd C:\\Users\\Ceder-ALAB\\Desktop\\Jessica\\Jessica\\robotics
  C:\\Users\\Ceder-ALAB\\anaconda3\\python.exe furnace_agents.py seed
  C:\\Users\\Ceder-ALAB\\anaconda3\\python.exe furnace_agents.py run --seconds 120
  C:\\Users\\Ceder-ALAB\\anaconda3\\python.exe furnace_agents.py train
  C:\\Users\\Ceder-ALAB\\anaconda3\\python.exe furnace_agents.py preview --seconds 20
  C:\\Users\\Ceder-ALAB\\anaconda3\\python.exe furnace_agents.py stop
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "furnace_state"
CAPTURES_DIR = STATE_DIR / "captures"
KEEPS_DIR = STATE_DIR / "keeps"
OVERLAYS_DIR = STATE_DIR / "overlays"
TEMPLATES_DIR = STATE_DIR / "templates"
DEFAULT_FURNACE_JSON = STATE_DIR / "furnace.json"
DEFAULT_TEMPLATE = STATE_DIR / "template_furnace.png"  # OPEN look (location + open match)
DEFAULT_TEMPLATE_OPEN_ALT = STATE_DIR / "template_furnace_alt.png"  # cream/live open variant
DEFAULT_TEMPLATE_CLOSED = STATE_DIR / "template_closed.png"  # diagnostic only; never wins
DEFAULT_SEED = STATE_DIR / "seed_crop.png"
DEFAULT_SEED_CLOSED = STATE_DIR / "seed_closed.png"
PID_FILE = STATE_DIR / "agents.pids"
RUN_LOG = STATE_DIR / "agents.log"
TRAIN_LOG = STATE_DIR / "train.jsonl"
SCORE_LOG = STATE_DIR / "capture_scores.jsonl"

DEFAULT_TOPIC = "/orbbec/cam0/color/image_raw"
DEFAULT_CAMERA = 0
MAX_SLOTS = 4
WARMUP_GRABS = 60
DEFAULT_INTERVAL_S = 0.5
DEFAULT_TRAIN_EVERY_S = 8.0
DEFAULT_THRESHOLD = 0.58
DEFAULT_SCALES = (0.75, 0.9, 1.0, 1.15, 1.3)
# Door/portal state shown on overlays: OPEN | CLOSED (not the rack tray state).
VALID_FURNACE_STATES = ("OPEN", "CLOSED")
DEFAULT_STATE = "CLOSED"
# OPEN only when the fixed ROI matches the open-cavity template; else CLOSED.
DEFAULT_CLOSED_STATE_THRESHOLD = 0.32
DEFAULT_OPEN_STATE_THRESHOLD = 0.48
DEFAULT_CLOSED_NOT_CLOSED_MAX = 0.50
# Legacy diagnostic only; cavity darkness is NOT required for OPEN (cam0 open
# doors often have bright cream interiors → negative mean-center delta).
DEFAULT_OPEN_CAVITY_DELTA = -999.0
NMS_IOU = 0.25
# Reject templates that stretch far from the seed aspect (~63x50).
ASPECT_TOL = 0.35
MIN_SCORE_PRIMARY = 0.62
SENTINEL = None

# Bundled seed path (Cursor asset); also accept furnace_state/seed_crop.png.
_CURSOR_ASSETS = Path(
    r"C:\Users\Ceder-ALAB\.cursor\projects"
    r"\c-Users-Ceder-ALAB-Documents-GitHub-alab-one\assets"
)


def _agent_log(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(f"%(asctime)s [{name}] %(levelname)s %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        # Also append to shared file.
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(RUN_LOG, encoding="utf-8")
        fh.setFormatter(
            logging.Formatter(f"%(asctime)s [{name}] %(levelname)s %(message)s", "%H:%M:%S")
        )
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def ensure_dirs() -> None:
    for d in (STATE_DIR, CAPTURES_DIR, KEEPS_DIR, OVERLAYS_DIR, TEMPLATES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Camera / ROS
# ---------------------------------------------------------------------------


def open_orbbec(camera_index: int = DEFAULT_CAMERA) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_OBSENSOR)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open Orbbec index {camera_index} (CAP_OBSENSOR)")
    return cap


def grab_color(cap: cv2.VideoCapture) -> np.ndarray | None:
    if not cap.grab():
        return None
    ok, color = cap.retrieve(None, cv2.CAP_OBSENSOR_BGR_IMAGE)
    if ok and color is not None:
        return color
    return None


def warmup_camera(cap: cv2.VideoCapture, grabs: int = WARMUP_GRABS) -> np.ndarray | None:
    last = None
    for _ in range(int(grabs)):
        frame = grab_color(cap)
        if frame is not None:
            last = frame
        time.sleep(0.02)
    return last


def _ros_site_packages() -> list[Path]:
    ros_ws = ROOT.parent / "ros2_ws"
    return [
        ros_ws / "install" / "Lib" / "site-packages",
        ros_ws / "src" / "orbbec_dual_sync",
    ]


def _orbbec_ros_process_running() -> bool:
    """Heuristic: orbbec_camera / dual_sync publishers already own USB."""
    try:
        import subprocess as _sp

        out = _sp.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'orbbec_camera|orbbec_dual' } | "
             "Select-Object -First 1 -ExpandProperty ProcessId"],
            text=True,
            timeout=8,
        )
        return bool(out.strip())
    except Exception:
        return False


def ros_cam0_alive(topic: str = DEFAULT_TOPIC, timeout_s: float = 4.0) -> bool:
    """True if at least one message arrives on cam0 color topic."""
    try:
        for p in _ros_site_packages():
            s = str(p)
            if p.is_dir() and s not in sys.path:
                sys.path.insert(0, s)
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from sensor_msgs.msg import Image
    except Exception:
        # Publishers may still be up even if this process cannot import rclpy.
        return _orbbec_ros_process_running()

    got = {"ok": False}
    created_ctx = False
    try:
        if not rclpy.ok():
            rclpy.init()
            created_ctx = True
        node = Node(f"furnace_ros_probe_{os.getpid()}")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        def cb(_msg: Image) -> None:
            got["ok"] = True

        node.create_subscription(Image, topic, cb, qos)
        t0 = time.time()
        while not got["ok"] and time.time() - t0 < timeout_s:
            rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
    except Exception:
        return _orbbec_ros_process_running()
    if got["ok"]:
        return True
    return _orbbec_ros_process_running()


def grab_frames_ros(
    n: int,
    *,
    topic: str = DEFAULT_TOPIC,
    timeout_s: float = 30.0,
    interval_s: float = 0.0,
) -> list[np.ndarray]:
    for p in _ros_site_packages():
        s = str(p)
        if p.is_dir() and s not in sys.path:
            sys.path.insert(0, s)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from sensor_msgs.msg import Image

    from orbbec_dual_sync.image_codec import imgmsg_to_numpy

    frames: list[np.ndarray] = []
    last_t = 0.0
    if not rclpy.ok():
        rclpy.init()
    node = Node("furnace_capture_ros")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )

    def cb(msg: Image) -> None:
        nonlocal last_t
        if len(frames) >= n:
            return
        now = time.time()
        if interval_s > 0 and (now - last_t) < interval_s:
            return
        try:
            img = imgmsg_to_numpy(msg)
            if img is not None and img.ndim == 3:
                frames.append(img.copy())
                last_t = now
        except Exception:
            pass

    node.create_subscription(Image, topic, cb, qos)
    t0 = time.time()
    while len(frames) < n and time.time() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.01)
    node.destroy_node()
    return frames


def grab_one_frame(
    *,
    source: str = "auto",
    camera: int = DEFAULT_CAMERA,
    topic: str = DEFAULT_TOPIC,
    warmup: int = WARMUP_GRABS,
) -> tuple[Optional[np.ndarray], str]:
    """Return (bgr, source_used). source: auto|usb|ros."""
    use = source
    if use == "auto":
        use = "ros" if ros_cam0_alive(topic) else "usb"
    if use == "ros":
        frames = grab_frames_ros(1, topic=topic, timeout_s=12.0)
        if frames:
            return frames[0], "ros"
        if source == "ros":
            return None, "ros"
        use = "usb"
    try:
        cap = open_orbbec(camera)
    except Exception:
        return None, "usb"
    try:
        frame = warmup_camera(cap, warmup)
        return frame, "usb"
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Config / slots
# ---------------------------------------------------------------------------


@dataclass
class FurnaceSlot:
    id: int
    x: int
    y: int
    w: int
    h: int
    score: float = 0.0
    occluded: bool = False
    label: str = "furnace"


def normalize_furnace_state(raw: Any) -> str:
    s = str(raw or DEFAULT_STATE).strip().upper()
    if s in ("OPEN", "OPENED"):
        return "OPEN"
    if s in ("CLOSED", "CLOSE", "SHUT"):
        return "CLOSED"
    if s in ("FURNACE", ""):
        return DEFAULT_STATE
    return s if s in VALID_FURNACE_STATES else DEFAULT_STATE


def ncc_score(roi_bgr: np.ndarray, template_bgr: np.ndarray) -> float:
    """Normalized cross-correlation of ROI vs template (resized to template size)."""
    if roi_bgr is None or template_bgr is None or roi_bgr.size == 0:
        return -1.0
    th, tw = template_bgr.shape[:2]
    if th < 4 or tw < 4:
        return -1.0
    roi = cv2.resize(roi_bgr, (tw, th), interpolation=cv2.INTER_AREA)
    a = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    a = cv2.GaussianBlur(a, (3, 3), 0)
    b = cv2.GaussianBlur(b, (3, 3), 0)
    a = (a - float(a.mean())) / (float(a.std()) + 1e-6)
    b = (b - float(b.mean())) / (float(b.std()) + 1e-6)
    return float((a * b).mean())


def cavity_darkness_delta(roi_bgr: np.ndarray) -> float:
    """Positive when the inner rectangle is darker than the whole ROI (open cavity)."""
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return 0.0
    center = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    if center.size == 0:
        return 0.0
    return float(gray.mean()) - float(center.mean())


def _open_match_score(
    roi_bgr: np.ndarray,
    open_tmpls: list[np.ndarray],
) -> float:
    """Best NCC of ROI vs any OPEN appearance template (dark cavity and/or cream)."""
    best = -1.0
    for tmpl in open_tmpls:
        if tmpl is None or tmpl.size == 0:
            continue
        best = max(best, ncc_score(roi_bgr, tmpl))
    return float(best)


def door_down_delta(roi_bgr: np.ndarray) -> float:
    """Positive when the bottom band is brighter than the top (door flipped down)."""
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, _w = gray.shape[:2]
    if h < 8:
        return 0.0
    bot = float(gray[int(h * 0.65) :, :].mean())
    top = float(gray[: int(h * 0.35), :].mean())
    return bot - top


def classify_furnace_door(
    roi_bgr: np.ndarray,
    *,
    open_tmpl: Optional[np.ndarray],
    closed_tmpl: Optional[np.ndarray] = None,
    closed_threshold: float = DEFAULT_CLOSED_STATE_THRESHOLD,
    open_threshold: float = DEFAULT_OPEN_STATE_THRESHOLD,
    closed_not_closed_max: float = DEFAULT_CLOSED_NOT_CLOSED_MAX,
    open_cavity_delta: float = DEFAULT_OPEN_CAVITY_DELTA,
    open_tmpls: Optional[list[np.ndarray]] = None,
) -> tuple[str, float, float, bool]:
    """OPEN if fixed ROI matches an open-door look; else CLOSED.

    Positions never move. Rule is open-match-else-closed:
      1) best NCC vs OPEN template(s) (dark cavity + optional cream alt)
      2) else door-down cue (bright bottom band) when NCC is near-threshold
    Closed-template scores never win. Cavity darkness is diagnostic only
    (cam0 open doors can be bright cream → negative mean-center delta).
    Returns (label, open_score, cavity_delta, ambiguous).
    ``ambiguous`` is True when neither open nor closed templates match
    (typical robot occlusion) — caller should keep the prior slot label.
    """
    tmpls: list[np.ndarray] = []
    if open_tmpls:
        tmpls.extend([t for t in open_tmpls if t is not None])
    elif open_tmpl is not None:
        tmpls.append(open_tmpl)
    open_s = _open_match_score(roi_bgr, tmpls) if tmpls else -1.0
    closed_s = (
        ncc_score(roi_bgr, closed_tmpl) if closed_tmpl is not None else -1.0
    )
    delta = cavity_darkness_delta(roi_bgr)
    # Optional legacy gate: only when open_cavity_delta is set meaningfully.
    delta_ok = True
    if float(open_cavity_delta) > -100.0:
        delta_ok = delta >= float(open_cavity_delta)
    if tmpls and open_s >= float(open_threshold) and delta_ok:
        return "OPEN", open_s, delta, False
    # Near-miss NCC: bright flipped-down door still counts as OPEN.
    if tmpls and open_s >= float(open_threshold) - 0.15 and door_down_delta(roi_bgr) >= 8.0:
        return "OPEN", open_s, delta, False
    # Clear closed-door match (closed template wins only as positive evidence).
    if closed_tmpl is not None and closed_s >= float(closed_threshold) and open_s < float(
        closed_not_closed_max
    ):
        return "CLOSED", open_s, delta, False
    # Neither look matches → ambiguous (occlusion / glare); keep prior label.
    if open_s < float(open_threshold) and (
        closed_tmpl is None or closed_s < float(closed_threshold)
    ):
        return "CLOSED", open_s, delta, True
    return "CLOSED", open_s, delta, False


def load_open_templates(
    cfg: Optional[dict[str, Any]] = None,
) -> list[np.ndarray]:
    """Load primary OPEN template plus optional cream/live alt."""
    cfg = cfg or load_furnace_config()
    open_name = str(cfg.get("template") or DEFAULT_TEMPLATE.name)
    alt_name = str(cfg.get("template_open_alt") or DEFAULT_TEMPLATE_OPEN_ALT.name)
    paths = [STATE_DIR / open_name, STATE_DIR / alt_name]
    if paths[0] != DEFAULT_TEMPLATE:
        paths.append(DEFAULT_TEMPLATE)
    if paths[1] != DEFAULT_TEMPLATE_OPEN_ALT:
        paths.append(DEFAULT_TEMPLATE_OPEN_ALT)
    out: list[np.ndarray] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p.resolve()) if p.is_file() else ""
        if not key or key in seen:
            continue
        img = _safe_imread(p)
        if img is not None:
            seen.add(key)
            out.append(img)
    return out


def load_state_templates(
    cfg: Optional[dict[str, Any]] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (open_bgr, closed_bgr) templates from furnace_state."""
    cfg = cfg or load_furnace_config()
    open_tmpls = load_open_templates(cfg)
    open_t = open_tmpls[0] if open_tmpls else None
    closed_name = str(cfg.get("template_closed") or DEFAULT_TEMPLATE_CLOSED.name)
    closed_path = STATE_DIR / closed_name
    if not closed_path.is_file():
        closed_path = DEFAULT_TEMPLATE_CLOSED
    closed_t = _safe_imread(closed_path) if closed_path.is_file() else None
    return open_t, closed_t


class DoorStateSmoother:
    """Per-slot hysteresis so OPEN/CLOSED labels do not flicker frame-to-frame."""

    def __init__(self, hold_hits: int = 4) -> None:
        self.hold_hits = max(1, int(hold_hits))
        self._stable: dict[int, str] = {}
        self._pending: dict[int, str] = {}
        self._hits: dict[int, int] = {}

    def update(self, slot_id: int, raw: str) -> str:
        raw_u = normalize_furnace_state(raw)
        if slot_id not in self._stable:
            self._stable[slot_id] = raw_u
            self._pending.pop(slot_id, None)
            self._hits[slot_id] = 0
            return raw_u
        if raw_u == self._stable[slot_id]:
            self._pending.pop(slot_id, None)
            self._hits[slot_id] = 0
            return self._stable[slot_id]
        if self._pending.get(slot_id) == raw_u:
            self._hits[slot_id] = int(self._hits.get(slot_id, 0)) + 1
        else:
            self._pending[slot_id] = raw_u
            self._hits[slot_id] = 1
        if self._hits[slot_id] >= self.hold_hits:
            self._stable[slot_id] = raw_u
            self._pending.pop(slot_id, None)
            self._hits[slot_id] = 0
        return self._stable[slot_id]


# Shared smoother for live dual_view / overlay (module-level = one session).
_DOOR_SMOOTHER = DoorStateSmoother(hold_hits=4)


def reset_door_smoother(*, hold_hits: int = 4) -> None:
    global _DOOR_SMOOTHER
    _DOOR_SMOOTHER = DoorStateSmoother(hold_hits=hold_hits)


def classify_slots_door_state(
    frame_bgr: np.ndarray,
    slots: list[FurnaceSlot],
    *,
    cfg: Optional[dict[str, Any]] = None,
    smooth: bool = True,
) -> list[FurnaceSlot]:
    """Overwrite each slot.label via OPEN/CLOSED template match on its ROI."""
    cfg = cfg or load_furnace_config()
    mode = str(cfg.get("state_mode") or "classify").strip().lower()
    if mode in ("force_open", "force_closed", "force", "fixed"):
        forced = (
            "OPEN"
            if mode == "force_open"
            else "CLOSED"
            if mode == "force_closed"
            else normalize_furnace_state(cfg.get("state", DEFAULT_STATE))
        )
        return [
            FurnaceSlot(
                id=s.id,
                x=s.x,
                y=s.y,
                w=s.w,
                h=s.h,
                score=s.score,
                occluded=s.occluded,
                label=forced,
            )
            for s in slots
        ]

    open_tmpls = load_open_templates(cfg)
    _open_primary, closed_t = load_state_templates(cfg)
    open_t = open_tmpls[0] if open_tmpls else _open_primary
    if open_t is None and closed_t is None:
        return slots
    closed_thr = float(cfg.get("closed_state_threshold", DEFAULT_CLOSED_STATE_THRESHOLD))
    open_thr = float(cfg.get("open_state_threshold", DEFAULT_OPEN_STATE_THRESHOLD))
    not_closed_max = float(
        cfg.get("closed_not_closed_max", DEFAULT_CLOSED_NOT_CLOSED_MAX)
    )
    cavity_delta = float(cfg.get("open_cavity_delta", DEFAULT_OPEN_CAVITY_DELTA))
    hold = int(cfg.get("state_hold_hits", 4))
    if smooth and _DOOR_SMOOTHER.hold_hits != hold:
        reset_door_smoother(hold_hits=hold)
    fh, fw = frame_bgr.shape[:2]
    out: list[FurnaceSlot] = []
    for s in slots:
        x = max(0, int(s.x))
        y = max(0, int(s.y))
        w = max(1, int(s.w))
        h = max(1, int(s.h))
        if x >= fw or y >= fh:
            out.append(s)
            continue
        roi = frame_bgr[y : min(fh, y + h), x : min(fw, x + w)]
        label, open_s, _delta, ambiguous = classify_furnace_door(
            roi,
            open_tmpl=open_t,
            closed_tmpl=closed_t,
            closed_threshold=closed_thr,
            open_threshold=open_thr,
            closed_not_closed_max=not_closed_max,
            open_cavity_delta=cavity_delta,
            open_tmpls=open_tmpls or None,
        )
        if ambiguous:
            # Robot arm / glare: keep prior OPEN/CLOSED instead of flipping.
            label = normalize_furnace_state(s.label)
        if smooth:
            label = _DOOR_SMOOTHER.update(int(s.id), label)
        score = open_s
        out.append(
            FurnaceSlot(
                id=s.id,
                x=s.x,
                y=s.y,
                w=s.w,
                h=s.h,
                score=float(score) if score >= 0 else s.score,
                occluded=s.occluded,
                label=label,
            )
        )
    return out


def load_furnace_config(path: Path = DEFAULT_FURNACE_JSON) -> dict[str, Any]:
    if not path.is_file():
        return {
            "camera_index": DEFAULT_CAMERA,
            "topic": DEFAULT_TOPIC,
            "template": DEFAULT_TEMPLATE.name,
            "template_closed": DEFAULT_TEMPLATE_CLOSED.name,
            "threshold": DEFAULT_THRESHOLD,
            "scales": list(DEFAULT_SCALES),
            "slots": [],
            "frame_size": None,
            "keep_occluded": True,
            "state": DEFAULT_STATE,
            "state_mode": "classify",
            "closed_state_threshold": DEFAULT_CLOSED_STATE_THRESHOLD,
            "open_state_threshold": DEFAULT_OPEN_STATE_THRESHOLD,
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    data["state"] = normalize_furnace_state(data.get("state", DEFAULT_STATE))
    data.setdefault("template_closed", DEFAULT_TEMPLATE_CLOSED.name)
    data.setdefault("state_mode", "classify")
    return data


def save_furnace_config(data: dict[str, Any], path: Path = DEFAULT_FURNACE_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["state"] = normalize_furnace_state(data.get("state", DEFAULT_STATE))
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def slots_from_config(data: dict[str, Any]) -> list[FurnaceSlot]:
    state = normalize_furnace_state(data.get("state", DEFAULT_STATE))
    out: list[FurnaceSlot] = []
    for i, raw in enumerate(data.get("slots") or []):
        label = str(raw.get("label", state) or state)
        # Prefer global furnace door state over legacy "furnace" tag.
        if label.lower() in ("furnace", "furn", ""):
            label = state
        else:
            label = normalize_furnace_state(label)
        out.append(
            FurnaceSlot(
                id=int(raw.get("id", i)),
                x=int(raw["x"]),
                y=int(raw["y"]),
                w=int(raw["w"]),
                h=int(raw["h"]),
                score=float(raw.get("score", 0.0)),
                occluded=bool(raw.get("occluded", False)),
                label=label,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Template matching
# ---------------------------------------------------------------------------


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter) / float(union) if union > 0 else 0.0


def _nms(
    boxes: list[tuple[int, int, int, int, float]],
    iou_thresh: float = NMS_IOU,
    max_keep: int = MAX_SLOTS,
) -> list[tuple[int, int, int, int, float]]:
    boxes = sorted(boxes, key=lambda t: t[4], reverse=True)
    kept: list[tuple[int, int, int, int, float]] = []
    for b in boxes:
        if all(_iou(b[:4], k[:4]) < iou_thresh for k in kept):
            kept.append(b)
        if len(kept) >= max_keep:
            break
    return kept


def find_furnaces(
    frame_bgr: np.ndarray,
    template_bgr: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    scales: tuple[float, ...] = DEFAULT_SCALES,
    max_slots: int = MAX_SLOTS,
    label: str = DEFAULT_STATE,
) -> list[FurnaceSlot]:
    """Multi-scale grayscale NCC; return up to max_slots non-overlapping boxes."""
    label = normalize_furnace_state(label)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    tmpl0 = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    # Slight blur helps low-res template vs live noise.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    tmpl0 = cv2.GaussianBlur(tmpl0, (3, 3), 0)
    th, tw = tmpl0.shape[:2]
    if th < 8 or tw < 8:
        return []

    candidates: list[tuple[int, int, int, int, float]] = []
    fh, fw = gray.shape[:2]
    seed_aspect = float(tw) / float(th)
    for scale in scales:
        tw_s = max(16, int(round(tw * scale)))
        th_s = max(14, int(round(th * scale)))
        if tw_s >= fw or th_s >= fh:
            continue
        aspect = float(tw_s) / float(th_s)
        if abs(aspect - seed_aspect) / seed_aspect > ASPECT_TOL:
            continue
        tmpl = cv2.resize(tmpl0, (tw_s, th_s), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
        thr_s = float(threshold) + (0.06 if scale < 0.9 else 0.0)
        loc = np.where(res >= thr_s)
        for y, x in zip(loc[0].tolist(), loc[1].tolist()):
            score = float(res[y, x])
            candidates.append((int(x), int(y), int(tw_s), int(th_s), score))
        _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(res)
        if max_v >= thr_s:
            mx, my = int(max_l[0]), int(max_l[1])
            candidates.append((mx, my, int(tw_s), int(th_s), float(max_v)))

    kept = _nms(candidates, iou_thresh=NMS_IOU, max_keep=max_slots)
    # Prefer high-confidence hits; drop weak leftovers that are usually clutter.
    if any(t[4] >= MIN_SCORE_PRIMARY for t in kept):
        kept = [t for t in kept if t[4] >= MIN_SCORE_PRIMARY] or kept[:1]
    kept = sorted(kept, key=lambda t: (t[1] // 40, t[0]))
    slots: list[FurnaceSlot] = []
    for i, (x, y, w, h, score) in enumerate(kept):
        slots.append(
            FurnaceSlot(
                id=i,
                x=x,
                y=y,
                w=w,
                h=h,
                score=score,
                occluded=False,
                label=label,
            )
        )
    return slots


def merge_with_fixed_slots(
    detected: list[FurnaceSlot],
    fixed: list[FurnaceSlot],
    *,
    keep_occluded: bool = True,
    match_iou: float = 0.15,
    state: str | None = None,
) -> list[FurnaceSlot]:
    """Prefer detections; for unmatched fixed slots, keep box and mark occluded."""
    state_u = normalize_furnace_state(state) if state else None
    if not fixed:
        if state_u:
            return [
                FurnaceSlot(
                    id=d.id,
                    x=d.x,
                    y=d.y,
                    w=d.w,
                    h=d.h,
                    score=d.score,
                    occluded=d.occluded,
                    label=state_u,
                )
                for d in detected
            ]
        return detected
    if not keep_occluded and not detected:
        return []

    used_det: set[int] = set()
    merged: list[FurnaceSlot] = []

    # Assign detections to nearest fixed slot by IoU / center distance.
    for fs in fixed:
        label = state_u or normalize_furnace_state(fs.label)
        best_i = -1
        best_iou = 0.0
        fbox = (fs.x, fs.y, fs.w, fs.h)
        for i, d in enumerate(detected):
            if i in used_det:
                continue
            iou = _iou(fbox, (d.x, d.y, d.w, d.h))
            if iou > best_iou:
                best_iou = iou
                best_i = i
        if best_i >= 0 and best_iou >= match_iou:
            d = detected[best_i]
            used_det.add(best_i)
            merged.append(
                FurnaceSlot(
                    id=fs.id,
                    x=d.x,
                    y=d.y,
                    w=d.w,
                    h=d.h,
                    score=d.score,
                    occluded=False,
                    label=label,
                )
            )
        elif keep_occluded:
            merged.append(
                FurnaceSlot(
                    id=fs.id,
                    x=fs.x,
                    y=fs.y,
                    w=fs.w,
                    h=fs.h,
                    score=0.0,
                    occluded=True,
                    label=label,
                )
            )

    # Extra detections not matched to fixed slots (new furnaces).
    next_id = max([s.id for s in merged], default=-1) + 1
    for i, d in enumerate(detected):
        if i in used_det:
            continue
        if len(merged) >= MAX_SLOTS:
            break
        merged.append(
            FurnaceSlot(
                id=next_id,
                x=d.x,
                y=d.y,
                w=d.w,
                h=d.h,
                score=d.score,
                occluded=False,
                label=state_u or normalize_furnace_state(d.label),
            )
        )
        next_id += 1

    return sorted(merged, key=lambda s: s.id)[:MAX_SLOTS]


def draw_furnace_overlay(
    frame_bgr: np.ndarray,
    slots: list[FurnaceSlot],
    *,
    hide_occluded: bool = True,
) -> np.ndarray:
    """Draw furnace boxes. Occluded slots are skipped by default (arm covering)."""
    out = frame_bgr.copy()
    for s in slots:
        if s.occluded and hide_occluded:
            continue
        state = normalize_furnace_state(s.label)
        if s.occluded:
            color = (0, 165, 255)  # orange = occluded (only if hide_occluded=False)
        elif state == "OPEN":
            color = (0, 180, 255)  # amber OPEN (match rack OPEN cue)
        else:
            color = (0, 220, 0)  # green CLOSED
        cv2.rectangle(out, (s.x, s.y), (s.x + s.w, s.y + s.h), color, 2)
        tag = f"{state} {s.id}"
        if s.score > 0:
            tag += f" {s.score:.2f}"
        ty = max(16, s.y - 6)
        cv2.putText(
            out,
            tag,
            (s.x, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def _find_seed_source(explicit: Optional[Path] = None) -> Optional[Path]:
    if explicit is not None and explicit.is_file():
        return explicit
    for p in (
        DEFAULT_SEED,
        DEFAULT_TEMPLATE,
        TEMPLATES_DIR / "template_furnace.png",
    ):
        if p.is_file():
            return p
    if _CURSOR_ASSETS.is_dir():
        # Prefer newest OPEN crop (d634…), then older seeds; shorter c__ path.
        for pattern in ("*d6341105*.png", "*bb0218c8*.png", "*7866baba*.png"):
            hits = sorted(
                _CURSOR_ASSETS.glob(pattern),
                key=lambda p: (0 if p.name.startswith("c__") else 1, len(p.name)),
            )
            if hits:
                return hits[0]
    return None


def _is_greenish_pixel_mask(bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(bgr)
    g_i = g.astype(np.int16)
    return (g_i > r.astype(np.int16) + 20) & (g_i > b.astype(np.int16) + 20) & (g > 80)


def _strip_green_border(bgr: np.ndarray, *, max_inset: int = 8) -> np.ndarray:
    """Drop thin green ROI/QA overlay border if present on an OPEN crop."""
    h, w = bgr.shape[:2]
    if h < 16 or w < 16:
        return bgr

    def _edge_green_frac(strip: np.ndarray) -> float:
        if strip.size == 0:
            return 0.0
        return float(_is_greenish_pixel_mask(strip).mean())

    # Per-side insets (green overlay can be thicker on one edge).
    top_i = bot_i = left_i = right_i = 0
    for cand in range(1, max_inset + 1):
        if _edge_green_frac(bgr[cand - 1 : cand, :, :]) >= 0.45:
            top_i = cand
        else:
            break
    for cand in range(1, max_inset + 1):
        if _edge_green_frac(bgr[h - cand : h - cand + 1, :, :]) >= 0.45:
            bot_i = cand
        else:
            break
    for cand in range(1, max_inset + 1):
        if _edge_green_frac(bgr[:, cand - 1 : cand, :]) >= 0.45:
            left_i = cand
        else:
            break
    for cand in range(1, max_inset + 1):
        if _edge_green_frac(bgr[:, w - cand : w - cand + 1, :]) >= 0.45:
            right_i = cand
        else:
            break
    if top_i + bot_i + left_i + right_i == 0:
        crop = bgr
    else:
        y0, y1 = top_i, h - bot_i
        x0, x1 = left_i, w - right_i
        if y1 - y0 < 12 or x1 - x0 < 12:
            crop = bgr
        else:
            crop = bgr[y0:y1, x0:x1].copy()
    # Drop residual pure-green overlay columns/rows (anti-aliased inset).
    keep_c = [
        i
        for i in range(crop.shape[1])
        if not (
            float(crop[:, i, 1].mean()) > 180
            and float(crop[:, i, 1].mean()) > float(crop[:, i, 2].mean()) + 40
            and float(crop[:, i, 1].mean()) > float(crop[:, i, 0].mean()) + 40
        )
    ]
    if len(keep_c) >= 12:
        crop = crop[:, keep_c]
    keep_r = [
        i
        for i in range(crop.shape[0])
        if not (
            float(crop[i, :, 1].mean()) > 180
            and float(crop[i, :, 1].mean()) > float(crop[i, :, 2].mean()) + 40
            and float(crop[i, :, 1].mean()) > float(crop[i, :, 0].mean()) + 40
        )
    ]
    if len(keep_r) >= 12:
        crop = crop[keep_r, :]
    return crop


def _seed_open_alt_from_assets() -> Optional[Path]:
    """Optional cream/live OPEN variant so cam0 bright cavities also match."""
    if not _CURSOR_ASSETS.is_dir():
        return None
    hits = sorted(
        _CURSOR_ASSETS.glob("*bb0218c8*.png"),
        key=lambda p: (0 if p.name.startswith("c__") else 1, len(p.name)),
    )
    if not hits:
        return None
    raw = _safe_imread(hits[0])
    if raw is None:
        data = np.fromfile(str(hits[0]), dtype=np.uint8)
        raw = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if raw is None:
        return None
    crop = _strip_green_border(raw)
    cv2.imwrite(str(DEFAULT_TEMPLATE_OPEN_ALT), crop)
    cv2.imwrite(str(TEMPLATES_DIR / "template_furnace_alt.png"), crop)
    return hits[0]


def seed_template(source: Optional[Path] = None) -> Path:
    ensure_dirs()
    src = _find_seed_source(source)
    if src is None:
        raise FileNotFoundError(
            "No furnace seed crop found. Pass --image path/to/crop.png"
        )
    raw = _safe_imread(src)
    if raw is None:
        # Path may be very long; try copy via numpy from imdecode.
        data = np.fromfile(str(src), dtype=np.uint8)
        raw = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if raw is None:
        raise FileNotFoundError(f"Unreadable OPEN crop: {src}")
    crop = _strip_green_border(raw)
    dst = DEFAULT_TEMPLATE
    cv2.imwrite(str(dst), crop)
    cv2.imwrite(str(DEFAULT_SEED), crop)
    cv2.imwrite(str(TEMPLATES_DIR / "template_furnace.png"), crop)
    alt_src = _seed_open_alt_from_assets()
    data = load_furnace_config()
    data["template"] = dst.name
    data["template_open_alt"] = DEFAULT_TEMPLATE_OPEN_ALT.name
    data["seed_source"] = str(src)
    data["seeded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    data["classify_rule"] = "open_match_else_closed"
    data["open_cavity_delta"] = DEFAULT_OPEN_CAVITY_DELTA
    data["open_state_threshold"] = DEFAULT_OPEN_STATE_THRESHOLD
    data["fixed_slots_only"] = True
    data["state_mode"] = "classify"
    if alt_src is not None:
        data["open_alt_seed_source"] = str(alt_src)
    save_furnace_config(data)
    log = _agent_log("Seed")
    log.info(
        "seeded OPEN template %s shape=%s from %s (green-stripped); alt=%s",
        dst,
        list(crop.shape),
        src,
        DEFAULT_TEMPLATE_OPEN_ALT.is_file(),
    )
    return dst


def _strip_orange_border(bgr: np.ndarray, *, inset: int = 5) -> np.ndarray:
    """Drop thick orange QA box from a closed-crop screenshot if present."""
    h, w = bgr.shape[:2]
    if h < 20 or w < 20:
        return bgr
    inset = max(1, min(inset, h // 4, w // 4))
    return bgr[inset : h - inset, inset : w - inset].copy()


def seed_closed_template(source: Optional[Path] = None) -> Path:
    """Seed the single canonical CLOSED door template (rack-style)."""
    ensure_dirs()
    src = source
    if src is None or not Path(src).is_file():
        # Prefer newest closed crop asset, then local seed_closed.
        candidates: list[Path] = []
        if DEFAULT_SEED_CLOSED.is_file():
            candidates.append(DEFAULT_SEED_CLOSED)
        if _CURSOR_ASSETS.is_dir():
            candidates.extend(sorted(_CURSOR_ASSETS.glob("*0730b70c*.png")))
            candidates.extend(sorted(_CURSOR_ASSETS.glob("*closed*.png")))
        src = next((p for p in candidates if p.is_file()), None)
    if src is None or not Path(src).is_file():
        raise FileNotFoundError(
            "No closed furnace crop. Pass --image path/to/closed_crop.png"
        )
    src = Path(src)
    raw = cv2.imread(str(src))
    if raw is None:
        raise FileNotFoundError(f"Unreadable closed crop: {src}")
    crop = _strip_orange_border(raw, inset=5)
    open_ref = _safe_imread(DEFAULT_TEMPLATE)
    if open_ref is not None:
        th, tw = open_ref.shape[:2]
        closed = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
    else:
        closed = crop
    dst = DEFAULT_TEMPLATE_CLOSED
    cv2.imwrite(str(dst), closed)
    cv2.imwrite(str(DEFAULT_SEED_CLOSED), closed)
    cv2.imwrite(str(TEMPLATES_DIR / "template_closed.png"), closed)
    data = load_furnace_config()
    data["template_closed"] = dst.name
    data["state_mode"] = "classify"
    data["closed_seed_source"] = str(src.resolve())
    data["closed_seeded_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    data["note"] = "seeded single CLOSED template; state_mode=classify"
    save_furnace_config(data)
    log = _agent_log("Seed")
    log.info("seeded CLOSED template %s shape=%s from %s", dst, list(closed.shape), src)
    return dst


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def capture_agent(
    stop_event: mp.Event,
    *,
    source: str,
    camera: int,
    topic: str,
    interval_s: float,
    warmup: int,
    max_frames: int,
) -> None:
    log = _agent_log("CaptureAgent")
    ensure_dirs()
    use = source
    if use == "auto":
        use = "ros" if ros_cam0_alive(topic) else "usb"
    log.info("source=%s camera=%s topic=%s interval=%.2f", use, camera, topic, interval_s)
    seq = 0
    try:
        if use == "ros":
            while not stop_event.is_set():
                if max_frames > 0 and seq >= max_frames:
                    break
                frames = grab_frames_ros(
                    1, topic=topic, timeout_s=max(10.0, interval_s * 6), interval_s=0.0
                )
                if not frames:
                    log.warning("no ROS frame seq=%s — will keep trying", seq)
                    # Soft fallback: one USB grab if ROS publishers are silent.
                    if source == "auto" or seq == 0:
                        frame, used = grab_one_frame(
                            source="usb", camera=camera, topic=topic, warmup=min(warmup, 40)
                        )
                        if frame is not None:
                            use = "usb"
                            log.info("switched CaptureAgent to USB after ROS silence")
                            # Fall into USB continuous path below via break+restructure.
                            seq = _save_capture(frame, seq, used, log)
                            break
                    stop_event.wait(timeout=1.0)
                    continue
                seq = _save_capture(frames[0], seq, "ros", log)
                stop_event.wait(timeout=float(interval_s))

        if use == "usb" and not stop_event.is_set():
            # Keep one exclusive CAP_OBSENSOR handle (Train/Preview must not open USB).
            cap = None
            try:
                log.info("opening USB cam0 exclusive (warmup=%s)", warmup)
                cap = open_orbbec(camera)
                first = warmup_camera(cap, warmup)
                if first is None:
                    log.error("USB warmup failed — no frames")
                    return
                log.info("warmup ok %sx%s", first.shape[1], first.shape[0])
                seq = _save_capture(first, seq, "usb", log)
                next_t = time.monotonic() + float(interval_s)
                while not stop_event.is_set():
                    if max_frames > 0 and seq >= max_frames:
                        break
                    now = time.monotonic()
                    if now < next_t:
                        grab_color(cap)
                        time.sleep(0.01)
                        continue
                    next_t = now + float(interval_s)
                    frame = grab_color(cap)
                    if frame is None:
                        log.warning("grab failed seq=%s", seq)
                        continue
                    seq = _save_capture(frame, seq, "usb", log)
            finally:
                if cap is not None:
                    cap.release()
                    log.info("camera released")
    except Exception as exc:  # noqa: BLE001
        log.exception("CaptureAgent failed: %s", exc)
    finally:
        log.info("stopped after %s frames", seq)


def _save_capture(frame: np.ndarray, seq: int, source: str, log: logging.Logger) -> int:
    seq += 1
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = CAPTURES_DIR / f"cam0_{stamp}_seq{seq:04d}.png"
    cv2.imwrite(str(path), frame)
    _append_jsonl(
        SCORE_LOG,
        {
            "agent": "CaptureAgent",
            "seq": seq,
            "t": time.time(),
            "source": source,
            "file": path.name,
            "shape": [int(frame.shape[1]), int(frame.shape[0])],
        },
    )
    log.info("saved %s %sx%s via %s", path.name, frame.shape[1], frame.shape[0], source)
    return seq


def _safe_imread(path: Path) -> Optional[np.ndarray]:
    """Read PNG/JPG; return None on corrupt/unreadable files (skip libpng junk)."""
    try:
        if not path.is_file() or path.stat().st_size < 64:
            return None
        # IMREAD_UNCHANGED then convert avoids some partial-decode quirks.
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None or img.size == 0 or img.ndim != 3:
            return None
        h, w = img.shape[:2]
        if h < 32 or w < 32:
            return None
        return img
    except Exception:
        return None


def _select_train_frame(
    tmpl: np.ndarray,
    *,
    threshold: float,
    scales: tuple[float, ...],
    max_scan: int = 40,
    log: Optional[logging.Logger] = None,
) -> tuple[Optional[np.ndarray], Optional[Path], list[FurnaceSlot], int, int]:
    """Pick best readable capture by (#detections, score sum). Newest-first scan.

    Returns (frame, path, detected, n_readable, n_corrupt).
    """
    caps = sorted(CAPTURES_DIR.glob("cam0_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not caps:
        return None, None, [], 0, 0

    best_frame: Optional[np.ndarray] = None
    best_path: Optional[Path] = None
    best_det: list[FurnaceSlot] = []
    best_key = (-1, -1.0)
    n_readable = 0
    n_corrupt = 0
    scan = caps[: max(1, int(max_scan))]

    for path in scan:
        frame = _safe_imread(path)
        if frame is None:
            n_corrupt += 1
            if log is not None:
                log.warning("skip corrupt/unreadable capture %s", path.name)
            continue
        n_readable += 1
        detected = find_furnaces(
            frame, tmpl, threshold=threshold, scales=scales, max_slots=MAX_SLOTS
        )
        score_sum = float(sum(s.score for s in detected))
        key = (len(detected), score_sum)
        if key > best_key:
            best_key = key
            best_frame = frame
            best_path = path
            best_det = detected
            # Early exit if we already hit max slots with strong scores.
            if len(detected) >= MAX_SLOTS and score_sum >= MIN_SCORE_PRIMARY * MAX_SLOTS:
                break

    if log is not None:
        log.info(
            "capture scan readable=%s corrupt=%s scanned=%s best=%s det=%s",
            n_readable,
            n_corrupt,
            len(scan),
            best_path.name if best_path else None,
            len(best_det),
        )
    return best_frame, best_path, best_det, n_readable, n_corrupt


def train_agent(
    stop_event: mp.Event,
    *,
    train_every_s: float,
    threshold: float,
    once: bool = False,
    disk_only: bool = False,
    max_scan: int = 40,
) -> None:
    log = _agent_log("TrainAgent")
    ensure_dirs()
    if not DEFAULT_TEMPLATE.is_file():
        try:
            seed_template()
        except Exception as exc:
            log.error("no template: %s", exc)
            return
    tmpl = _safe_imread(DEFAULT_TEMPLATE)
    if tmpl is None:
        log.error("unreadable template %s", DEFAULT_TEMPLATE)
        return
    log.info(
        "ready template=%sx%s threshold=%.3f every=%.1fs disk_only=%s",
        tmpl.shape[1],
        tmpl.shape[0],
        threshold,
        train_every_s,
        disk_only,
    )
    last_train = 0.0
    try:
        while not stop_event.is_set():
            now = time.time()
            if not once and (now - last_train) < float(train_every_s):
                stop_event.wait(timeout=0.5)
                continue
            last_train = now
            cfg = load_furnace_config()
            scales = tuple(float(s) for s in (cfg.get("scales") or DEFAULT_SCALES))
            thr = float(cfg.get("threshold", threshold))
            keep_occ = bool(cfg.get("keep_occluded", True))
            state = normalize_furnace_state(cfg.get("state", DEFAULT_STATE))
            fixed = slots_from_config(cfg)
            if bool(cfg.get("fixed_slots_only")) and fixed:
                log.info("fixed_slots_only: skip detect, keep %s pinned slots", len(fixed))
                if once:
                    break
                stop_event.wait(timeout=max(1.0, float(train_every_s)))
                continue

            # Prefer disk captures (skip corrupt PNGs). One-shot scans many;
            # continuous mode still prefers newest readable, with a short scan.
            scan_n = int(max_scan) if once else 8
            frame, frame_path, detected, n_readable, _n_corrupt = _select_train_frame(
                tmpl, threshold=thr, scales=scales, max_scan=scan_n, log=log
            )
            if frame is None and not once and not disk_only:
                log.info("waiting for captures...")
                stop_event.wait(timeout=2.0)
                frame, frame_path, detected, n_readable, _n_corrupt = _select_train_frame(
                    tmpl, threshold=thr, scales=scales, max_scan=scan_n, log=log
                )
            # Live grab only if explicitly not disk-only and no readable captures.
            if frame is None and once and not disk_only and n_readable == 0:
                frame, src = grab_one_frame(
                    source="auto",
                    camera=int(cfg.get("camera_index", DEFAULT_CAMERA)),
                    topic=str(cfg.get("topic", DEFAULT_TOPIC)),
                )
                log.info("live frame source=%s (once)", src)
                if frame is not None:
                    detected = find_furnaces(
                        frame,
                        tmpl,
                        threshold=thr,
                        scales=scales,
                        max_slots=MAX_SLOTS,
                        label=state,
                    )
            if frame is None:
                log.warning("no readable frame for train (disk_only=%s)", disk_only)
                if once:
                    break
                continue

            # Apply current door state to detections from disk scan.
            detected = [
                FurnaceSlot(
                    id=d.id,
                    x=d.x,
                    y=d.y,
                    w=d.w,
                    h=d.h,
                    score=d.score,
                    occluded=d.occluded,
                    label=state,
                )
                for d in detected
            ]

            # Drop historically bad fixed slots (tiny / never scoring) so they
            # do not linger forever as fake occlusions.
            th0, tw0 = tmpl.shape[:2]
            min_area = 0.55 * float(th0 * tw0)
            fixed_clean = [
                s
                for s in fixed
                if (s.w * s.h) >= min_area and (s.score >= 0.55 or s.occluded)
            ]
            # One-shot disk train: trust fresh detections over stale fixed slots.
            if once and detected:
                slots = detected
            else:
                slots = merge_with_fixed_slots(
                    detected, fixed_clean, keep_occluded=keep_occ, state=state
                )
                if detected and not fixed_clean:
                    slots = detected

            cfg["slots"] = [
                {
                    "id": s.id,
                    "x": s.x,
                    "y": s.y,
                    "w": s.w,
                    "h": s.h,
                    "score": round(s.score, 4),
                    "occluded": s.occluded,
                    "label": state,
                }
                for s in slots
            ]
            # Pad to 4 placeholder slots only if we already have >=1 real slot
            # and user wants fixed layout — do NOT invent fake boxes from nothing.
            cfg["frame_size"] = [int(frame.shape[1]), int(frame.shape[0])]
            cfg["camera_index"] = int(cfg.get("camera_index", DEFAULT_CAMERA))
            cfg["topic"] = str(cfg.get("topic", DEFAULT_TOPIC))
            cfg["template"] = DEFAULT_TEMPLATE.name
            cfg["threshold"] = thr
            cfg["scales"] = list(scales)
            cfg["keep_occluded"] = keep_occ
            cfg["state"] = state
            cfg["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            cfg["last_train_frame"] = frame_path.name if frame_path else "live"
            cfg["n_detected"] = len(detected)
            cfg["n_slots"] = len(slots)
            cfg["n_captures_scanned"] = n_readable
            save_furnace_config(cfg)

            # Persist per-slot template crops under templates/.
            for s in slots:
                if s.occluded:
                    continue
                crop = frame[s.y : s.y + s.h, s.x : s.x + s.w]
                if crop.size == 0:
                    continue
                cv2.imwrite(str(TEMPLATES_DIR / f"furnace{s.id}.png"), crop)

            overlay = draw_furnace_overlay(frame, slots)
            ov_path = OVERLAYS_DIR / f"overlay_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(str(ov_path), overlay)
            # Also refresh a stable latest overlay.
            cv2.imwrite(str(STATE_DIR / "preview_latest.png"), overlay)

            # Crop keeps for each visible detection.
            for s in slots:
                if s.occluded:
                    continue
                crop = frame[s.y : s.y + s.h, s.x : s.x + s.w]
                if crop.size == 0:
                    continue
                keep_name = (
                    f"furnace{s.id}_{time.strftime('%Y%m%d_%H%M%S')}_s{s.score:.3f}.png"
                )
                cv2.imwrite(str(KEEPS_DIR / keep_name), crop)

            row = {
                "t": time.time(),
                "n_detected": len(detected),
                "n_slots": len(slots),
                "slots": cfg["slots"],
                "frame": cfg["last_train_frame"],
                "overlay": ov_path.name,
                "n_captures_scanned": n_readable,
            }
            _append_jsonl(TRAIN_LOG, row)
            log.info(
                "trained detected=%s slots=%s (occluded=%s) captures_used=%s -> %s",
                len(detected),
                len(slots),
                sum(1 for s in slots if s.occluded),
                n_readable,
                DEFAULT_FURNACE_JSON.name,
            )
            if once:
                break
            stop_event.wait(timeout=0.2)
    except Exception as exc:  # noqa: BLE001
        log.exception("TrainAgent failed: %s", exc)
    finally:
        log.info("stopped")


def preview_agent(
    stop_event: mp.Event,
    *,
    source: str,
    camera: int,
    topic: str,
    interval_s: float,
    show_window: bool,
) -> None:
    log = _agent_log("PreviewAgent")
    ensure_dirs()
    use = source
    if use == "auto":
        use = "ros" if ros_cam0_alive(topic) else "usb"
    log.info("preview source=%s show_window=%s", use, show_window)
    try:
        while not stop_event.is_set():
            cfg = load_furnace_config()
            slots = slots_from_config(cfg)
            frame = None
            if use == "ros":
                frames = grab_frames_ros(1, topic=topic, timeout_s=8.0)
                if frames:
                    frame = frames[0]
            # Prefer latest capture (written by CaptureAgent) over opening USB again.
            if frame is None:
                caps = sorted(CAPTURES_DIR.glob("cam0_*.png"), key=lambda p: p.stat().st_mtime)
                if caps:
                    frame = cv2.imread(str(caps[-1]))
            if frame is None and use != "ros":
                frame, _ = grab_one_frame(source="usb", camera=camera, topic=topic, warmup=8)
            if frame is None:
                stop_event.wait(timeout=1.0)
                continue
            # Re-score quickly for occlusion flags if we have a template.
            tmpl = cv2.imread(str(DEFAULT_TEMPLATE)) if DEFAULT_TEMPLATE.is_file() else None
            if tmpl is not None and slots:
                thr = float(cfg.get("threshold", DEFAULT_THRESHOLD))
                scales = tuple(float(s) for s in (cfg.get("scales") or DEFAULT_SCALES))
                state = normalize_furnace_state(cfg.get("state", DEFAULT_STATE))
                detected = find_furnaces(
                    frame, tmpl, threshold=thr, scales=scales, label=state
                )
                slots = merge_with_fixed_slots(
                    detected,
                    slots,
                    keep_occluded=bool(cfg.get("keep_occluded", True)),
                    state=state,
                )
            overlay = draw_furnace_overlay(frame, slots)
            cv2.imwrite(str(STATE_DIR / "preview_latest.png"), overlay)
            if show_window:
                cv2.imshow("furnace_agents cam0", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break
            stop_event.wait(timeout=float(interval_s))
    except Exception as exc:  # noqa: BLE001
        log.exception("PreviewAgent failed: %s", exc)
    finally:
        if show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        log.info("stopped")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def write_pids(pids: dict[str, int]) -> None:
    ensure_dirs()
    payload = {
        "pids": pids,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "root": str(ROOT),
    }
    PID_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_pids() -> dict[str, int]:
    if not PID_FILE.is_file():
        return {}
    try:
        data = json.loads(PID_FILE.read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in (data.get("pids") or {}).items()}
    except Exception:
        return {}


def stop_pids() -> int:
    log = _agent_log("Orchestrator")
    pids = read_pids()
    if not pids:
        log.info("no PID file at %s", PID_FILE)
        return 0
    n = 0
    for name, pid in pids.items():
        try:
            os.kill(pid, 9)
            log.info("killed %s pid=%s", name, pid)
            n += 1
            continue
        except Exception as exc:
            log.warning("os.kill failed %s pid=%s: %s — trying taskkill", name, pid, exc)
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                capture_output=True,
                text=True,
            )
            log.info("taskkill %s pid=%s", name, pid)
            n += 1
        except Exception as exc2:
            log.warning("could not kill %s pid=%s: %s", name, pid, exc2)
    try:
        PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    return n


def run_agents(
    *,
    seconds: float,
    interval: float,
    train_every: float,
    source: str,
    camera: int,
    topic: str,
    threshold: float,
    warmup: int,
    preview: bool,
    show_window: bool,
    background: bool,
) -> dict[str, Any]:
    ensure_dirs()
    if not DEFAULT_TEMPLATE.is_file():
        seed_template()

    cfg = load_furnace_config()
    cfg["camera_index"] = int(camera)
    cfg["topic"] = topic
    cfg["threshold"] = float(threshold)
    save_furnace_config(cfg)

    stop_event = mp.Event()
    procs = [
        mp.Process(
            target=capture_agent,
            name="CaptureAgent",
            kwargs={
                "stop_event": stop_event,
                "source": source,
                "camera": camera,
                "topic": topic,
                "interval_s": interval,
                "warmup": warmup,
                "max_frames": 0,
            },
        ),
        mp.Process(
            target=train_agent,
            name="TrainAgent",
            kwargs={
                "stop_event": stop_event,
                "train_every_s": train_every,
                "threshold": threshold,
                "once": False,
            },
        ),
    ]
    if preview:
        procs.append(
            mp.Process(
                target=preview_agent,
                name="PreviewAgent",
                kwargs={
                    "stop_event": stop_event,
                    "source": source,
                    "camera": camera,
                    "topic": topic,
                    "interval_s": max(interval, 0.75),
                    "show_window": show_window,
                },
            )
        )

    log = _agent_log("Orchestrator")
    # Resolve actual source for logging.
    resolved = source
    if source == "auto":
        resolved = "ros" if ros_cam0_alive(topic) else "usb"
    log.info(
        "starting agents source=%s seconds=%s interval=%s train_every=%s",
        resolved,
        seconds,
        interval,
        train_every,
    )
    for p in procs:
        p.start()

    pids = {p.name: int(p.pid) for p in procs if p.pid is not None}
    pids["orchestrator"] = int(os.getpid())
    write_pids(pids)
    log.info("PIDs %s -> %s", pids, PID_FILE)

    if background:
        # Detach: leave children running; parent exits after writing PIDs.
        # Children ignore parent death because they only watch stop_event.
        # For background mode we spawn a tiny watcher that sets stop after seconds.
        if seconds > 0:

            def _timer_stop(ev: mp.Event, secs: float) -> None:
                time.sleep(secs)
                ev.set()

            tproc = mp.Process(
                target=_timer_stop,
                name="TimerAgent",
                args=(stop_event, float(seconds)),
            )
            tproc.start()
            if tproc.pid:
                pids["TimerAgent"] = int(tproc.pid)
                write_pids(pids)
        return {"pids": pids, "source": resolved, "background": True}

    try:
        if seconds > 0:
            time.sleep(float(seconds))
        else:
            # Run until stop file / KeyboardInterrupt.
            while any(p.is_alive() for p in procs):
                time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — stopping")
    finally:
        stop_event.set()
        for p in procs:
            p.join(timeout=12)
            if p.is_alive():
                log.warning("terminating %s", p.name)
                p.terminate()
                p.join(timeout=3)
        try:
            PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass

    cfg = load_furnace_config()
    return {
        "pids": pids,
        "source": resolved,
        "n_slots": len(cfg.get("slots") or []),
        "n_detected": cfg.get("n_detected"),
        "furnace_json": str(DEFAULT_FURNACE_JSON),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_seed(args: argparse.Namespace) -> int:
    path = Path(args.image) if args.image else None
    dst = seed_template(path)
    print(f"Seeded template -> {dst}")
    return 0


def cmd_seed_closed(args: argparse.Namespace) -> int:
    """Seed the single CLOSED door template from a crop image."""
    ensure_dirs()
    path = Path(args.image) if getattr(args, "image", None) else None
    dst = seed_closed_template(path)
    print(f"Seeded CLOSED template -> {dst}")
    print("state_mode=classify (live OPEN/CLOSED via template match)")
    return 0


def cmd_label_state(args: argparse.Namespace) -> int:
    """Set furnace door state OPEN|CLOSED on all slots + furnace.json."""
    ensure_dirs()
    state = normalize_furnace_state(args.state)
    cfg = load_furnace_config()
    cfg["state"] = state
    # Pin overlays to this state unless --classify keeps live matching.
    if bool(getattr(args, "classify", False)):
        cfg["state_mode"] = "classify"
    else:
        cfg["state_mode"] = "force_open" if state == "OPEN" else "force_closed"
    slots = []
    for i, raw in enumerate(cfg.get("slots") or []):
        raw = dict(raw)
        raw["label"] = state
        raw["id"] = int(raw.get("id", i))
        slots.append(raw)
    cfg["slots"] = slots
    cfg["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    cfg["note"] = f"operator set furnace state={state} mode={cfg['state_mode']}"
    save_furnace_config(cfg)

    # Optional: grab a few ROS/USB frames into keeps/open or keeps/closed.
    n = int(getattr(args, "capture", 0) or 0)
    if n > 0:
        keep_dir = KEEPS_DIR / state.lower()
        keep_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        frames = grab_frames_ros(
            n,
            topic=str(cfg.get("topic", DEFAULT_TOPIC)),
            timeout_s=25.0,
            interval_s=0.3,
        )
        src = "ros"
        if not frames:
            frame, src = grab_one_frame(
                source="usb",
                camera=int(cfg.get("camera_index", DEFAULT_CAMERA)),
                topic=str(cfg.get("topic", DEFAULT_TOPIC)),
            )
            frames = [frame] if frame is not None else []
        for i, fr in enumerate(frames, start=1):
            p = keep_dir / f"{state.lower()}_{stamp}_{i:03d}.png"
            cv2.imwrite(str(p), fr)
        print(f"CaptureAgent: wrote {len(frames)} {state} keepers via {src} -> {keep_dir}")

    # Refresh preview from latest capture or fixed slots alone.
    caps = sorted(CAPTURES_DIR.glob("cam0_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    frame = _safe_imread(caps[0]) if caps else None
    slot_objs = slots_from_config(cfg)
    if frame is not None:
        overlay = draw_furnace_overlay(frame, slot_objs)
        cv2.imwrite(str(STATE_DIR / "preview_latest.png"), overlay)
        cv2.imwrite(str(STATE_DIR / f"preview_{state.lower()}.png"), overlay)
    print(f"Furnace state -> {state} ({len(slots)} slots labeled)")
    print(f"Config: {DEFAULT_FURNACE_JSON}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    ensure_dirs()
    if not DEFAULT_TEMPLATE.is_file():
        seed_template(Path(args.image) if args.image else None)
    stop = mp.Event()
    disk_only = not bool(getattr(args, "allow_live", False))
    train_agent(
        stop,
        train_every_s=0.1,
        threshold=float(args.threshold),
        once=True,
        disk_only=disk_only,
        max_scan=int(getattr(args, "max_scan", 40)),
    )
    cfg = load_furnace_config()
    slots = cfg.get("slots") or []
    print(f"Train done: n_detected={cfg.get('n_detected')} n_slots={len(slots)}")
    print(f"Captures used (readable scanned): {cfg.get('n_captures_scanned')}")
    for s in slots:
        print(
            f"  slot{s['id']}: xywh=({s['x']},{s['y']},{s['w']},{s['h']}) "
            f"score={s.get('score')} occluded={s.get('occluded')}"
        )
    print(f"Config: {DEFAULT_FURNACE_JSON}")
    print(f"Preview: {STATE_DIR / 'preview_latest.png'}")
    print(f"Templates: {TEMPLATES_DIR}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    # Detach a full foreground run so shared mp.Event stays alive with children.
    if bool(args.background):
        ensure_dirs()
        if not DEFAULT_TEMPLATE.is_file():
            seed_template()
        py = sys.executable
        cmd = [
            py,
            str(ROOT / "furnace_agents.py"),
            "run",
            "--seconds",
            str(args.seconds),
            "--interval",
            str(args.interval),
            "--train-every",
            str(args.train_every),
            "--source",
            str(args.source),
            "--camera",
            str(args.camera),
            "--topic",
            str(args.topic),
            "--threshold",
            str(args.threshold),
            "--warmup",
            str(args.warmup),
        ]
        if args.preview:
            cmd.append("--preview")
        if args.show:
            cmd.append("--show")
        log_path = STATE_DIR / "agents_stdout.log"
        flags = 0
        if sys.platform == "win32":
            flags = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        with log_path.open("a", encoding="utf-8") as lf:
            lf.write(f"\n--- launch {time.strftime('%Y-%m-%dT%H:%M:%S')} ---\n")
            lf.write(" ".join(cmd) + "\n")
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=lf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=flags,
                close_fds=True,
            )
        # Brief wait so child can write agents.pids
        time.sleep(2.5)
        pids = read_pids()
        pids["launcher"] = int(proc.pid)
        write_pids(pids)
        print(
            json.dumps(
                {
                    "background": True,
                    "launcher_pid": int(proc.pid),
                    "pids": pids,
                    "log": str(log_path),
                    "pid_file": str(PID_FILE),
                },
                indent=2,
            )
        )
        print("Stop with: furnace_agents.py stop")
        return 0

    result = run_agents(
        seconds=float(args.seconds),
        interval=float(args.interval),
        train_every=float(args.train_every),
        source=str(args.source),
        camera=int(args.camera),
        topic=str(args.topic),
        threshold=float(args.threshold),
        warmup=int(args.warmup),
        preview=bool(args.preview),
        show_window=bool(args.show),
        background=False,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    stop = mp.Event()
    p = mp.Process(
        target=preview_agent,
        name="PreviewAgent",
        kwargs={
            "stop_event": stop,
            "source": args.source,
            "camera": int(args.camera),
            "topic": args.topic,
            "interval_s": float(args.interval),
            "show_window": bool(args.show),
        },
    )
    p.start()
    write_pids({"PreviewAgent": int(p.pid), "orchestrator": int(os.getpid())})
    try:
        time.sleep(float(args.seconds))
    finally:
        stop.set()
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
        PID_FILE.unlink(missing_ok=True)
    print(f"Latest overlay: {STATE_DIR / 'preview_latest.png'}")
    return 0


def cmd_stop(_args: argparse.Namespace) -> int:
    n = stop_pids()
    print(f"Stopped {n} process(es)")
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    ensure_dirs()
    cfg = load_furnace_config()
    pids = read_pids()
    print(f"template: {DEFAULT_TEMPLATE} exists={DEFAULT_TEMPLATE.is_file()}")
    print(f"captures: {len(list(CAPTURES_DIR.glob('cam0_*.png')))}")
    print(f"keeps: {len(list(KEEPS_DIR.glob('*.png')))}")
    print(f"slots: {len(cfg.get('slots') or [])}  n_detected={cfg.get('n_detected')}")
    for s in cfg.get("slots") or []:
        print(
            f"  slot{s['id']}: ({s['x']},{s['y']},{s['w']},{s['h']}) "
            f"score={s.get('score')} occ={s.get('occluded')}"
        )
    print(f"pids: {pids}")
    print(f"preview: {STATE_DIR / 'preview_latest.png'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Furnace multi-agent CV (cam0 only)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seed", help="Copy seed crop -> furnace_state/template_furnace.png")
    s.add_argument("--image", type=str, default=None)
    s.set_defaults(func=cmd_seed)

    t = sub.add_parser("train", help="One-shot TrainAgent on disk captures (best of N)")
    t.add_argument("--image", type=str, default=None, help="Optional seed image if missing")
    t.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    t.add_argument(
        "--max-scan",
        type=int,
        default=40,
        help="Max newest captures to scan (skip corrupt)",
    )
    t.add_argument(
        "--disk-only",
        action="store_true",
        default=True,
        help="Never open camera; train from furnace_state/captures only (default)",
    )
    t.add_argument(
        "--allow-live",
        action="store_true",
        help="If no readable captures, fall back to one live cam0 grab",
    )
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("run", help="Start CaptureAgent + TrainAgent (+ optional Preview)")
    r.add_argument("--seconds", type=float, default=90.0, help="0 = until stop")
    r.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    r.add_argument("--train-every", type=float, default=DEFAULT_TRAIN_EVERY_S)
    r.add_argument("--source", choices=("auto", "usb", "ros"), default="auto")
    r.add_argument("--camera", type=int, default=DEFAULT_CAMERA)
    r.add_argument("--topic", type=str, default=DEFAULT_TOPIC)
    r.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    r.add_argument("--warmup", type=int, default=WARMUP_GRABS)
    r.add_argument("--preview", action="store_true", help="Also run PreviewAgent")
    r.add_argument("--show", action="store_true", help="Open OpenCV window (PreviewAgent)")
    r.add_argument(
        "--background",
        action="store_true",
        help="Start agents and return immediately (PIDs in furnace_state/agents.pids)",
    )
    r.set_defaults(func=cmd_run)

    pr = sub.add_parser("preview", help="PreviewAgent only")
    pr.add_argument("--seconds", type=float, default=20.0)
    pr.add_argument("--interval", type=float, default=0.75)
    pr.add_argument("--source", choices=("auto", "usb", "ros"), default="auto")
    pr.add_argument("--camera", type=int, default=DEFAULT_CAMERA)
    pr.add_argument("--topic", type=str, default=DEFAULT_TOPIC)
    pr.add_argument("--show", action="store_true")
    pr.set_defaults(func=cmd_preview)

    st = sub.add_parser("stop", help="Kill PIDs from furnace_state/agents.pids")
    st.set_defaults(func=cmd_stop)

    ss = sub.add_parser("status", help="Show slots / captures / PIDs")
    ss.set_defaults(func=cmd_status)

    ls = sub.add_parser(
        "label-state",
        help="Pin or seed furnace door OPEN/CLOSED (use --classify for live match)",
    )
    ls.add_argument(
        "state",
        choices=("open", "closed", "OPEN", "CLOSED"),
        help="Furnace door/portal state to show on overlays",
    )
    ls.add_argument(
        "--capture",
        type=int,
        default=8,
        help="Also save N live frames into keeps/<state>/ (0=skip)",
    )
    ls.add_argument(
        "--classify",
        action="store_true",
        help="Keep state_mode=classify (template match) instead of force_*",
    )
    ls.set_defaults(func=cmd_label_state)

    sc = sub.add_parser(
        "seed-closed",
        help="Seed single CLOSED door template (rack-style canonical closed)",
    )
    sc.add_argument(
        "--image",
        type=str,
        default="",
        help="Path to closed furnace crop (orange box OK; will inset)",
    )
    sc.set_defaults(func=cmd_seed_closed)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    # Windows multiprocessing
    mp.freeze_support()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
