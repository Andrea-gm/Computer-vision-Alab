"""Build full-frame small-vial stills from real labeled crops (upright + sideways).

Places vial crops into large canvases at reduced scale so single-scale YOLO at
imgsz=640 under-resolves them, while SAHI tiles recover recall. Sideways samples
use 90° rotations of upright crops (same appearance family, rotated framing).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .types import Det


def _load_crop_pairs(raw_dir: Path, limit: int = 40) -> List[Tuple[np.ndarray, Det]]:
    pairs = []
    for img_path in sorted(raw_dir.glob("*.jpg")) + sorted(raw_dir.glob("*.png")):
        lbl = img_path.with_suffix(".txt")
        if not lbl.exists():
            continue
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        text = lbl.read_text(encoding="utf-8").strip()
        if not text:
            continue
        parts = text.splitlines()[0].split()
        if len(parts) != 5:
            continue
        cx, cy, bw, bh = [float(v) for v in parts[1:]]
        x1 = int(max(0, (cx - bw / 2) * w))
        y1 = int(max(0, (cy - bh / 2) * h))
        x2 = int(min(w, (cx + bw / 2) * w))
        y2 = int(min(h, (cy + bh / 2) * h))
        if x2 - x1 < 12 or y2 - y1 < 12:
            continue
        crop = bgr[y1:y2, x1:x2].copy()
        pairs.append((crop, Det(0, 0, crop.shape[1], crop.shape[0], 1.0, source="gt")))
        if len(pairs) >= limit:
            break
    return pairs


def build_small_object_items(
    camera_root: Path,
    *,
    canvas_w: int = 1280,
    canvas_h: int = 720,
    target_max_side: int = 64,
    seed: int = 42,
    per_orient: int = 24,
) -> dict:
    """Return {upright: [...], sideways: [...]} eval items with path=None and bgr inline.

    Each item: {id, gts, bgr, kind, orientation}
    """
    rng = np.random.default_rng(seed)
    raw = camera_root / "vial_dataset" / "raw"
    crops = _load_crop_pairs(raw, limit=60)
    if not crops:
        return {"upright": [], "sideways": []}

    out = {"upright": [], "sideways": []}
    for orient in ("upright", "sideways"):
        for i in range(per_orient):
            crop, _ = crops[int(rng.integers(0, len(crops)))]
            if orient == "sideways":
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            ch, cw = crop.shape[:2]
            scale = target_max_side / float(max(ch, cw))
            # Slight jitter so not all identical scale
            scale *= float(rng.uniform(0.7, 1.05))
            nw = max(8, int(round(cw * scale)))
            nh = max(8, int(round(ch * scale)))
            small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

            canvas = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8)
            # Mild noise / desk-like texture
            noise = rng.integers(0, 25, size=canvas.shape, dtype=np.uint8)
            canvas = cv2.add(canvas, noise)

            max_x = max(0, canvas_w - nw - 1)
            max_y = max(0, canvas_h - nh - 1)
            x0 = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
            y0 = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
            canvas[y0 : y0 + nh, x0 : x0 + nw] = small
            gt = Det(
                float(x0),
                float(y0),
                float(x0 + nw),
                float(y0 + nh),
                conf=1.0,
                label="vial",
                source="gt",
            )
            out[orient].append(
                {
                    "id": f"small_{orient}_{i:02d}",
                    "path": None,
                    "bgr": canvas,
                    "gts": [gt],
                    "kind": f"small_{orient}",
                }
            )
    return out


def maybe_write_preview(items: dict, out_dir: Path, n: int = 2) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for orient, lst in items.items():
        for it in lst[:n]:
            vis = it["bgr"].copy()
            for g in it["gts"]:
                cv2.rectangle(
                    vis,
                    (int(g.x1), int(g.y1)),
                    (int(g.x2), int(g.y2)),
                    (0, 255, 0),
                    2,
                )
            cv2.imwrite(str(out_dir / f"{it['id']}.jpg"), vis)
