"""Derive weak cap/base keypoint labels from YOLO AABBs + white-cap localization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .axis_skeleton import det_from_keypoints, far_end_from_box
from .cap_finder import localize_cap_in_box
from .types import Det, Keypoint, list_images


@dataclass
class WeakKeypoints:
    image: str
    box_xyxy: Tuple[float, float, float, float]
    cap: Tuple[float, float]
    base: Tuple[float, float]
    angle_deg: float
    diameter: float
    source: str = "yolo_weak"


def load_yolo_label(path: Path, width: int, height: int) -> List[Det]:
    if not path.is_file():
        return []
    dets: List[Det] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, bw, bh = [float(v) for v in parts[1:5]]
        x1 = (cx - bw * 0.5) * width
        y1 = (cy - bh * 0.5) * height
        x2 = (cx + bw * 0.5) * width
        y2 = (cy + bh * 0.5) * height
        dets.append(Det(x1=x1, y1=y1, x2=x2, y2=y2, conf=1.0, cls_id=cls_id))
    return dets


def weak_from_box(bgr: np.ndarray, det: Det) -> Optional[WeakKeypoints]:
    import math

    cap = localize_cap_in_box(bgr, det.as_xyxy())
    if cap is None:
        return None
    x1, y1, x2, y2 = det.as_xyxy()
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    # Snap cap to the brighter end of the long axis (fixes mid-body specular hits).
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    score = hsv[:, :, 2].astype(np.float32) - hsv[:, :, 1].astype(np.float32)
    if bh >= bw:
        band = max(2, int(bh * 0.22))
        top = float(np.mean(score[int(y1) : int(y1) + band, int(x1) : int(x2)]))
        bot = float(np.mean(score[int(y2) - band : int(y2), int(x1) : int(x2)]))
        if top >= bot:
            cap_xy = (float(np.clip(cap.cx, x1 + 1, x2 - 1)), float(y1 + 0.12 * bh))
        else:
            cap_xy = (float(np.clip(cap.cx, x1 + 1, x2 - 1)), float(y2 - 0.12 * bh))
    else:
        band = max(2, int(bw * 0.22))
        left = float(np.mean(score[int(y1) : int(y2), int(x1) : int(x1) + band]))
        right = float(np.mean(score[int(y1) : int(y2), int(x2) - band : int(x2)]))
        if left >= right:
            cap_xy = (float(x1 + 0.12 * bw), float(np.clip(cap.cy, y1 + 1, y2 - 1)))
        else:
            cap_xy = (float(x2 - 0.12 * bw), float(np.clip(cap.cy, y1 + 1, y2 - 1)))
    base = far_end_from_box(cap_xy, det.as_xyxy())
    angle = math.degrees(math.atan2(base[1] - cap_xy[1], base[0] - cap_xy[0]))
    return WeakKeypoints(
        image="",
        box_xyxy=det.as_xyxy(),
        cap=cap_xy,
        base=base,
        angle_deg=float(angle),
        diameter=float(cap.diameter),
    )


def derive_dataset_weak_labels(
    image_dir: Path,
    label_dir: Optional[Path] = None,
    *,
    max_images: Optional[int] = None,
) -> List[Dict]:
    """Build weak keypoint JSON records for training / diagnostics."""
    images = list_images(image_dir)
    if max_images is not None:
        images = images[: max(0, int(max_images))]
    records: List[Dict] = []
    for img_path in images:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        lbl = (label_dir or image_dir) / f"{img_path.stem}.txt"
        if label_dir is not None and label_dir.is_dir():
            lbl = label_dir / f"{img_path.stem}.txt"
        gts = load_yolo_label(lbl, w, h)
        for gi, gt in enumerate(gts):
            weak = weak_from_box(bgr, gt)
            if weak is None:
                continue
            weak.image = img_path.name
            rec = asdict(weak)
            rec["stem"] = img_path.stem
            rec["gt_index"] = gi
            records.append(rec)
    return records


def weak_det_from_record(bgr: np.ndarray, rec: Dict) -> Det:
    return det_from_keypoints(
        tuple(rec["cap"]),  # type: ignore[arg-type]
        tuple(rec["base"]),  # type: ignore[arg-type]
        float(rec["diameter"]),
        conf=0.9,
        image_shape=bgr.shape[:2],
        source="weak_label",
    )


def save_weak_labels(records: Sequence[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(list(records), indent=2), encoding="utf-8")
