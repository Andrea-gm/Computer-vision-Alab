"""Cap/base keypoint axis detector → oriented vial box.

Pipeline:
  1. Classical white-cap localization (HSV + morphology).
  2. Far-end (base) estimate by scoring clear-body strips on an angular sweep.
  3. Cap + base keypoints define the vial axis; build oriented box along it.
  4. Optional YOLO proposals seed ROI caps when classical recall dips.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .axis_skeleton import (
    det_from_keypoints,
    estimate_base,
    far_end_from_box,
    nms_dets,
    oriented_corners,
)
from .cap_finder import CapBlob, find_cap_blobs, localize_cap_in_box, white_cap_mask
from .types import Det, InferStats, box_iou, list_images
from .weak_labels import load_yolo_label, weak_from_box


DEFAULT_CAMERA = Path(r"E:\Desktop\Jessica\robotics")
DEFAULT_ASPECT = 72.69 / 54.22


@dataclass
class AxisParams:
    confidence_threshold: float = 0.35
    max_detections: int = 3
    white_s_max: int = 55
    white_v_min: int = 200
    min_cap_area: int = 35
    nms_iou: float = 0.40
    expected_aspect: float = DEFAULT_ASPECT
    use_yolo_seed: bool = True
    yolo_weights: str = ""
    yolo_conf: float = 0.15
    min_base_score: float = 0.85


class KeypointAxisDetector:
    def __init__(self, params: Optional[AxisParams] = None) -> None:
        self.params = params or AxisParams()
        self.backend = "keypoint_axis_classical"
        self._yolo = None
        self._aspect = self._load_aspect()
        if self.params.use_yolo_seed:
            self._try_load_yolo()

    def _load_aspect(self) -> float:
        dims = DEFAULT_CAMERA / "vial_3d_dims.json"
        if dims.is_file():
            try:
                data = json.loads(dims.read_text(encoding="utf-8"))
                h = float(data.get("total_height_mm", 72.69))
                d = float(data.get("lid_diameter_mm", 54.22))
                if h > 1 and d > 1:
                    return h / d
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        return float(self.params.expected_aspect)

    def _try_load_yolo(self) -> None:
        weights = self.params.yolo_weights
        if not weights:
            for cand in (
                DEFAULT_CAMERA / "runs" / "train-race" / "r3-winner" / "weights" / "best.pt",
                DEFAULT_CAMERA / "runs" / "vial" / "weights" / "best.pt",
                DEFAULT_CAMERA / "runs" / "bon-winner" / "weights" / "best.pt",
            ):
                if cand.is_file():
                    weights = str(cand)
                    break
        if not weights or not Path(weights).is_file():
            return
        try:
            from ultralytics import YOLO  # type: ignore

            self._yolo = YOLO(weights)
            self.backend = "keypoint_axis+yolo_seed"
        except Exception:  # noqa: BLE001
            self._yolo = None

    def _yolo_boxes(self, bgr: np.ndarray) -> List[Tuple[float, float, float, float]]:
        if self._yolo is None:
            return []
        try:
            res = self._yolo.predict(
                bgr, conf=float(self.params.yolo_conf), verbose=False
            )
        except Exception:  # noqa: BLE001
            return []
        boxes: List[Tuple[float, float, float, float]] = []
        if not res:
            return boxes
        r0 = res[0]
        if r0.boxes is None:
            return boxes
        xyxy = r0.boxes.xyxy.cpu().numpy()
        for row in xyxy:
            boxes.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
        return boxes

    def _caps_from_seeds(
        self, bgr: np.ndarray, seeds: Sequence[Tuple[float, float, float, float]]
    ) -> List[CapBlob]:
        caps: List[CapBlob] = []
        for box in seeds:
            c = localize_cap_in_box(
                bgr, box, s_max=self.params.white_s_max, v_min=self.params.white_v_min
            )
            if c is not None:
                caps.append(c)
        return caps

    def predict(
        self,
        bgr: np.ndarray,
        *,
        depth_mm: Optional[np.ndarray] = None,
    ) -> Tuple[List[Det], InferStats]:
        t0 = time.perf_counter()
        if bgr is None or getattr(bgr, "size", 0) == 0:
            return [], InferStats(inference_ms=0.0)
        image = np.asarray(bgr)
        h, w = image.shape[:2]
        mask = white_cap_mask(
            image, s_max=self.params.white_s_max, v_min=self.params.white_v_min
        )
        caps = find_cap_blobs(
            image,
            mask=mask,
            s_max=self.params.white_s_max,
            v_min=self.params.white_v_min,
            min_area=self.params.min_cap_area,
        )
        seeds = self._yolo_boxes(image)
        if seeds:
            caps.extend(self._caps_from_seeds(image, seeds))

        # Dedup caps by proximity.
        caps = _dedup_caps(caps, min_dist=12.0)

        dets: List[Det] = []
        for cap in caps[:12]:
            if _looks_like_wall(image, cap):
                continue
            est = estimate_base(
                image,
                cap,
                depth_mm=depth_mm,
                expected_aspect=self._aspect,
                min_score=float(self.params.min_base_score),
            )
            if est is None:
                # Fallback: if YOLO seed box contains this cap, use far end of that box.
                fallback = self._fallback_from_seeds(cap, seeds, image.shape[:2])
                if fallback is None:
                    continue
                base_xy, score = fallback
            else:
                base_xy, score, _axis = est
            conf = float(np.clip(0.15 + 0.50 * score + 0.20 * min(cap.score, 1.0), 0.0, 0.98))
            if score < self.params.min_base_score and conf < self.params.confidence_threshold:
                continue
            if conf < self.params.confidence_threshold:
                continue
            diam = float(cap.w if cap.kind == "side_on" else max(cap.w, cap.h))
            det = det_from_keypoints(
                (cap.cx, cap.cy),
                base_xy,
                max(diam, 8.0),
                conf=conf,
                image_shape=(h, w),
                source=self.backend,
            )
            dets.append(det)

        # YOLO AABB → white-cap + far-end weak keypoints (domain transfer to real photos).
        seed_dets: List[Det] = []
        for box in seeds:
            seed_det = Det(x1=box[0], y1=box[1], x2=box[2], y2=box[3], conf=1.0)
            weak = weak_from_box(image, seed_det)
            if weak is None:
                continue
            seed_dets.append(
                det_from_keypoints(
                    weak.cap,
                    weak.base,
                    max(weak.diameter, 8.0),
                    conf=0.78,
                    image_shape=(h, w),
                    source="yolo_weak_axis",
                )
            )
        if seed_dets:
            # Prefer oriented YOLO-weak boxes; keep classical only if it overlaps a seed.
            kept_classical: List[Det] = []
            for d in dets:
                if any(box_iou(d, s) >= 0.15 for s in seed_dets):
                    # Classical axis often better than AABB far-end — keep higher-conf.
                    kept_classical.append(d)
            dets = seed_dets + kept_classical
        elif depth_mm is None:
            # RGB-only classical over-fires on desk clutter without depth holes.
            dets = [d for d in dets if d.conf >= max(0.55, self.params.confidence_threshold)]

        dets = nms_dets(dets, self.params.nms_iou)
        dets = [d for d in dets if d.conf >= self.params.confidence_threshold]
        dets = dets[: self.params.max_detections]
        ms = (time.perf_counter() - t0) * 1000.0
        stats = InferStats(
            backend=self.backend,
            inference_ms=ms,
            n_caps=len(caps),
            n_kept=len(dets),
        )
        return dets, stats

    def _fallback_from_seeds(
        self,
        cap: CapBlob,
        seeds: Sequence[Tuple[float, float, float, float]],
        shape: Tuple[int, int],
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        for box in seeds:
            x1, y1, x2, y2 = box
            if x1 <= cap.cx <= x2 and y1 <= cap.cy <= y2:
                base = far_end_from_box((cap.cx, cap.cy), box)
                return base, 0.40
        return None

    def annotate(self, bgr: np.ndarray, dets: Sequence[Det]) -> np.ndarray:
        out = bgr.copy()
        for d in dets:
            corners = oriented_corners(d.cx, d.cy, d.length, d.diameter, d.angle_deg)
            pts = np.rint(corners).astype(np.int32)
            cv2.polylines(out, [pts], True, (40, 220, 80), 2, cv2.LINE_AA)
            cv2.rectangle(
                out,
                (int(d.x1), int(d.y1)),
                (int(d.x2), int(d.y2)),
                (80, 180, 255),
                1,
                cv2.LINE_AA,
            )
            if d.cap is not None:
                cv2.circle(out, (int(d.cap.x), int(d.cap.y)), 5, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    out,
                    "cap",
                    (int(d.cap.x) + 6, int(d.cap.y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            if d.base is not None:
                cv2.circle(
                    out, (int(d.base.x), int(d.base.y)), 5, (255, 128, 0), -1, cv2.LINE_AA
                )
                cv2.putText(
                    out,
                    "base",
                    (int(d.base.x) + 6, int(d.base.y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 128, 0),
                    1,
                    cv2.LINE_AA,
                )
            if d.cap is not None and d.base is not None:
                cv2.line(
                    out,
                    (int(d.cap.x), int(d.cap.y)),
                    (int(d.base.x), int(d.base.y)),
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                out,
                f"{d.pose} {d.angle_deg:.0f}deg {d.conf:.2f}",
                (int(d.x1), max(14, int(d.y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
        return out


def _dedup_caps(caps: Sequence[CapBlob], *, min_dist: float) -> List[CapBlob]:
    kept: List[CapBlob] = []
    for c in sorted(caps, key=lambda x: x.score, reverse=True):
        if all((c.cx - k.cx) ** 2 + (c.cy - k.cy) ** 2 >= min_dist**2 for k in kept):
            kept.append(c)
    return kept


def _looks_like_wall(bgr: np.ndarray, cap: CapBlob) -> bool:
    """Reject large bright slabs whose neighborhood is also white (empty / clutter)."""
    h, w = bgr.shape[:2]
    if cap.w * cap.h > 0.12 * w * h:
        return True
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    pad = max(6, int(max(cap.w, cap.h) * 0.55))
    x0 = max(0, cap.x - pad)
    y0 = max(0, cap.y - pad)
    x1 = min(w, cap.x + cap.w + pad)
    y1 = min(h, cap.y + cap.h + pad)
    ring = np.ones((y1 - y0, x1 - x0), dtype=bool)
    ix0, iy0 = cap.x - x0, cap.y - y0
    ring[iy0 : iy0 + cap.h, ix0 : ix0 + cap.w] = False
    if not ring.any():
        return False
    neigh_white = ((val[y0:y1, x0:x1] >= 200) & (sat[y0:y1, x0:x1] <= 55))[ring]
    return float(neigh_white.mean()) > 0.55 if neigh_white.size else False


def load_yolo_label_file(path: Path, width: int, height: int) -> List[Det]:
    return load_yolo_label(path, width, height)


__all__ = [
    "AxisParams",
    "KeypointAxisDetector",
    "DEFAULT_CAMERA",
    "list_images",
    "load_yolo_label_file",
]
