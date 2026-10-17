"""HSV/LAB white-cap segmentation + morphology + CC / contour vial detector.

Calibrated on ``vial_dataset/split_70_10_20`` train samples:
  - Cap is near-achromatic white (low S, high V / high L).
  - Clear cup has no stable hue — body recovery uses Canny edges under the
    cap plus upright aspect-ratio priors (still classical color-seg family).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import time

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Det:
    """Axis-aligned detection in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int = 0
    label: str = "vial"
    source: str = "hsv_morph"


@dataclass
class InferStats:
    backend: str
    inference_ms: float
    num_dets: int
    mask_pixels: int = 0


@dataclass
class HsvMorphParams:
    """Tunable ranges — mirrored by config/params.yaml."""

    hsv_h_min: int = 0
    hsv_h_max: int = 179
    hsv_s_min: int = 0
    hsv_s_max: int = 40
    hsv_v_min: int = 195
    hsv_v_max: int = 255
    lab_l_min: int = 190
    lab_l_max: int = 255
    lab_a_center: int = 128
    lab_a_tol: int = 12
    lab_b_center: int = 128
    lab_b_tol: int = 12
    morph_open_ksize: int = 5
    morph_close_ksize: int = 5
    use_edge_assist: bool = True
    canny_low: int = 50
    canny_high: int = 120
    body_expand_cap_h_mult: float = 2.6
    body_expand_cap_w_mult: float = 1.35
    min_area_frac: float = 0.0006
    max_area_frac: float = 0.08
    min_box_aspect: float = 0.75
    max_box_aspect: float = 2.6
    max_box_area_frac: float = 0.28
    min_cap_fill: float = 0.25
    min_score: float = 0.42
    max_dets: int = 3
    nms_iou: float = 0.35


def default_camera_root() -> Path:
    """Locate Camera/ (contains vial_dataset) above this package.

    Package lives at Camera/ros2_ws/src/agent_04_hsv_morph/ — walk up until markers hit.
    """
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / "vial_dataset").is_dir():
            return parent
    # Fallback: .../agent_04_hsv_morph/agent_04_hsv_morph → parents[4]=Camera
    for parent in here.parents:
        if parent.name == "ros2_ws":
            sibling = parent.parent / "robotics"
            if sibling.is_dir():
                return sibling
            break
    return here.parents[4]


def list_images(folder: Union[str, Path]) -> List[Path]:
    root = Path(folder)
    if not root.is_dir():
        return []
    out: List[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return out


def load_yolo_label(path: Path, width: int, height: int) -> List[Det]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    dets: List[Det] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        dets.append(
            Det(x1, y1, x2, y2, conf=1.0, cls_id=cls_id, label="vial", source="gt")
        )
    return dets


def box_iou(a: Det, b: Det) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    bb = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = aa + bb - inter
    return float(inter / union) if union > 0 else 0.0


def det_to_dict(d: Det) -> Dict:
    return asdict(d)


def annotate(bgr: np.ndarray, dets: Sequence[Det], *, color=(40, 200, 80)) -> np.ndarray:
    out = bgr.copy()
    for d in dets:
        p1 = (int(d.x1), int(d.y1))
        p2 = (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, color, 2)
        tag = f"{d.label} {d.conf:.2f}"
        cv2.putText(
            out,
            tag,
            (p1[0], max(16, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def _odd_ksize(k: int) -> int:
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1


def white_cap_mask(bgr: np.ndarray, p: HsvMorphParams) -> np.ndarray:
    """Union of HSV low-S/high-V and LAB high-L near-neutral masks."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    h, s, v = cv2.split(hsv)
    L, a, bch = cv2.split(lab)

    hsv_ok = (
        (h >= p.hsv_h_min)
        & (h <= p.hsv_h_max)
        & (s >= p.hsv_s_min)
        & (s <= p.hsv_s_max)
        & (v >= p.hsv_v_min)
        & (v <= p.hsv_v_max)
    )
    a16 = a.astype(np.int16)
    b16 = bch.astype(np.int16)
    lab_ok = (
        (L >= p.lab_l_min)
        & (L <= p.lab_l_max)
        & (np.abs(a16 - p.lab_a_center) <= p.lab_a_tol)
        & (np.abs(b16 - p.lab_b_center) <= p.lab_b_tol)
    )
    mask = (hsv_ok | lab_ok).astype(np.uint8) * 255

    open_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd_ksize(p.morph_open_ksize),) * 2
    )
    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd_ksize(p.morph_close_ksize),) * 2
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    return mask


def _nms(scored: List[tuple], iou_thr: float, max_dets: int) -> List[Det]:
    scored = sorted(scored, key=lambda t: t[0], reverse=True)
    kept: List[Det] = []
    for score, det in scored:
        if all(box_iou(det, k) < iou_thr for k in kept):
            kept.append(det)
        if len(kept) >= max_dets:
            break
    return kept


class VialHsvMorphDetector:
    """Classical HSV/LAB + morphology vial instance detector."""

    def __init__(self, params: Optional[HsvMorphParams] = None) -> None:
        self.params = params or HsvMorphParams()
        self.backend = "hsv_morph"

    @classmethod
    def from_mapping(cls, mapping: Dict) -> "VialHsvMorphDetector":
        fields = {f.name for f in HsvMorphParams.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: mapping[k] for k in mapping if k in fields}
        return cls(HsvMorphParams(**kwargs))

    def predict(self, bgr: np.ndarray) -> tuple[List[Det], InferStats]:
        t0 = time.perf_counter()
        dets, mask_px = self._detect(bgr)
        ms = (time.perf_counter() - t0) * 1000.0
        stats = InferStats(
            backend=self.backend,
            inference_ms=ms,
            num_dets=len(dets),
            mask_pixels=mask_px,
        )
        return dets, stats

    def annotate(self, bgr: np.ndarray, dets: Sequence[Det]) -> np.ndarray:
        return annotate(bgr, dets)

    def _detect(self, bgr: np.ndarray) -> tuple[List[Det], int]:
        p = self.params
        if bgr is None or bgr.size == 0:
            return [], 0
        h, w = bgr.shape[:2]
        area_img = float(h * w)
        mask = white_cap_mask(bgr, p)
        mask_px = int((mask > 0).sum())

        edges = None
        if p.use_edge_assist:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(
                cv2.GaussianBlur(gray, (3, 3), 0), int(p.canny_low), int(p.canny_high)
            )

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        n_labels, _labels, stats, _cents = cv2.connectedComponentsWithStats(mask, 8)
        scored: List[tuple] = []

        for i in range(1, n_labels):
            x, y, ww, hh, area = stats[i]
            frac = float(area) / area_img
            if frac < p.min_area_frac or frac > p.max_area_frac:
                continue
            if ww < 10 or hh < 8:
                continue
            # Reject large wall / table white strips.
            if ww > 0.35 * w or hh > 0.45 * h:
                continue
            ar_cap = hh / max(1.0, float(ww))
            if ar_cap < 0.25 or ar_cap > 2.2:
                continue

            roi = mask[y : y + hh, x : x + ww]
            fill = float((roi > 0).sum()) / max(1.0, float(ww * hh))
            if fill < p.min_cap_fill:
                continue

            cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            peri = cv2.arcLength(cnt, True)
            circ = float(4.0 * np.pi * cv2.contourArea(cnt) / (peri * peri + 1e-6))
            if circ < 0.25 and ar_cap < 0.5:
                continue

            x1 = max(0, x - int(0.12 * ww))
            x2 = min(w - 1, x + ww + int(0.12 * ww))
            y1 = max(0, y - int(0.10 * hh))
            search_h = int(max(hh * p.body_expand_cap_h_mult, ww * p.body_expand_cap_w_mult))
            y_end = min(h - 1, y + hh + search_h)

            if edges is not None and y_end > y + hh:
                band = edges[y + hh : y_end, x1:x2]
                if band.size:
                    row_sum = (band > 0).sum(axis=1)
                    thr = max(2, int(0.12 * (x2 - x1)))
                    good = np.where(row_sum >= thr)[0]
                    if len(good):
                        y2 = y + hh + int(good[-1]) + 2
                    else:
                        y2 = min(h - 1, y + int(hh + max(hh * 1.8, ww * 1.0)))
                else:
                    y2 = min(h - 1, y + int(hh + max(hh * 1.8, ww * 1.0)))
            else:
                y2 = min(h - 1, y + int(hh + max(hh * 1.8, ww * 1.0)))

            box_h = max(1, y2 - y1)
            box_w = max(1, x2 - x1)
            ar = box_h / float(box_w)
            if ar < p.min_box_aspect or ar > p.max_box_aspect:
                continue
            if (box_w * box_h) / area_img > p.max_box_area_frac:
                continue

            # Dataset median vial area frac ~0.027 on mid-field desk shots.
            size_score = 1.0 - min(1.0, abs((box_w * box_h) / area_img - 0.03) / 0.08)
            bright = float(hsv[y : y + max(1, hh), x : x + ww, 2].mean() / 255.0)
            score = float(
                np.clip(
                    0.2
                    + 0.3 * bright
                    + 0.2 * circ
                    + 0.2 * size_score
                    + 0.15 * (1.0 - abs(ar - 1.2) / 1.2),
                    0.1,
                    0.95,
                )
            )
            if score < p.min_score:
                continue
            scored.append(
                (
                    score,
                    Det(
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                        conf=score,
                        source="hsv_morph",
                    ),
                )
            )

        return _nms(scored, p.nms_iou, p.max_dets), mask_px
