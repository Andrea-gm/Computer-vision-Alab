"""Classical vial detector: white-cap prefilter + contours/ellipse + HoughCircles.

Suitability note
----------------
Desk-top vials often present an opaque white screw cap that is nearly circular
when viewed from above — a good Hough / circularity target. Many Orbbec stills
are off-axis, so the lid projects as an **ellipse**. This pipeline therefore:

1. HSV white / low-saturation prefilter (opaque cap).
2. Contour hierarchy (external) + circularity + ellipse aspect filters.
3. ``cv2.fitEllipse`` for geometry (primary).
4. Optional ``HoughCircles`` as a secondary cue (requires white support under disk).
5. Expand the cap box downward to cover the clear cup (GT boxes are full-vial).

Pure OpenCV — no deep learning.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
    source: str = "contour"


@dataclass
class InferStats:
    backend: str
    model_path: str
    inference_ms: float
    num_dets: int


@dataclass
class HoughContourParams:
    """Tunable classical geometry parameters (also mirrored in params.yaml)."""

    # Color prefilter (HSV)
    v_min: int = 170
    s_max: int = 70
    # Contour / circularity
    min_circularity: float = 0.45
    max_circularity: float = 1.0
    min_ellipse_ar: float = 0.45  # min(axis)/max(axis)
    min_area_frac: float = 0.0015
    max_area_frac: float = 0.08
    # Cap → full-vial expansion
    cap_expand_down: float = 2.6
    cap_expand_side: float = 0.20
    # Full-vial box aspect (h/w)
    min_box_aspect: float = 0.60
    max_box_aspect: float = 2.80
    # HoughCircles
    use_hough: bool = True
    dp: float = 1.2
    min_dist_frac: float = 0.08
    canny: float = 110.0
    accum: float = 28.0
    min_radius_frac: float = 0.02
    max_radius_frac: float = 0.12
    hough_white_support: float = 0.25
    # Selection
    score_thresh: float = 0.40
    nms_iou: float = 0.35
    max_dets: int = 2


def default_camera_root() -> Path:
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / "vial_dataset").is_dir():
            return parent
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
    imgs = [p for p in sorted(root.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return imgs


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
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height
        dets.append(Det(x1, y1, x2, y2, 1.0, cls_id=cls_id, label="vial", source="gt"))
    return dets


def box_iou(a: Det, b: Det) -> float:
    ax1, ay1, ax2, ay2 = a.x1, a.y1, a.x2, a.y2
    bx1, by1, bx2, by2 = b.x1, b.y1, b.x2, b.y2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / max(ua, 1e-6))


def det_to_dict(d: Det) -> dict:
    return asdict(d)


def _circularity(cnt: np.ndarray) -> float:
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))


def _white_mask(bgr: np.ndarray, v_min: int, s_max: int) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] >= int(v_min)) & (hsv[:, :, 1] <= int(s_max))).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _expand_cap_box(
    x: int,
    y: int,
    bw: int,
    bh: int,
    *,
    width: int,
    height: int,
    expand_down: float,
    expand_side: float,
) -> Tuple[int, int, int, int]:
    pad_x = int(expand_side * bw)
    x1 = max(0, x - pad_x)
    x2 = min(width - 1, x + bw + pad_x)
    y1 = max(0, y - int(0.15 * bh))
    y2 = min(height - 1, y + int(expand_down * max(bh, bw * 0.55)))
    return x1, y1, x2, y2


def _nms(cands: List[Det], iou_thr: float, max_dets: int) -> List[Det]:
    ordered = sorted(cands, key=lambda d: d.conf, reverse=True)
    kept: List[Det] = []
    for d in ordered:
        if all(box_iou(d, k) < iou_thr for k in kept):
            kept.append(d)
        if len(kept) >= max_dets:
            break
    return kept


class VialHoughContourDetector:
    """Classical geometry detector for clear vials with white screw caps."""

    def __init__(self, params: Optional[Union[HoughContourParams, Dict]] = None) -> None:
        if params is None:
            self.params = HoughContourParams()
        elif isinstance(params, HoughContourParams):
            self.params = params
        else:
            base = HoughContourParams()
            for key, value in dict(params).items():
                if hasattr(base, key):
                    setattr(base, key, type(getattr(base, key))(value))
            self.params = base
        self.backend = "hough_contours"
        self.model_path: Optional[Path] = None

    def predict(self, bgr: np.ndarray) -> Tuple[List[Det], InferStats]:
        t0 = time.perf_counter()
        dets = self.detect(bgr)
        ms = (time.perf_counter() - t0) * 1000.0
        stats = InferStats(
            backend=self.backend,
            model_path="",
            inference_ms=ms,
            num_dets=len(dets),
        )
        return dets, stats

    def detect(self, bgr: np.ndarray) -> List[Det]:
        if bgr is None or bgr.size == 0:
            return []
        p = self.params
        height, width = bgr.shape[:2]
        area_img = float(max(1, height * width))
        mask = _white_mask(bgr, p.v_min, p.s_max)
        cands: List[Det] = []

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            frac = area / area_img
            if frac < p.min_area_frac or frac > p.max_area_frac:
                continue
            if len(cnt) < 5:
                continue
            circ = _circularity(cnt)
            if circ < p.min_circularity or circ > p.max_circularity:
                continue
            (_cx, _cy), (ma, MA), _ang = cv2.fitEllipse(cnt)
            ear = float(min(ma, MA) / max(ma, MA, 1e-6))
            if ear < p.min_ellipse_ar:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            x1, y1, x2, y2 = _expand_cap_box(
                x,
                y,
                bw,
                bh,
                width=width,
                height=height,
                expand_down=p.cap_expand_down,
                expand_side=p.cap_expand_side,
            )
            box_h = max(1, y2 - y1)
            box_w = max(1, x2 - x1)
            aspect = box_h / float(box_w)
            if aspect < p.min_box_aspect or aspect > p.max_box_aspect:
                continue
            # Prefer mid-sized caps (~0.6% of image) with high circularity×ellipse-AR.
            size_score = float(
                np.exp(-((np.log(max(frac, 1e-8)) - np.log(0.006)) ** 2) / (2 * 0.8**2))
            )
            score = float(np.clip(0.25 + 0.55 * circ * ear + 0.20 * size_score, 0.15, 0.98))
            if score < p.score_thresh:
                continue
            cands.append(
                Det(
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    score,
                    source="contour_ellipse",
                )
            )

        if p.use_hough:
            cands.extend(self._hough_candidates(bgr, mask))

        return _nms(cands, p.nms_iou, p.max_dets)

    def _hough_candidates(self, bgr: np.ndarray, mask: np.ndarray) -> List[Det]:
        p = self.params
        height, width = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 1.5)
        min_dist = max(20, int(p.min_dist_frac * min(height, width)))
        min_r = max(6, int(p.min_radius_frac * min(height, width)))
        max_r = max(min_r + 5, int(p.max_radius_frac * min(height, width)))
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=float(p.dp),
            minDist=float(min_dist),
            param1=float(p.canny),
            param2=float(p.accum),
            minRadius=int(min_r),
            maxRadius=int(max_r),
        )
        out: List[Det] = []
        if circles is None:
            return out
        yy, xx = np.ogrid[:height, :width]
        for x, y, r in np.round(circles[0]).astype(float):
            xi, yi, ri = int(x), int(y), int(r)
            if ri <= 0:
                continue
            disk = (xx - xi) ** 2 + (yy - yi) ** 2 <= (ri * ri)
            support = float(mask[disk].mean() / 255.0) if disk.any() else 0.0
            if support < p.hough_white_support:
                continue
            x1 = max(0, int(xi - 1.1 * ri))
            x2 = min(width - 1, int(xi + 1.1 * ri))
            y1 = max(0, int(yi - 1.0 * ri))
            y2 = min(height - 1, int(yi + 3.0 * ri))
            box_h = max(1, y2 - y1)
            box_w = max(1, x2 - x1)
            aspect = box_h / float(box_w)
            if aspect < p.min_box_aspect or aspect > p.max_box_aspect:
                continue
            score = float(
                np.clip(
                    0.30 + 0.40 * support + 0.15 * (ri / max(0.08 * min(height, width), 1.0)),
                    0.20,
                    0.90,
                )
            )
            if score < p.score_thresh:
                continue
            out.append(
                Det(float(x1), float(y1), float(x2), float(y2), score, source="hough")
            )
        return out

    def annotate(
        self,
        bgr: np.ndarray,
        dets: Sequence[Det],
        *,
        color: Tuple[int, int, int] = (40, 200, 80),
    ) -> np.ndarray:
        out = bgr.copy()
        for d in dets:
            p1 = (int(d.x1), int(d.y1))
            p2 = (int(d.x2), int(d.y2))
            cv2.rectangle(out, p1, p2, color, 2)
            tag = f"{d.label} {d.conf:.2f} [{d.source}]"
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


def params_from_ros(node_get_parameter) -> HoughContourParams:
    """Build params from a ROS node get_parameter callable (name -> Parameter)."""
    p = HoughContourParams()
    mapping = {
        "v_min": ("v_min", int),
        "s_max": ("s_max", int),
        "min_circularity": ("min_circularity", float),
        "max_circularity": ("max_circularity", float),
        "min_ellipse_ar": ("min_ellipse_ar", float),
        "min_area_frac": ("min_area_frac", float),
        "max_area_frac": ("max_area_frac", float),
        "cap_expand_down": ("cap_expand_down", float),
        "cap_expand_side": ("cap_expand_side", float),
        "min_box_aspect": ("min_box_aspect", float),
        "max_box_aspect": ("max_box_aspect", float),
        "use_hough": ("use_hough", bool),
        "dp": ("dp", float),
        "min_dist_frac": ("min_dist_frac", float),
        "canny": ("canny", float),
        "accum": ("accum", float),
        "min_radius_frac": ("min_radius_frac", float),
        "max_radius_frac": ("max_radius_frac", float),
        "hough_white_support": ("hough_white_support", float),
        "score_thresh": ("score_thresh", float),
        "nms_iou": ("nms_iou", float),
        "max_dets": ("max_dets", int),
    }
    for attr, (pname, caster) in mapping.items():
        try:
            param = node_get_parameter(pname)
            val = param.get_parameter_value()
            if caster is bool:
                setattr(p, attr, bool(val.bool_value))
            elif caster is int:
                # integers may arrive as integer_value or double_value
                if val.type == 2:  # INTEGER
                    setattr(p, attr, int(val.integer_value))
                else:
                    setattr(p, attr, int(val.double_value))
            else:
                if val.type == 2:
                    setattr(p, attr, float(val.integer_value))
                else:
                    setattr(p, attr, float(val.double_value))
        except Exception:  # noqa: BLE001
            continue
    return p
