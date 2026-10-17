"""Cap→body geometric cascade — orientation-agnostic vial localization.

Pipeline
--------
1. White-cap mask (HSV) → compact blob seeds (reject table glare).
2. Body cue (edges; optional depth holes) kept only where a component
   **touches** a dilated cap (dual-blob link).
3. Major-axis from dual-blob / PCA / cardinal probes on the local combined
   mask. Never assume "expand downward" only.
4. AABB from geometric prior (length/diam × cap size) along that axis.
5. RGB-only stills: fuse YOLO proposals (upright dataset rescue) with
   classical candidates via NMS. Depth/sideways path stays classical.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .types import Det, box_iou
from .yolo_proposals import YoloProposer

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

_DEFAULT_LENGTH_MM = 72.69
_DEFAULT_DIAM_MM = 54.22
_DIMS_PATH = Path(r"E:\Desktop\Jessica\robotics\vial_3d_dims.json")


@dataclass
class CascadeParams:
    confidence_threshold: float = 0.35
    max_detections: int = 2
    white_s_max: int = 55
    white_v_min: int = 185
    cap_min_area: int = 35
    cap_max_area: int = 4500
    cap_max_area_frac: float = 0.04
    cap_max_side_frac: float = 0.22
    body_v_min: int = 35
    body_v_max: int = 195
    body_s_max: int = 100
    edge_thresh: float = 16.0
    link_search_scale: float = 3.2  # max radius = scale * cap_diam
    length_over_diam: float = _DEFAULT_LENGTH_MM / _DEFAULT_DIAM_MM
    length_scale: float = 1.05
    diam_scale: float = 1.15
    min_axis_aspect: float = 1.08
    nms_iou: float = 0.45
    use_depth_holes: bool = True
    depth_hole_mm: float = 80.0
    max_box_area_frac: float = 0.22
    # YOLO RGB proposal stage (used when depth is absent).
    use_yolo_rgb: bool = True
    yolo_conf: float = 0.25
    yolo_model_path: str = ""
    yolo_boost: float = 0.12  # added to YOLO conf so it outranks glare FPs


@dataclass
class InferStats:
    backend: str = "cap_body_cascade_pca"
    inference_ms: float = 0.0
    n_caps: int = 0
    n_linked: int = 0
    n_kept: int = 0
    axis_methods: Dict[str, int] = field(default_factory=dict)
    extras: Dict = field(default_factory=dict)


def list_images(folder: Path) -> List[Path]:
    if not folder or not Path(folder).exists():
        return []
    folder = Path(folder)
    return sorted(
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def load_yolo_label(path: Path, width: int, height: int) -> List[Det]:
    if not path.exists():
        return []
    dets: List[Det] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        dets.append(Det(x1, y1, x2, y2, conf=1.0, cls_id=cls_id, label="vial", source="gt"))
    return dets


def _load_aspect() -> float:
    if _DIMS_PATH.is_file():
        try:
            data = json.loads(_DIMS_PATH.read_text(encoding="utf-8"))
            h = float(data.get("total_height_mm", _DEFAULT_LENGTH_MM))
            d = float(data.get("lid_diameter_mm", _DEFAULT_DIAM_MM))
            if h > 1 and d > 1:
                return h / d
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return _DEFAULT_LENGTH_MM / _DEFAULT_DIAM_MM


class CapBodyCascadeDetector:
    """Orientation-agnostic white-cap → transparent-body cascade."""

    def __init__(self, params: Optional[CascadeParams] = None) -> None:
        self.params = params or CascadeParams()
        if abs(self.params.length_over_diam - (_DEFAULT_LENGTH_MM / _DEFAULT_DIAM_MM)) < 1e-6:
            self.params.length_over_diam = _load_aspect()
        self.backend = "cap_body_cascade_pca"
        self._yolo: Optional[YoloProposer] = None
        if self.params.use_yolo_rgb:
            self._yolo = YoloProposer(
                self.params.yolo_model_path or None,
                conf=self.params.yolo_conf,
            )
            if self._yolo.available:
                self.backend = "cascade+yolo_rgb"

    def update_params(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.params, k):
                setattr(self.params, k, v)
        if "use_yolo_rgb" in kwargs or "yolo_conf" in kwargs or "yolo_model_path" in kwargs:
            if self.params.use_yolo_rgb:
                self._yolo = YoloProposer(
                    self.params.yolo_model_path or None,
                    conf=self.params.yolo_conf,
                )
                self.backend = (
                    "cascade+yolo_rgb" if self._yolo.available else "cap_body_cascade_pca"
                )
            else:
                self._yolo = None
                self.backend = "cap_body_cascade_pca"

    def white_cap_mask(self, bgr: np.ndarray) -> np.ndarray:
        p = self.params
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat, val = hsv[:, :, 1], hsv[:, :, 2]
        best = ((val >= int(p.white_v_min)) & (sat <= int(p.white_s_max))).astype(np.uint8) * 255
        best_frac = float(best.mean() / 255.0)
        if not (0.0003 <= best_frac <= 0.12):
            for vmin, smax in (
                (200, 50),
                (190, 55),
                (175, 60),
                (210, 45),
                (220, 40),
            ):
                trial = ((val >= vmin) & (sat <= smax)).astype(np.uint8) * 255
                frac = float(trial.mean() / 255.0)
                if 0.0003 <= frac <= 0.12 and abs(frac - 0.02) < abs(best_frac - 0.02):
                    best, best_frac = trial, frac
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        best = cv2.morphologyEx(best, cv2.MORPH_OPEN, k)
        best = cv2.morphologyEx(best, cv2.MORPH_CLOSE, k)
        return best

    def body_cue_mask(
        self, bgr: np.ndarray, depth_mm: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Body silhouette cue. Depth holes win when present (clear plastic)."""
        p = self.params
        h, w = bgr.shape[:2]
        if depth_mm is not None and p.use_depth_holes:
            d = np.asarray(depth_mm, dtype=np.float32)
            if d.shape[:2] != (h, w):
                d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
            holes = (d <= p.depth_hole_mm).astype(np.uint8) * 255
            hole_frac = float(holes.mean() / 255.0)
            # Synth / live clear cups leave compact depth failures — use them alone.
            if 0.0002 <= hole_frac <= 0.08:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                holes = cv2.morphologyEx(holes, cv2.MORPH_CLOSE, k)
                return holes

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat, val = hsv[:, :, 1], hsv[:, :, 2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        # Sparse rim-only body (avoid flooding mid-tone benches).
        rim = (mag >= float(p.edge_thresh * 1.35)) & (sat <= p.body_s_max) & (val < 220)
        body = rim.astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, k)
        return body

    def _cap_seeds(self, white: np.ndarray) -> List[Dict]:
        p = self.params
        h, w = white.shape[:2]
        img_area = float(h * w)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(white, 8)
        seeds: List[Dict] = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < p.cap_min_area or area > p.cap_max_area:
                continue
            if area > p.cap_max_area_frac * img_area:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            if bw < 5 or bh < 5:
                continue
            if max(bw, bh) > p.cap_max_side_frac * max(h, w):
                continue
            fill = area / max(float(bw * bh), 1.0)
            if fill < 0.40:
                continue
            # Reject very elongated bright streaks (not a screw cap).
            ar = max(bw, bh) / float(min(bw, bh) + 1e-6)
            if ar > 3.5:
                continue
            mask_i = (labels == i).astype(np.uint8) * 255
            cx, cy = float(cents[i, 0]), float(cents[i, 1])
            # Side-on end-cap: short×long strip → vial diameter ≈ longer side.
            # Near-round top: sides similar → max side is fine.
            diam = float(max(bw, bh))
            seeds.append(
                {
                    "box": (x, y, bw, bh),
                    "cx": cx,
                    "cy": cy,
                    "diam": diam,
                    "area": float(area),
                    "mask": mask_i,
                    "fill": fill,
                    "ar": ar,
                    "bw": float(bw),
                    "bh": float(bh),
                }
            )
        seeds.sort(key=lambda s: s["area"], reverse=True)
        return seeds[:16]

    def _pca_axis(
        self, ys: np.ndarray, xs: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        if ys.size < 10:
            return None
        pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = np.cov(centered, rowvar=False)
        if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
            return None
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        if eigvals[0] <= 1e-6:
            return None
        aspect = float(eigvals[0] / max(eigvals[1], 1e-6))
        major = eigvecs[:, 0]
        nrm = float(np.linalg.norm(major))
        if nrm < 1e-9:
            return None
        return mean, major / nrm, aspect

    def _ellipse_axis(
        self, mask: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        if len(cnt) < 5:
            return None
        try:
            (cx, cy), (ma, mi), angle = cv2.fitEllipse(cnt)
        except cv2.error:
            return None
        if ma < 1 or mi < 1:
            return None
        if ma < mi:
            ma, mi = mi, ma
            angle = float(angle) + 90.0
        rad = np.deg2rad(float(angle))
        major = np.array([np.cos(rad), np.sin(rad)], dtype=np.float64)
        return np.array([cx, cy], dtype=np.float64), major, float(ma / max(mi, 1e-6))

    def _minarearect_axis(
        self, mask: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 16:
            return None
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(cnt)
        if rw < 1 or rh < 1:
            return None
        if rw >= rh:
            rad = np.deg2rad(float(angle))
            aspect = rw / max(rh, 1e-6)
        else:
            rad = np.deg2rad(float(angle) + 90.0)
            aspect = rh / max(rw, 1e-6)
        major = np.array([np.cos(rad), np.sin(rad)], dtype=np.float64)
        return np.array([cx, cy], dtype=np.float64), major, float(aspect)

    def _link_body_touching(self, body: np.ndarray, cap: Dict) -> np.ndarray:
        """Keep only body CC that touch a dilated cap, inside a local radius."""
        p = self.params
        h, w = body.shape[:2]
        cx, cy, diam = cap["cx"], cap["cy"], max(cap["diam"], 8.0)
        radius = min(max(diam * p.link_search_scale, 28.0), 0.45 * max(h, w))

        dil_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(3, int(diam * 0.35) | 1),) * 2
        )
        cap_dil = cv2.dilate(cap["mask"], dil_k, iterations=2)

        # Restrict body to circular neighborhood around cap.
        yy, xx = np.ogrid[:h, :w]
        roi = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2
        local = np.zeros_like(body)
        local[roi] = body[roi]

        # Morphological reconstruction-ish: components intersecting cap_dil.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (local > 0).astype(np.uint8) * 255, 8
        )
        linked = np.zeros_like(body)
        expect_area = diam * diam * p.length_over_diam
        touch_ids = set(int(v) for v in np.unique(labels[cap_dil > 0]) if v > 0)
        for i in touch_ids:
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 20:
                continue
            # Reject components that are huge / tiny relative to expected vial.
            if area > expect_area * 5.0 or area < expect_area * 0.08:
                continue
            cw = int(stats[i, cv2.CC_STAT_WIDTH])
            ch = int(stats[i, cv2.CC_STAT_HEIGHT])
            if max(cw, ch) > diam * p.length_over_diam * 2.8:
                continue
            linked[labels == i] = 255

        # If nothing touched, take nearest compact component within radius.
        if not linked.any() and n > 1:
            best_i, best_d = -1, 1e9
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < 20 or area > expect_area * 5.0:
                    continue
                bx = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] * 0.5
                by = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] * 0.5
                d = float(np.hypot(bx - cx, by - cy))
                if d < best_d:
                    best_d, best_i = d, i
            if best_i > 0 and best_d <= radius:
                linked[labels == best_i] = 255
        return linked

    def _probe_axis_from_body(
        self, combined: np.ndarray, cap: Dict
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, str]]:
        """Dual-blob: vial axis = cap→body vector (works any orientation)."""
        body_only = combined.copy()
        body_only[cap["mask"] > 0] = 0
        bys, bxs = np.where(body_only > 0)
        if bys.size < 8:
            return None
        bmean = np.array([bxs.mean(), bys.mean()], dtype=np.float64)
        cap_c = np.array([cap["cx"], cap["cy"]], dtype=np.float64)
        vec = bmean - cap_c
        nrm = float(np.linalg.norm(vec))
        if nrm < 3.0:
            return None
        return cap_c, vec / nrm, 1.6, "dual_blob_link"

    def _estimate_axis(
        self, combined: np.ndarray, cap: Dict
    ) -> Tuple[np.ndarray, np.ndarray, float, str]:
        # Prefer dual-blob link when body mass is offset from the cap — this is
        # orientation-agnostic and avoids tall end-cap PCA locking to vertical.
        link = self._probe_axis_from_body(combined, cap)
        if link is not None:
            return link

        ys, xs = np.where(combined > 0)
        pca = self._pca_axis(ys, xs)
        if pca is not None and pca[2] >= self.params.min_axis_aspect:
            # If PCA major aligns with the *long* side of a thin end-cap, flip
            # to the perpendicular (vial length is usually orthogonal to cap strip).
            major = pca[1]
            bw, bh = cap["bw"], cap["bh"]
            if max(bw, bh) / max(min(bw, bh), 1.0) >= 1.8:
                cap_long = np.array([1.0, 0.0]) if bw >= bh else np.array([0.0, 1.0])
                if abs(float(np.dot(major, cap_long))) > 0.75:
                    major = np.array([-major[1], major[0]])
                    return pca[0], major, pca[2], "pca_perp_cap"
            return pca[0], major, pca[2], "pca_combined"

        ell = self._ellipse_axis(combined)
        if ell is not None and ell[2] >= 1.05:
            return ell[0], ell[1], ell[2], "ellipse_combined"

        mar = self._minarearect_axis(combined)
        if mar is not None and mar[2] >= 1.05:
            return mar[0], mar[1], mar[2], "minAreaRect_combined"

        # Cap strip: vial axis ≈ perpendicular to longer cap side.
        bw, bh = cap["bw"], cap["bh"]
        if bw >= bh:
            major = np.array([0.0, 1.0])  # cap wide → try vertical vial
        else:
            major = np.array([1.0, 0.0])  # cap tall → try horizontal vial
        return np.array([cap["cx"], cap["cy"]]), major, 1.0, "cap_perp_axis"

    def _oriented_aabb(
        self,
        cap: Dict,
        major: np.ndarray,
        h: int,
        w: int,
        combined: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Geometric-prior box along major axis; body only chooses side + mild trim."""
        p = self.params
        diam = max(float(cap["diam"]), 10.0)
        half_minor = 0.5 * diam * p.diam_scale
        half_major = 0.5 * diam * p.length_over_diam * p.length_scale
        # Cap often spans the short axis; for side-on end-caps, diam ~ height of cap box.
        x, y, bw, bh = cap["box"]
        # Refine diam from the dimension perpendicular to major if possible.
        minor = np.array([-major[1], major[0]], dtype=np.float64)
        # Project cap corners onto minor → better short-axis width.
        cap_pts = np.array(
            [[x, y], [x + bw, y], [x, y + bh], [x + bw, y + bh]], dtype=np.float64
        )
        cap_c = np.array([cap["cx"], cap["cy"]], dtype=np.float64)
        n_proj = (cap_pts - cap_c) @ minor
        half_minor = max(half_minor, 0.55 * (n_proj.max() - n_proj.min()))
        half_major = max(half_major, half_minor * p.length_over_diam * 0.95)

        body_only = combined.copy()
        body_only[cap["mask"] > 0] = 0
        bys, bxs = np.where(body_only > 0)
        if bys.size >= 5:
            bmean = np.array([bxs.mean(), bys.mean()], dtype=np.float64)
            side = float(np.dot(bmean - cap_c, major))
            # Cap at one end: center shifted toward body by ~0.5*(L - lid).
            offset = np.sign(side) * half_major * 0.50 if abs(side) > 1.0 else 0.0
        else:
            offset = 0.0
        center = cap_c + major * offset

        # Geometric corners (prior).
        t0, t1 = -half_major, half_major
        n0, n1 = -half_minor, half_minor

        # Mild tighten using observed combined pixels projected on axes —
        # only shrink, and never grow beyond 1.25× prior.
        ys, xs = np.where(combined > 0)
        if ys.size >= 8:
            pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
            rel = pts - center
            t = rel @ major
            n = rel @ minor
            ot0, ot1 = float(t.min()), float(t.max())
            on0, on1 = float(n.min()), float(n.max())
            # Grow only up to 1.25× prior if body extends further.
            t0 = min(t0, max(ot0, -half_major * 1.25))
            t1 = max(t1, min(ot1, half_major * 1.25))
            n0 = min(n0, max(on0, -half_minor * 1.25))
            n1 = max(n1, min(on1, half_minor * 1.25))
            # Cap must remain inside along major.
            cap_t = float(np.dot(cap_c - center, major))
            t0 = min(t0, cap_t - half_minor * 0.2)
            t1 = max(t1, cap_t + half_minor * 0.2)

        corners = np.array(
            [
                center + major * t0 + minor * n0,
                center + major * t0 + minor * n1,
                center + major * t1 + minor * n0,
                center + major * t1 + minor * n1,
            ]
        )
        x1 = float(np.clip(corners[:, 0].min(), 0, w - 1))
        y1 = float(np.clip(corners[:, 1].min(), 0, h - 1))
        x2 = float(np.clip(corners[:, 0].max(), 0, w - 1))
        y2 = float(np.clip(corners[:, 1].max(), 0, h - 1))
        return x1, y1, x2, y2

    def _white_end_score(
        self, white: np.ndarray, cap: Dict, box: Tuple[float, float, float, float]
    ) -> float:
        """Fraction of box-end white that belongs to the cap (RGB upright cue)."""
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        h, w = white.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        bw, bh = x2 - x1, y2 - y1
        cx, cy = cap["cx"], cap["cy"]
        # Cap-end strip = 30% of longer side nearest the cap.
        if bw >= bh:
            t = (cx - x1) / max(bw, 1)
            if t < 0.5:
                strip = white[y1:y2, x1 : x1 + max(1, int(0.35 * bw))]
            else:
                strip = white[y1:y2, x2 - max(1, int(0.35 * bw)) : x2]
        else:
            t = (cy - y1) / max(bh, 1)
            if t < 0.5:
                strip = white[y1 : y1 + max(1, int(0.35 * bh)), x1:x2]
            else:
                strip = white[y2 - max(1, int(0.35 * bh)) : y2, x1:x2]
        if strip.size == 0:
            return 0.0
        # Cap mask overlap with strip + white density.
        x, y, cw, ch = cap["box"]
        cap_in = white[y : y + ch, x : x + cw]
        cap_fill = float(cap_in.mean() / 255.0) if cap_in.size else 0.0
        strip_fill = float(strip.mean() / 255.0)
        return float(np.clip(0.55 * strip_fill + 0.45 * cap_fill, 0.0, 1.0))

    def _score(
        self,
        cap: Dict,
        combined: np.ndarray,
        aspect: float,
        axis_method: str,
        box: Tuple[float, float, float, float],
        h: int,
        w: int,
        linked: bool,
        *,
        has_depth: bool = True,
        white: Optional[np.ndarray] = None,
        gray: Optional[np.ndarray] = None,
        major: Optional[np.ndarray] = None,
    ) -> float:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw < 12 or bh < 10:
            return 0.0
        area_frac = (bw * bh) / float(h * w)
        if area_frac > self.params.max_box_area_frac:
            return 0.0
        cx, cy = cap["cx"], cap["cy"]
        if not (x1 - 2 <= cx <= x2 + 2 and y1 - 2 <= cy <= y2 + 2):
            return 0.0
        if bw >= bh:
            t = (cx - x1) / max(bw, 1e-6)
            hw = bh / max(bw, 1e-6)
        else:
            t = (cy - y1) / max(bh, 1e-6)
            hw = bw / max(bh, 1e-6)
        end_score = max(0.0, 1.0 - min(t, 1.0 - t) * 2.2)
        aspect_ok = 0.25 <= hw <= 0.95
        if not aspect_ok:
            return 0.12

        # --- Depth / sideways path: trust dual-blob + holes ---
        if has_depth:
            method_bonus = {
                "dual_blob_link": 0.16,
                "pca_combined": 0.12,
                "pca_perp_cap": 0.14,
                "ellipse_combined": 0.10,
                "minAreaRect_combined": 0.08,
                "cap_perp_axis": 0.06,
                "probe_down": 0.04,
                "probe_up": 0.04,
                "probe_left": 0.04,
                "probe_right": 0.04,
            }.get(axis_method, 0.0)
            if not linked:
                if axis_method == "topdown_cap":
                    return float(np.clip(0.48 + 0.30 * cap["fill"], 0.0, 0.90))
                return 0.22 if axis_method.startswith("probe") else 0.18
            fill = float((combined > 0).sum()) / max(bw * bh, 1.0)
            sc = (
                0.32
                + 0.18 * min(1.0, cap["fill"])
                + 0.18 * end_score
                + 0.10 * min(1.0, aspect / 1.3)
                + 0.12 * min(1.0, fill * 4.0)
                + method_bonus
                + 0.10
            )
            return float(np.clip(sc, 0.0, 0.99))

        # --- RGB-only path: geometric probes + white-end; distrust rim dual-blob ---
        w_end = self._white_end_score(white, cap, box) if white is not None else 0.0
        e_sup = 0.0
        if gray is not None and major is not None:
            e_sup = self._edge_support(gray, box, major)
        method_bonus = {
            "probe_down": 0.14,
            "probe_up": 0.12,
            "probe_left": 0.10,
            "probe_right": 0.10,
            "pca_perp_cap": 0.10,
            "cap_perp_axis": 0.08,
            "pca_combined": 0.06,
            "dual_blob_link": -0.08,  # rim noise on real benches
            "topdown_cap": 0.05,
        }.get(axis_method, 0.0)
        # Require some white at the cap end or the candidate is glare.
        if w_end < 0.12 and not (linked and axis_method == "dual_blob_link"):
            return 0.10
        sc = (
            0.28
            + 0.28 * w_end
            + 0.16 * end_score
            + 0.14 * min(1.0, e_sup * 2.0)
            + 0.10 * min(1.0, cap["fill"])
            + method_bonus
        )
        if linked and axis_method != "dual_blob_link":
            sc += 0.05
        return float(np.clip(sc, 0.0, 0.99))

    def _nms(self, cands: List[Det]) -> List[Det]:
        p = self.params
        cands = [c for c in cands if c.conf >= p.confidence_threshold]
        cands.sort(key=lambda d: d.conf, reverse=True)
        kept: List[Det] = []
        for d in cands:
            if any(box_iou(d, k) > p.nms_iou for k in kept):
                continue
            kept.append(d)
            if len(kept) >= int(p.max_detections):
                break
        return kept

    def _axis_candidates(self, combined: np.ndarray, cap: Dict) -> List[Tuple[np.ndarray, float, str]]:
        """Several orientation-agnostic axis hypotheses (unit vector, aspect, method)."""
        out: List[Tuple[np.ndarray, float, str]] = []
        link = self._probe_axis_from_body(combined, cap)
        if link is not None:
            out.append((link[1], link[2], link[3]))
        ys, xs = np.where(combined > 0)
        pca = self._pca_axis(ys, xs)
        if pca is not None and pca[2] >= self.params.min_axis_aspect:
            major = pca[1]
            out.append((major, pca[2], "pca_combined"))
            bw, bh = cap["bw"], cap["bh"]
            if max(bw, bh) / max(min(bw, bh), 1.0) >= 1.8:
                cap_long = np.array([1.0, 0.0]) if bw >= bh else np.array([0.0, 1.0])
                if abs(float(np.dot(major, cap_long))) > 0.75:
                    out.append((np.array([-major[1], major[0]]), pca[2], "pca_perp_cap"))
        # Cardinal probes — covers upright/sideways when body link is weak (RGB).
        for vec, name in (
            (np.array([0.0, 1.0]), "probe_down"),
            (np.array([0.0, -1.0]), "probe_up"),
            (np.array([1.0, 0.0]), "probe_right"),
            (np.array([-1.0, 0.0]), "probe_left"),
        ):
            out.append((vec, 1.2, name))
        # Cap-perpendicular (side-on end-cap prior).
        bw, bh = cap["bw"], cap["bh"]
        if bh >= bw * 1.4:
            out.append((np.array([1.0, 0.0]), 1.2, "cap_perp_axis"))
            out.append((np.array([-1.0, 0.0]), 1.2, "cap_perp_axis"))
        elif bw >= bh * 1.4:
            out.append((np.array([0.0, 1.0]), 1.2, "cap_perp_axis"))
            out.append((np.array([0.0, -1.0]), 1.2, "cap_perp_axis"))
        # Dedup by angle bin.
        uniq: List[Tuple[np.ndarray, float, str]] = []
        seen = set()
        for maj, asp, meth in out:
            nrm = float(np.linalg.norm(maj))
            if nrm < 1e-9:
                continue
            maj = maj / nrm
            key = int(round(np.rad2deg(np.arctan2(maj[1], maj[0])) / 15.0)) % 24
            if key in seen:
                continue
            seen.add(key)
            uniq.append((maj, asp, meth))
        return uniq

    def _edge_support(
        self, gray: np.ndarray, box: Tuple[float, float, float, float], major: np.ndarray
    ) -> float:
        """How well box borders sit on edges — helps RGB upright without depth."""
        h, w = gray.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return 0.0
        edges = cv2.Canny(gray, 60, 140)
        # Sample perimeter band.
        band = np.zeros_like(edges)
        band[y1:y2, x1 : min(x1 + 2, x2)] = 1
        band[y1:y2, max(x2 - 2, x1) : x2] = 1
        band[y1 : min(y1 + 2, y2), x1:x2] = 1
        band[max(y2 - 2, y1) : y2, x1:x2] = 1
        support = float((edges > 0)[band > 0].mean()) if band.any() else 0.0
        # Prefer elongated boxes aligned with major.
        bw, bh = x2 - x1, y2 - y1
        elong = max(bw, bh) / max(min(bw, bh), 1)
        align = abs(float(major[0])) if bw >= bh else abs(float(major[1]))
        return support * min(1.0, elong / 1.2) * (0.5 + 0.5 * align)

    def predict(
        self,
        bgr: np.ndarray,
        depth_mm: Optional[np.ndarray] = None,
    ) -> Tuple[List[Det], InferStats]:
        if bgr is None or bgr.size == 0:
            return [], InferStats(inference_ms=0.0)
        t0 = time.perf_counter()
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        white = self.white_cap_mask(bgr)
        body = self.body_cue_mask(bgr, depth_mm)
        seeds = self._cap_seeds(white)
        cands: List[Det] = []
        axis_counts: Dict[str, int] = {}
        n_linked = 0
        n_yolo = 0
        has_depth = depth_mm is not None

        for cap in seeds:
            linked_mask = self._link_body_touching(body, cap)
            linked = bool(linked_mask.any())
            if linked:
                n_linked += 1
            combined = cv2.bitwise_or(cap["mask"], linked_mask)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)

            best_det: Optional[Det] = None
            # Top-down: nearly round compact cap, no body link → AABB ≈ padded cap.
            if (
                (not linked)
                and cap["ar"] <= 1.22
                and cap["fill"] >= 0.62
                and 80 <= cap["area"] <= 3500
            ):
                x, y, bw, bh = cap["box"]
                pad = 0.06 * max(bw, bh)
                box = (
                    max(0.0, x - pad),
                    max(0.0, y - pad),
                    min(w - 1.0, x + bw + pad),
                    min(h - 1.0, y + bh + pad),
                )
                conf = self._score(
                    cap,
                    combined,
                    1.0,
                    "topdown_cap",
                    box,
                    h,
                    w,
                    linked,
                    has_depth=has_depth,
                    white=white,
                    gray=gray,
                    major=np.array([1.0, 0.0]),
                )
                best_det = Det(
                    box[0],
                    box[1],
                    box[2],
                    box[3],
                    conf=float(conf),
                    source="cascade",
                    axis_angle_deg=0.0,
                    axis_method="topdown_cap",
                )

            for major, aspect, method in self._axis_candidates(combined, cap):
                box = self._oriented_aabb(cap, major, h, w, combined)
                conf = self._score(
                    cap,
                    combined,
                    aspect,
                    method,
                    box,
                    h,
                    w,
                    linked,
                    has_depth=has_depth,
                    white=white,
                    gray=gray,
                    major=major,
                )
                if has_depth and linked and method == "dual_blob_link":
                    conf = min(0.99, conf + 0.08)
                if best_det is None or conf > best_det.conf:
                    ang = float(np.rad2deg(np.arctan2(major[1], major[0])))
                    best_det = Det(
                        box[0],
                        box[1],
                        box[2],
                        box[3],
                        conf=float(np.clip(conf, 0.0, 0.99)),
                        source="cascade",
                        axis_angle_deg=ang,
                        axis_method=method,
                    )
            if best_det is not None:
                axis_counts[best_det.axis_method] = axis_counts.get(best_det.axis_method, 0) + 1
                cands.append(best_det)

        # RGB-only rescue: YOLO proposals (trained on upright vial stills).
        # Skip when depth is present so sideways synth stays classical-driven.
        if (not has_depth) and self.params.use_yolo_rgb and self._yolo is not None:
            for yd in self._yolo.propose(bgr):
                yd.conf = min(0.99, yd.conf + float(self.params.yolo_boost))
                cands.append(yd)
                n_yolo += 1
                axis_counts["yolo_proposal"] = axis_counts.get("yolo_proposal", 0) + 1

        kept = self._nms(cands)
        ms = (time.perf_counter() - t0) * 1000.0
        return kept, InferStats(
            backend=self.backend,
            inference_ms=ms,
            n_caps=len(seeds),
            n_linked=n_linked,
            n_kept=len(kept),
            axis_methods=axis_counts,
            extras={"n_yolo": n_yolo, "has_depth": has_depth},
        )

    def annotate(self, bgr: np.ndarray, dets: Sequence[Det]) -> np.ndarray:
        out = bgr.copy()
        for d in dets:
            x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
            cv2.rectangle(out, (x1, y1), (x2, y2), (40, 200, 255), 2)
            cx, cy = 0.5 * (d.x1 + d.x2), 0.5 * (d.y1 + d.y2)
            rad = np.deg2rad(d.axis_angle_deg)
            L = 0.35 * max(d.x2 - d.x1, d.y2 - d.y1)
            p1 = (int(cx - L * np.cos(rad)), int(cy - L * np.sin(rad)))
            p2 = (int(cx + L * np.cos(rad)), int(cy + L * np.sin(rad)))
            cv2.line(out, p1, p2, (0, 255, 180), 2, cv2.LINE_AA)
            label = f"{d.conf:.2f} {d.axis_method} {d.axis_angle_deg:.0f}d"
            cv2.putText(
                out,
                label,
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (40, 200, 255),
                1,
                cv2.LINE_AA,
            )
        return out
