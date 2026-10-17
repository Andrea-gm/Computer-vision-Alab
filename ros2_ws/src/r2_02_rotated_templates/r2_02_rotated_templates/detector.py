"""Multi-scale + multi-angle template matching for upright and sideways vials.

Round-1 failure modes this fixes (agent_03_template_orb):
  - Templates built only from upright GT (aspect filter 0.95–1.60)
  - No in-plane rotation search → sideways NCC score collapses silently
  - White-cap gate checked only the top strip (upright assumption)
  - Aspect gate 0.75–2.20 rejected true sideways boxes (~0.45–0.68)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .template_bank import (
    Template,
    default_camera_root,
    ensure_templates,
    resolve_templates_dir,
    rotate_bound_gray,
)
from .types import Det, annotate, nms_dets, orientation_from_angle

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Default angles: full circle at 45° steps (covers upright + sideways + diagonal).
DEFAULT_ANGLES = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)


@dataclass
class InferStats:
    backend: str
    model_path: str
    inference_ms: float
    num_dets: int
    n_templates: int = 0
    n_rotated_views: int = 0
    best_angle: float = 0.0


def _rotate_cap_side(cap_side: str, angle: float) -> str:
    """Rotate cardinal cap side by template angle (CW-positive OpenCV)."""
    order = ["top", "right", "bottom", "left"]
    if cap_side not in order:
        return cap_side
    steps = int(round((float(angle) % 360.0) / 90.0)) % 4
    # OpenCV positive angle is CCW; our rotate_bound uses CCW. Map accordingly.
    # getRotationMatrix2D positive angle = CCW, so top→left at +90.
    idx = order.index(cap_side)
    return order[(idx - steps) % 4]


def _observed_cap_side(bgr: np.ndarray, det: Det) -> str:
    box = _box_roi(bgr, det)
    if box is None:
        return "unknown"
    x1, y1, x2, y2 = box
    roi = bgr[y1:y2, x1:x2]
    white = _white_mask(roi, v_min=165, s_max=70).astype(np.float32) / 255.0
    h, w = white.shape
    left = float(white[:, : max(1, w // 3)].mean())
    right = float(white[:, 2 * w // 3 :].mean())
    top = float(white[: max(1, h // 3), :].mean())
    bot = float(white[2 * h // 3 :, :].mean())
    scores = {"left": left, "right": right, "top": top, "bottom": bot}
    return max(scores, key=scores.get)


@dataclass
class _RotatedView:
    base_name: str
    pose: str
    angle: float
    gray: np.ndarray
    edge: np.ndarray
    expected_cap_side: str = "top"


def list_images(folder: Union[str, Path]) -> List[Path]:
    root = Path(folder)
    if not root.exists():
        return []
    return [
        p
        for p in sorted(root.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def load_yolo_label(label_path: Path, width: int, height: int) -> List[Det]:
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    dets: List[Det] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, bw, bh = [float(v) for v in parts[1:]]
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        aspect = (y2 - y1) / max(x2 - x1, 1e-6)
        orient = "sideways" if aspect < 0.85 else "upright"
        dets.append(
            Det(
                x1,
                y1,
                x2,
                y2,
                conf=1.0,
                cls_id=cls_id,
                label="vial",
                source="gt",
                orientation=orient,
            )
        )
    return dets


def _white_mask(bgr: np.ndarray, *, v_min: int = 185, s_max: int = 55) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _h, s, v = cv2.split(hsv)
    mask = ((v >= int(v_min)) & (s <= int(s_max))).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _has_cup_rim_edges(bgr: np.ndarray, det: Det) -> bool:
    """Sideways/upright vials have parallel dark rim lines on the *body*, not cap."""
    box = _box_roi(bgr, det)
    if box is None:
        return False
    x1, y1, x2, y2 = box
    roi = bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    h, w = edges.shape
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    aspect = bh / float(bw)
    white = _white_mask(roi, v_min=165, s_max=70).astype(np.float32) / 255.0
    # Restrict rim check to the body side (opposite the white cap).
    if aspect < 0.85:  # sideways
        left_w = float(white[:, : max(1, w // 3)].mean())
        right_w = float(white[:, 2 * w // 3 :].mean())
        if left_w >= right_w:
            body = edges[:, w // 3 :]  # cap left → body right
        else:
            body = edges[:, : 2 * w // 3]
        if body.size == 0:
            return False
        bh2, bw2 = body.shape
        band = max(1, bh2 // 6)
        top = float(body[:band, :].mean())
        bot = float(body[bh2 - band :, :].mean())
        # Body should also be darker than a bright slab leftover.
        body_gray = gray[:, w // 3 :] if left_w >= right_w else gray[:, : 2 * w // 3]
        if float(body_gray.mean()) > 140:
            return False
        return top >= 10.0 and bot >= 10.0
    if aspect > 1.15:  # upright
        top_w = float(white[: max(1, h // 3), :].mean())
        bot_w = float(white[2 * h // 3 :, :].mean())
        body = edges[h // 3 :, :] if top_w >= bot_w else edges[: 2 * h // 3, :]
        if body.size == 0:
            return False
        _bh2, bw2 = body.shape
        band = max(1, bw2 // 6)
        left = float(body[:, :band].mean())
        right = float(body[:, bw2 - band :].mean())
        return left >= 8.0 and right >= 8.0
    band_h = max(1, h // 6)
    band_w = max(1, w // 6)
    top = float(edges[:band_h, :].mean())
    bot = float(edges[h - band_h :, :].mean())
    left = float(edges[:, :band_w].mean())
    right = float(edges[:, w - band_w :].mean())
    return (top >= 8.0 and bot >= 8.0) or (left >= 8.0 and right >= 8.0)


def _box_roi(bgr: np.ndarray, det: Det):
    h, w = bgr.shape[:2]
    x1 = max(0, int(det.x1))
    y1 = max(0, int(det.y1))
    x2 = min(w, int(det.x2))
    y2 = min(h, int(det.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _box_has_white_cap(bgr: np.ndarray, det: Det, frac_min: float = 0.015) -> bool:
    """Orientation-aware: check top/bottom/left/right strips for a white cap band."""
    box = _box_roi(bgr, det)
    if box is None:
        return False
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    strips = [
        bgr[y1 : y1 + max(1, int(0.42 * bh)), x1:x2],  # top (upright)
        bgr[y2 - max(1, int(0.42 * bh)) : y2, x1:x2],  # bottom (upside-down)
        bgr[y1:y2, x1 : x1 + max(1, int(0.42 * bw))],  # left (sideways)
        bgr[y1:y2, x2 - max(1, int(0.42 * bw)) : x2],  # right (sideways)
    ]
    for roi in strips:
        if roi.size == 0:
            continue
        if float(_white_mask(roi).mean()) / 255.0 >= frac_min:
            return True
    return False


def _asymmetric_end_cap(white: np.ndarray) -> bool:
    """True if white mass is concentrated on exactly one end (cap), not scattered."""
    h, w = white.shape[:2]
    if h < 4 or w < 4:
        return False
    wf = white.astype(np.float32) / 255.0
    left = float(wf[:, : max(1, w // 3)].mean())
    right = float(wf[:, 2 * w // 3 :].mean())
    top = float(wf[: max(1, h // 3), :].mean())
    bot = float(wf[2 * h // 3 :, :].mean())
    mid_x = float(wf[:, w // 3 : 2 * w // 3].mean())
    mid_y = float(wf[h // 3 : 2 * h // 3, :].mean())
    side_ok = (left >= 0.35 and right <= 0.12 and mid_x <= 0.20) or (
        right >= 0.35 and left <= 0.12 and mid_x <= 0.20
    )
    up_ok = (top >= 0.35 and bot <= 0.12 and mid_y <= 0.20) or (
        bot >= 0.35 and top <= 0.12 and mid_y <= 0.20
    )
    if not (side_ok or up_ok):
        return False
    # Cap must span most of the short axis (full vial diameter), not a slab corner.
    ys, xs = np.where(white > 0)
    if ys.size < 10:
        return False
    if side_ok:
        span = (ys.max() - ys.min() + 1) / float(h)
        return span >= 0.70
    span = (xs.max() - xs.min() + 1) / float(w)
    return span >= 0.70


def _box_has_vial_structure(bgr: np.ndarray, det: Det) -> bool:
    """Pose-aware structure gate.

    Upright real photos: soft white-cap check (clear cup rarely has crisp synth rims).
    Sideways: strict end-cap + dark body + cup-rim check (rejects desk/slab FP).
    """
    box = _box_roi(bgr, det)
    if box is None:
        return False
    x1, y1, x2, y2 = box
    roi = bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    bw = float(x2 - x1)
    bh = float(y2 - y1)
    aspect = bh / max(bw, 1.0)

    # --- Upright / near-square path (real Orbbec stills) ---
    if aspect >= 0.90:
        white = _white_mask(roi, v_min=175, s_max=60)
        white_frac = float(white.mean()) / 255.0
        if white_frac < 0.02 or white_frac > 0.50:
            return False
        # Top/bottom cap only (do not accept left/right strips — desk clutter FP).
        h = roi.shape[0]
        top = roi[: max(1, int(0.42 * h)), :]
        bot = roi[h - max(1, int(0.42 * h)) :, :]
        top_f = float(_white_mask(top, v_min=175, s_max=60).mean()) / 255.0
        bot_f = float(_white_mask(bot, v_min=175, s_max=60).mean()) / 255.0
        if max(top_f, bot_f) < 0.04:
            return False
        # Cap end should dominate the opposite end.
        if max(top_f, bot_f) < 1.8 * (min(top_f, bot_f) + 0.01):
            return False
        return True

    # --- Sideways path (synth + fallen vials) ---
    white = _white_mask(roi, v_min=165, s_max=70)
    white_frac = float(white.mean()) / 255.0
    if white_frac < 0.05 or white_frac > 0.42:
        return False
    if not _asymmetric_end_cap(white):
        return False
    if 0.75 <= aspect <= 0.89 and white_frac >= 0.28:
        return False
    non_white = white < 128
    if int(non_white.sum()) < 20:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    body_v = float(v[non_white].mean())
    cap_v = float(v[~non_white].mean()) if int((~non_white).sum()) else 255.0
    if body_v > 160:
        return False
    if cap_v - body_v < 30:
        return False
    if not _has_cup_rim_edges(bgr, det):
        return False
    return True


def _local_peaks(
    score_map: np.ndarray,
    *,
    thr: float,
    max_peaks: int = 3,
    min_dist: int = 18,
) -> List[Tuple[float, int, int]]:
    """Greedy peaks so a clutter max cannot hide a second vial peak."""
    sm = score_map.copy()
    peaks: List[Tuple[float, int, int]] = []
    for _ in range(max(1, max_peaks)):
        _minv, maxv, _minloc, maxloc = cv2.minMaxLoc(sm)
        if maxv < thr:
            break
        x, y = int(maxloc[0]), int(maxloc[1])
        peaks.append((float(maxv), x, y))
        x0 = max(0, x - min_dist)
        y0 = max(0, y - min_dist)
        x1 = min(sm.shape[1], x + min_dist + 1)
        y1 = min(sm.shape[0], y + min_dist + 1)
        sm[y0:y1, x0:x1] = -1.0
    return peaks


def _structure_bonus(bgr: np.ndarray, det: Det) -> float:
    """Soft score boost for cap+body contrast (used for ranking)."""
    box = _box_roi(bgr, det)
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    roi = bgr[y1:y2, x1:x2]
    white = _white_mask(roi)
    white_frac = float(white.mean()) / 255.0
    # Ideal white fraction near ~0.2
    return float(max(0.0, 1.0 - abs(white_frac - 0.22) / 0.35))


def _parse_angles(angles: Optional[Sequence[float]]) -> Tuple[float, ...]:
    if not angles:
        return DEFAULT_ANGLES
    out = tuple(float(a) % 360.0 for a in angles)
    return out if out else DEFAULT_ANGLES


class RotatedTemplateDetector:
    """Primary: multi-scale × multi-angle NCC on gray+edge templates."""

    def __init__(
        self,
        templates_dir: Optional[Union[str, Path]] = None,
        *,
        camera_root: Optional[Union[str, Path]] = None,
        template_score_threshold: float = 0.48,
        require_white_cap: bool = True,
        require_vial_structure: bool = True,
        nms_iou: float = 0.40,
        max_detections: int = 1,
        aspect_min: float = 0.35,
        aspect_max: float = 3.00,
        scale_min: float = 0.55,
        scale_max: float = 2.00,
        scale_steps: int = 6,
        match_max_width: int = 400,
        gray_weight: float = 0.65,
        edge_weight: float = 0.35,
        angles_deg: Optional[Sequence[float]] = None,
        early_exit_score: float = 0.97,
    ) -> None:
        self.camera_root = Path(camera_root) if camera_root else default_camera_root()
        self.templates_dir = resolve_templates_dir(templates_dir, self.camera_root)
        self.template_score_threshold = float(template_score_threshold)
        self.require_white_cap = bool(require_white_cap)
        self.require_vial_structure = bool(require_vial_structure)
        self.nms_iou = float(nms_iou)
        self.max_detections = int(max_detections)
        self.aspect_min = float(aspect_min)
        self.aspect_max = float(aspect_max)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.scale_steps = max(3, int(scale_steps))
        self.match_max_width = int(match_max_width)
        self.gray_weight = float(gray_weight)
        self.edge_weight = float(edge_weight)
        self.angles_deg = _parse_angles(angles_deg)
        self.early_exit_score = float(early_exit_score)

        self.templates: List[Template] = list(
            ensure_templates(self.templates_dir, self.camera_root)
        )
        self.rotated_views: List[_RotatedView] = self._build_rotated_views(self.templates)
        self.backend = "rotated_templates"
        self.model_path = str(self.templates_dir)

    def _build_rotated_views(self, templates: Sequence[Template]) -> List[_RotatedView]:
        views: List[_RotatedView] = []
        for tmpl in templates:
            for ang in self.angles_deg:
                # Sideways base templates already cover ~90° pose; still rotate
                # so ±45° and upside-down are covered without silent miss.
                g = rotate_bound_gray(tmpl.gray, ang, edge=False)
                e = rotate_bound_gray(tmpl.edge, ang, edge=True)
                if g.shape[0] < 10 or g.shape[1] < 10:
                    continue
                views.append(
                    _RotatedView(
                        base_name=tmpl.name,
                        pose=tmpl.pose,
                        angle=float(ang),
                        gray=g,
                        edge=e,
                        expected_cap_side=_rotate_cap_side(tmpl.cap_side, ang),
                    )
                )
        return views

    def predict(self, bgr: np.ndarray) -> Tuple[List[Det], InferStats]:
        t0 = time.perf_counter()
        dets, best_ang = self._detect(bgr)
        ms = (time.perf_counter() - t0) * 1000.0
        stats = InferStats(
            backend=self.backend,
            model_path=self.model_path,
            inference_ms=ms,
            num_dets=len(dets),
            n_templates=len(self.templates),
            n_rotated_views=len(self.rotated_views),
            best_angle=best_ang,
        )
        return dets, stats

    def _detect(self, bgr: np.ndarray) -> Tuple[List[Det], float]:
        if bgr is None or getattr(bgr, "size", 0) == 0 or not self.rotated_views:
            return [], 0.0
        h0, w0 = bgr.shape[:2]
        scale_down = 1.0
        work = bgr
        if self.match_max_width > 0 and w0 > self.match_max_width:
            scale_down = self.match_max_width / float(w0)
            work = cv2.resize(bgr, (int(w0 * scale_down), int(h0 * scale_down)))
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 40, 120)
        Hh, Ww = gray.shape
        scales = np.linspace(self.scale_min, self.scale_max, self.scale_steps)

        cands: List[Det] = []
        best_angle = 0.0
        global_best = -1.0

        # Prioritize native poses: sideways@0/180 and upright@0/180 first, then
        # diagonals. Avoids early-exit locking onto clutter with a wrong angle.
        def _view_priority(v: _RotatedView) -> Tuple[int, float]:
            a = v.angle % 180.0
            axis = min(a, 180.0 - a)  # 0 for axis-aligned
            native = 0 if axis <= 1.0 else (1 if axis <= 46.0 else 2)
            pose_rank = 0 if v.pose == "sideways" else 1
            return (native, float(pose_rank))

        ordered = sorted(self.rotated_views, key=_view_priority)
        # Cap expensive search: keep all axis-aligned views + a subset of diagonals.
        axis_views = [v for v in ordered if _view_priority(v)[0] == 0]
        diag_views = [v for v in ordered if _view_priority(v)[0] != 0]
        # Keep every base template at 0/180 and at 90/270 (pose flip), plus
        # limited diagonals for 45° coverage.
        search_views = axis_views + diag_views[: max(16, len(self.templates) * 2)]

        for view in search_views:
            th, tw = view.gray.shape[:2]
            view_cands: List[Tuple[float, int, int, int, int, float]] = []
            # (score, x, y, nw, nh, edge_score)
            for s in scales:
                nw = int(round(tw * s))
                nh = int(round(th * s))
                if nw < 12 or nh < 12 or nw >= Ww or nh >= Hh:
                    continue
                t_gray = cv2.resize(view.gray, (nw, nh), interpolation=cv2.INTER_AREA)
                t_edge = cv2.resize(view.edge, (nw, nh), interpolation=cv2.INTER_AREA)
                r_gray = cv2.matchTemplate(gray, t_gray, cv2.TM_CCOEFF_NORMED)
                r_edge = cv2.matchTemplate(
                    edge.astype(np.float32),
                    t_edge.astype(np.float32),
                    cv2.TM_CCOEFF_NORMED,
                )
                score_map = self.gray_weight * r_gray + self.edge_weight * r_edge
                for score, x, y in _local_peaks(
                    score_map, thr=self.template_score_threshold, max_peaks=3, min_dist=20
                ):
                    edge_s = float(r_edge[y, x]) if 0 <= y < r_edge.shape[0] and 0 <= x < r_edge.shape[1] else 0.0
                    # Textureless desk mottling can match gray; require some edge agreement.
                    if edge_s < 0.15:
                        continue
                    view_cands.append((score, x, y, nw, nh, edge_s))
            if not view_cands:
                continue
            # Keep top few peaks for this view across scales.
            view_cands.sort(key=lambda t: t[0], reverse=True)
            for score, x, y, nw, nh, edge_s in view_cands[:4]:
                if view.pose == "sideways":
                    eff_angle = (view.angle + 90.0) % 360.0
                else:
                    eff_angle = view.angle % 360.0
                orient = orientation_from_angle(eff_angle)
                aspect = nh / float(max(nw, 1))
                if aspect < 0.80:
                    orient = "sideways"
                elif aspect > 1.15:
                    orient = "upright"
                # Blend edge agreement into confidence.
                conf = float(0.85 * score + 0.15 * max(0.0, edge_s))
                # Pose consistency: a native sideways template at ~0/180 must
                # produce a wide box; upright@0/180 must stay tall. Prevents
                # sideways templates from claiming upright clutter peaks.
                ang_mod = abs(view.angle % 180.0)
                axis_aligned = ang_mod <= 20.0 or abs(ang_mod - 180.0) <= 20.0
                if axis_aligned:
                    if view.pose == "sideways" and aspect > 0.88:
                        continue
                    if view.pose == "upright" and aspect < 0.92:
                        continue
                cands.append(
                    Det(
                        float(x),
                        float(y),
                        float(x + nw),
                        float(y + nh),
                        conf=conf,
                        source=f"tmpl_{view.pose}@{view.angle:.0f}|{view.expected_cap_side}",
                        angle_deg=float(view.angle),
                        orientation=orient,
                    )
                )
                if conf > global_best:
                    global_best = conf
                    best_angle = float(view.angle)

        if scale_down != 1.0:
            inv = 1.0 / scale_down
            for d in cands:
                d.x1 *= inv
                d.y1 *= inv
                d.x2 *= inv
                d.y2 *= inv

        filtered: List[Det] = []
        for d in cands:
            # Template scale often undershoots GT by a few px (cuts cup rims).
            # Try a short expand/shrink refine before structure gating.
            refined = self._refine_box(bgr, d, w0, h0)
            if refined is None:
                continue
            d = refined
            bw = max(1.0, d.x2 - d.x1)
            bh = max(1.0, d.y2 - d.y1)
            aspect = bh / bw
            if aspect < self.aspect_min or aspect > self.aspect_max:
                continue
            area_frac = (bw * bh) / float(max(w0 * h0, 1))
            if area_frac < 0.002 or area_frac > 0.85:
                continue
            # Rank by NCC × structure (penalize near-solid white slabs).
            d.conf = float(np.clip(d.conf * (0.70 + 0.30 * _structure_bonus(bgr, d)), 0.0, 0.999))
            filtered.append(d)

        kept = nms_dets(filtered, self.nms_iou, self.max_detections)
        if kept:
            best_angle = float(kept[0].angle_deg)
        return kept, best_angle

    def _refine_box(self, bgr: np.ndarray, det: Det, w0: int, h0: int) -> Optional[Det]:
        """Try modest expand/shrink; only if the seed already looks cap-like."""
        box = _box_roi(bgr, det)
        if box is None:
            return None
        x1, y1, x2, y2 = box
        seed_roi = bgr[y1:y2, x1:x2]
        if seed_roi.size == 0:
            return None
        seed_white = _white_mask(seed_roi, v_min=165, s_max=70)
        seed_wf = float(seed_white.mean()) / 255.0
        def _cap_ok(cand: Det) -> bool:
            expected = "unknown"
            if "|" in det.source:
                expected = det.source.rsplit("|", 1)[-1]
            if expected not in ("left", "right", "top", "bottom"):
                return True
            # Cap polarity is decisive for sideways; upright photos vary.
            bw = max(1.0, cand.x2 - cand.x1)
            bh = max(1.0, cand.y2 - cand.y1)
            if bh / bw >= 0.90 and expected in ("left", "right"):
                return True
            return _observed_cap_side(bgr, cand) == expected

        # Refuse to grow random desk patches — seed must already look like a cap+body.
        if seed_wf < 0.06 or seed_wf > 0.45 or not _asymmetric_end_cap(seed_white):
            ok = True
            if self.require_vial_structure:
                ok = _box_has_vial_structure(bgr, det)
            elif self.require_white_cap:
                ok = _box_has_white_cap(bgr, det)
            if ok and _cap_ok(det):
                det.source = det.source.split("|", 1)[0]
                return det
            return None

        scales = (1.0, 1.10, 1.18, 0.94)
        for s in scales:
            cx = 0.5 * (det.x1 + det.x2)
            cy = 0.5 * (det.y1 + det.y2)
            bw = max(1.0, (det.x2 - det.x1) * s)
            bh = max(1.0, (det.y2 - det.y1) * s)
            cand = Det(
                max(0.0, cx - bw / 2.0),
                max(0.0, cy - bh / 2.0),
                min(float(w0), cx + bw / 2.0),
                min(float(h0), cy + bh / 2.0),
                conf=det.conf,
                source=det.source,
                angle_deg=det.angle_deg,
                orientation=det.orientation,
            )
            if self.require_vial_structure:
                if not _box_has_vial_structure(bgr, cand):
                    continue
            elif self.require_white_cap:
                if not _box_has_white_cap(bgr, cand):
                    continue
            # Cap polarity must match the template that produced this peak.
            expected = "unknown"
            if "|" in det.source:
                expected = det.source.rsplit("|", 1)[-1]
            observed = _observed_cap_side(bgr, cand)
            if expected in ("left", "right", "top", "bottom") and observed != expected:
                continue
            cand.source = det.source.split("|", 1)[0]
            return cand
        return None

    def annotate(self, bgr: np.ndarray, dets: Sequence[Det]) -> np.ndarray:
        return annotate(bgr, list(dets), color=(40, 200, 80))


VialDetector = RotatedTemplateDetector
