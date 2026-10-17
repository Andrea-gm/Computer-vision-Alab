"""Multi-scale template matching (+ optional ORB/AKAZE/SIFT homography).

Suitability pivot (documented in README):
  Clear plastic vials have almost no ORB/AKAZE keypoints on the cup body
  (dataset median ORB count on GT crops ≈ 0). Primary path is therefore
  multi-scale NCC on grayscale + Canny-edge templates. Feature matching runs
  only when a template/scene pair yields enough keypoints; boxes come from
  RANSAC homography inliers when available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .template_bank import Template, default_camera_root, ensure_templates, resolve_templates_dir
from .types import Det, annotate, nms_dets

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class InferStats:
    backend: str
    model_path: str
    inference_ms: float
    num_dets: int
    n_templates: int = 0
    n_feature_hits: int = 0


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
        dets.append(Det(x1, y1, x2, y2, conf=1.0, cls_id=cls_id, label="vial", source="gt"))
    return dets


def _white_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _h, s, v = cv2.split(hsv)
    mask = ((v >= 185) & (s <= 55)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _box_has_white_cap(bgr: np.ndarray, det: Det, frac_min: float = 0.02) -> bool:
    h, w = bgr.shape[:2]
    x1 = max(0, int(det.x1))
    y1 = max(0, int(det.y1))
    x2 = min(w, int(det.x2))
    y2 = min(h, int(det.y2))
    if x2 <= x1 or y2 <= y1:
        return False
    # Cap is typically in the upper ~40% of an upright vial box.
    y_cap = y1 + max(1, int(0.42 * (y2 - y1)))
    roi = bgr[y1:y_cap, x1:x2]
    if roi.size == 0:
        return False
    mask = _white_mask(roi)
    return float(mask.mean()) / 255.0 >= frac_min


def _make_feature_detector(backend: str):
    name = (backend or "AKAZE").upper()
    if name == "ORB":
        return cv2.ORB_create(nfeatures=800), "ORB"
    if name == "SIFT" and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=800), "SIFT"
    return cv2.AKAZE_create(), "AKAZE"


def _match_features(
    tmpl_gray: np.ndarray,
    scene_gray: np.ndarray,
    *,
    backend: str,
    min_matches: int,
    min_inliers: int,
) -> Optional[Tuple[Det, float, int]]:
    """Return (Det, score, inliers) from homography, or None."""
    det, used = _make_feature_detector(backend)
    kp1, des1 = det.detectAndCompute(tmpl_gray, None)
    kp2, des2 = det.detectAndCompute(scene_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None
    if des1.dtype != np.float32 and used in ("SIFT",):
        # BF Hamming for binary (ORB/AKAZE), L2 for SIFT.
        pass
    if used in ("ORB", "AKAZE"):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    try:
        pairs = matcher.knnMatch(des1, des2, k=2)
    except cv2.error:
        return None
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < min_matches:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return None
    inliers = int(mask.ravel().sum())
    if inliers < min_inliers:
        return None
    th, tw = tmpl_gray.shape[:2]
    corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
    try:
        projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    except cv2.error:
        return None
    xs = projected[:, 0]
    ys = projected[:, 1]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    score = float(np.clip(0.35 + 0.55 * (inliers / max(len(good), 1)), 0.0, 0.99))
    return (
        Det(x1, y1, x2, y2, conf=score, source=f"feature_{used.lower()}"),
        score,
        inliers,
    )


class VialTemplateOrbDetector:
    """Primary: multi-scale template NCC. Secondary: feature homography."""

    def __init__(
        self,
        templates_dir: Optional[Union[str, Path]] = None,
        *,
        camera_root: Optional[Union[str, Path]] = None,
        template_score_threshold: float = 0.52,
        enable_feature_match: bool = True,
        feature_backend: str = "AKAZE",
        min_feature_matches: int = 8,
        min_homography_inliers: int = 6,
        require_white_cap: bool = True,
        nms_iou: float = 0.40,
        max_detections: int = 1,
        aspect_min: float = 0.75,
        aspect_max: float = 2.20,
        scale_min: float = 0.50,
        scale_max: float = 2.20,
        scale_steps: int = 8,
        match_max_width: int = 480,
        gray_weight: float = 0.65,
        edge_weight: float = 0.35,
    ) -> None:
        self.camera_root = Path(camera_root) if camera_root else default_camera_root()
        self.templates_dir = resolve_templates_dir(templates_dir, self.camera_root)
        self.template_score_threshold = float(template_score_threshold)
        self.enable_feature_match = bool(enable_feature_match)
        self.feature_backend = feature_backend
        self.min_feature_matches = int(min_feature_matches)
        self.min_homography_inliers = int(min_homography_inliers)
        self.require_white_cap = bool(require_white_cap)
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

        self.templates: List[Template] = list(
            ensure_templates(self.templates_dir, self.camera_root)
        )
        self.backend = "template_orb"
        self.model_path = str(self.templates_dir)

    def predict(self, bgr: np.ndarray) -> Tuple[List[Det], InferStats]:
        t0 = time.perf_counter()
        dets, n_feat = self._detect(bgr)
        ms = (time.perf_counter() - t0) * 1000.0
        stats = InferStats(
            backend=self.backend,
            model_path=self.model_path,
            inference_ms=ms,
            num_dets=len(dets),
            n_templates=len(self.templates),
            n_feature_hits=n_feat,
        )
        return dets, stats

    def _detect(self, bgr: np.ndarray) -> Tuple[List[Det], int]:
        if bgr is None or getattr(bgr, "size", 0) == 0 or not self.templates:
            return [], 0
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
        for tmpl in self.templates:
            th, tw = tmpl.gray.shape[:2]
            best: Optional[Tuple[float, int, int, int, int]] = None  # score,x,y,nw,nh
            for s in scales:
                nw = int(round(tw * s))
                nh = int(round(th * s))
                if nw < 12 or nh < 16 or nw >= Ww or nh >= Hh:
                    continue
                t_gray = cv2.resize(tmpl.gray, (nw, nh), interpolation=cv2.INTER_AREA)
                t_edge = cv2.resize(tmpl.edge, (nw, nh), interpolation=cv2.INTER_AREA)
                r_gray = cv2.matchTemplate(gray, t_gray, cv2.TM_CCOEFF_NORMED)
                r_edge = cv2.matchTemplate(
                    edge.astype(np.float32),
                    t_edge.astype(np.float32),
                    cv2.TM_CCOEFF_NORMED,
                )
                score_map = self.gray_weight * r_gray + self.edge_weight * r_edge
                _minv, maxv, _minloc, maxloc = cv2.minMaxLoc(score_map)
                if maxv < self.template_score_threshold:
                    continue
                if best is None or maxv > best[0]:
                    best = (float(maxv), int(maxloc[0]), int(maxloc[1]), nw, nh)
            if best is not None:
                score, x, y, nw, nh = best
                cands.append(
                    Det(
                        float(x),
                        float(y),
                        float(x + nw),
                        float(y + nh),
                        conf=score,
                        source="template",
                    )
                )

        n_feat = 0
        if self.enable_feature_match:
            # Feature match is expensive; try a few templates with the most corners.
            def _corner_count(t: Template) -> int:
                pts = cv2.goodFeaturesToTrack(t.gray, 32, 0.01, 3)
                return 0 if pts is None else int(pts.shape[0])

            ranked = sorted(self.templates, key=_corner_count, reverse=True)[:5]
            for tmpl in ranked:
                hit = _match_features(
                    tmpl.gray,
                    gray,
                    backend=self.feature_backend,
                    min_matches=self.min_feature_matches,
                    min_inliers=self.min_homography_inliers,
                )
                if hit is None:
                    continue
                det, _score, _inl = hit
                # Clip to frame.
                det.x1 = float(np.clip(det.x1, 0, Ww - 1))
                det.y1 = float(np.clip(det.y1, 0, Hh - 1))
                det.x2 = float(np.clip(det.x2, 0, Ww - 1))
                det.y2 = float(np.clip(det.y2, 0, Hh - 1))
                cands.append(det)
                n_feat += 1

        # Scale boxes back to original resolution.
        if scale_down != 1.0:
            inv = 1.0 / scale_down
            for d in cands:
                d.x1 *= inv
                d.y1 *= inv
                d.x2 *= inv
                d.y2 *= inv

        filtered: List[Det] = []
        for d in cands:
            bw = max(1.0, d.x2 - d.x1)
            bh = max(1.0, d.y2 - d.y1)
            aspect = bh / bw
            if aspect < self.aspect_min or aspect > self.aspect_max:
                continue
            area_frac = (bw * bh) / float(max(w0 * h0, 1))
            if area_frac < 0.002 or area_frac > 0.85:
                continue
            if self.require_white_cap and not _box_has_white_cap(bgr, d):
                continue
            filtered.append(d)

        kept = nms_dets(filtered, self.nms_iou, self.max_detections)
        return kept, n_feat

    def annotate(self, bgr: np.ndarray, dets: Sequence[Det]) -> np.ndarray:
        return annotate(bgr, list(dets), color=(0, 200, 255))


# Keep name alias used by evaluate / node style parity with agent_01.
VialDetector = VialTemplateOrbDetector
