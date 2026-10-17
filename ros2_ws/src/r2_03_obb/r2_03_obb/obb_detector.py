"""YOLOv8-OBB detector + classical oriented-rect fallback + AABB→OBB refine."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .obb_geometry import OBBDet, annotate_obb, canonical_obb, corners_to_obb  # noqa: F401


@dataclass
class InferStats:
    backend: str
    model_path: str
    inference_ms: float
    num_dets: int


def default_camera_root() -> Path:
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / "vial_dataset").is_dir() and (
            (parent / "runs").is_dir() or (parent / "yolov8s.pt").is_file()
        ):
            return parent
    for parent in here.parents:
        if parent.name == "ros2_ws":
            sibling = parent.parent / "robotics"
            if sibling.is_dir():
                return sibling
            break
    return here.parents[4]


def default_package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_obb_weights(
    explicit: Optional[Union[str, Path]],
    camera_root: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Prefer package-trained OBB weights, then Camera OBB runs, then none."""
    root = Path(camera_root) if camera_root else default_camera_root()
    pkg = default_package_root()
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    candidates = [
        pkg / "models" / "best.pt",
        pkg / "models" / "yolov8n-obb-vial.pt",
        root / "runs" / "obb-vial" / "weights" / "best.pt",
        root / "runs" / "r2_03_obb" / "weights" / "best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_aabb_weights(camera_root: Optional[Union[str, Path]] = None) -> Optional[Path]:
    root = Path(camera_root) if camera_root else default_camera_root()
    for path in [
        root / "runs" / "vial-live" / "weights" / "best.pt",
        root / "runs" / "train-race" / "r1-winner" / "weights" / "best.pt",
        root / "runs" / "bon-winner" / "weights" / "best.pt",
        root / "yolov8s.pt",
        root / "yolov8n.pt",
    ]:
        if path.exists():
            return path
    return None


def _white_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = ((v >= 175) & (s <= 60)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def detect_classical_upright(bgr: np.ndarray, *, max_dets: int = 3) -> List[OBBDet]:
    """White-cap → expand downward upright OBB (helps synth/painted upright vials)."""
    h, w = bgr.shape[:2]
    mask = _white_mask(bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = float(h * w)
    scored: List[Tuple[float, OBBDet]] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        frac = area / max(area_img, 1.0)
        if frac < 0.0006 or frac > 0.15:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 8 or bh < 6:
            continue
        # Cap on top → cup below (upright prior).
        expand_h = int(max(bh * 2.6, bw * 1.25))
        x1 = max(0, x - int(0.12 * bw))
        y1 = max(0, y - int(0.10 * bh))
        x2 = min(w - 1, x + bw + int(0.12 * bw))
        y2 = min(h - 1, y + expand_h)
        box_w = float(x2 - x1)
        box_h = float(y2 - y1)
        if box_h < box_w * 1.05:
            continue
        # Require some darker pixels under the cap (clear cup / body).
        mid_y0 = y + bh
        mid_y1 = min(h, y + expand_h)
        if mid_y1 <= mid_y0:
            continue
        body = bgr[mid_y0:mid_y1, max(0, x) : min(w, x + bw)]
        if body.size == 0:
            continue
        if float(body.mean()) > 155:
            continue
        # Cap should be brighter than body (white lid vs darker cup).
        cap_patch = bgr[y : y + bh, x : x + bw]
        if cap_patch.size == 0 or float(cap_patch.mean()) < float(body.mean()) + 25:
            continue
        det = canonical_obb(
            0.5 * (x1 + x2),
            0.5 * (y1 + y2),
            box_h,
            box_w,
            -90.0,
            conf=0.58,
            source="classical_upright",
        )
        scored.append((det.conf, det))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [d for _, d in scored[:max_dets]]


def detect_classical_obb(bgr: np.ndarray, *, max_dets: int = 3) -> List[OBBDet]:
    """White-cap blob → grow along PCA → minAreaRect oriented box.

    Handles upright (grow "down" along vertical) and sideways (grow along
    horizontal / principal axis) without forcing axis-aligned boxes.
    """
    h, w = bgr.shape[:2]
    mask = _white_mask(bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = float(h * w)
    scored: List[Tuple[float, OBBDet]] = []

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        frac = area / max(area_img, 1.0)
        if frac < 0.0006 or frac > 0.20:
            continue
        if len(cnt) < 5:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (cw, ch), ang = rect
        cap_long = max(cw, ch)
        cap_short = max(1.0, min(cw, ch))
        if cap_long < 8:
            continue

        # Principal axis of the cap blob.
        pts = cnt.reshape(-1, 2).astype(np.float32)
        mean, eigenvectors = cv2.PCACompute(pts, mean=None)
        axis = eigenvectors[0].astype(np.float64)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
        # Prefer more horizontal axis when cap is elongated sideways.
        if abs(axis[0]) < abs(axis[1]) and max(cw, ch) / min(cw, ch) < 1.35:
            # nearly round top-down / upright cap — grow vertically
            axis = np.array([0.0, 1.0], dtype=np.float64)

        # Search body extent along ±axis using edge density / darker cup.
        center = mean.reshape(2)
        step = max(2.0, cap_short * 0.25)
        best_dir = axis
        best_len = cap_long * 1.2
        for direction in (axis, -axis):
            length = 0.0
            dark_hits = 0
            for i in range(1, 48):
                p = center + direction * (i * step)
                x, y = int(round(p[0])), int(round(p[1]))
                if x < 0 or y < 0 or x >= w or y >= h:
                    break
                # stop on bright clutter far from vial; continue through dark cup
                pix = bgr[y, x]
                bright = int(pix[0]) > 200 and int(pix[1]) > 200 and int(pix[2]) > 200
                edge_here = edges[y, x] > 0
                if bright and i > 3:
                    break
                if int(gray[y, x]) < 140 or edge_here:
                    dark_hits += 1
                length = i * step
                if dark_hits >= 2 and i > int(cap_long / step):
                    # keep extending a bit past cup
                    if i > int(2.4 * cap_long / step):
                        break
            if length > best_len:
                best_len = length
                best_dir = direction

        total_len = float(np.clip(best_len + 0.5 * cap_long, cap_long * 1.6, cap_long * 3.8))
        diameter = float(np.clip(cap_short * 1.15, 10.0, cap_long * 1.8))
        # Shift center toward body.
        body_center = center + best_dir * (0.35 * total_len)
        angle = float(np.degrees(np.arctan2(best_dir[1], best_dir[0])))
        det = canonical_obb(
            float(body_center[0]),
            float(body_center[1]),
            total_len,
            diameter,
            angle,
            conf=0.45,
            source="classical_obb",
        )
        # Aspect prior: vials are elongated.
        if det.width / max(det.height, 1.0) < 1.15:
            continue
        if det.width * det.height > 0.35 * area_img:
            continue
        score = float(np.clip(0.3 + 0.08 * (det.width / max(det.height, 1.0)), 0.25, 0.8))
        det.conf = score
        scored.append((score, det))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [d for _, d in scored[:max_dets]]


def refine_aabb_to_obb(bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float, conf: float) -> OBBDet:
    """Fit minAreaRect inside an AABB crop (edge mask) for oriented output."""
    h, w = bgr.shape[:2]
    xa, ya = max(0, int(x1)), max(0, int(y1))
    xb, yb = min(w, int(x2)), min(h, int(y2))
    if xb - xa < 6 or yb - ya < 6:
        return canonical_obb(
            0.5 * (x1 + x2),
            0.5 * (y1 + y2),
            max(x2 - x1, y2 - y1),
            min(x2 - x1, y2 - y1),
            0.0 if (x2 - x1) >= (y2 - y1) else -90.0,
            conf=conf,
            source="aabb_refine",
        )
    crop = bgr[ya:yb, xa:xb]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 140)
    # Prefer white-cap + darker body union.
    mask = _white_mask(crop)
    dark = (gray < 130).astype(np.uint8) * 255
    union = cv2.bitwise_or(mask, cv2.bitwise_and(dark, edges))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, k)
    ys, xs = np.where(union > 0)
    if xs.size < 12:
        ys, xs = np.where(edges > 0)
    if xs.size < 8:
        return canonical_obb(
            0.5 * (x1 + x2),
            0.5 * (y1 + y2),
            max(x2 - x1, y2 - y1),
            min(max(x2 - x1, 1.0), max(y2 - y1, 1.0)),
            0.0 if (x2 - x1) >= (y2 - y1) else -90.0,
            conf=conf,
            source="aabb_refine",
        )
    pts = np.stack([xs + xa, ys + ya], axis=1).astype(np.float32)
    det = corners_to_obb(pts, conf=conf, source="aabb_refine")
    return det


class VialOBBDetector:
    """Oriented vial detector.

    Backend priority:
      1. Fine-tuned YOLOv8-OBB weights
      2. AABB YOLO (vial-live) refined to OBB via minAreaRect
      3. Classical PCA / minAreaRect OBB
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        *,
        camera_root: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        allow_classical_fallback: bool = True,
        allow_aabb_refine: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.allow_classical_fallback = bool(allow_classical_fallback)
        self.allow_aabb_refine = bool(allow_aabb_refine)
        self.camera_root = Path(camera_root) if camera_root else default_camera_root()
        self.device = device
        self.model = None
        self.aabb_model = None
        self.backend = "none"
        self.model_path: Optional[Path] = resolve_obb_weights(model_path, self.camera_root)
        self.aabb_path: Optional[Path] = None
        self._load()

    def _load(self) -> None:
        from ultralytics import YOLO

        if self.model_path is not None:
            try:
                self.model = YOLO(str(self.model_path))
                self.backend = "yolov8_obb"
            except Exception as exc:  # noqa: BLE001
                print(f"[r2_03_obb] OBB weights failed to load ({exc}); trying AABB refine")
                self.model = None
                self.model_path = None

        # Always try vial-live AABB alongside OBB — recovers upright when OBB is weak.
        if self.allow_aabb_refine:
            self.aabb_path = resolve_aabb_weights(self.camera_root)
            if self.aabb_path is not None:
                try:
                    self.aabb_model = YOLO(str(self.aabb_path))
                    if self.backend != "yolov8_obb":
                        self.backend = "aabb_refine"
                        self.model_path = self.aabb_path
                except Exception as exc:  # noqa: BLE001
                    print(f"[r2_03_obb] AABB YOLO failed ({exc})")
                    self.aabb_model = None

        if self.backend == "yolov8_obb":
            return
        if self.backend == "aabb_refine":
            return
        if self.allow_classical_fallback:
            self.backend = "classical_obb"
            self.model_path = None
            return
        raise FileNotFoundError("No OBB/AABB weights and classical fallback disabled")

    def _upright_aabb_dets(self, bgr: np.ndarray) -> List[OBBDet]:
        """vial-live AABB→OBB, upright-biased (suppress sideways-looking clutter)."""
        if self.aabb_model is None:
            return []
        dets = self._predict_aabb_refine(bgr)
        out: List[OBBDet] = []
        for d in dets:
            if d.conf < self.confidence_threshold:
                continue
            # Keep upright / unknown; skip wide horizontal boxes (sideways path owns those).
            if d.orientation == "sideways" and d.width / max(d.height, 1.0) >= 1.35:
                continue
            d.source = "vial_live_aabb"
            out.append(d)
        return out

    def predict(
        self,
        bgr: np.ndarray,
        depth_mm: Optional[np.ndarray] = None,
        intrinsics=None,
    ) -> Tuple[List[OBBDet], InferStats]:
        from .sideways_obb import detect_sideways_obb

        t0 = time.perf_counter()
        if self.backend == "yolov8_obb" and self.model is not None:
            dets = self._predict_obb(bgr)
            side = detect_sideways_obb(bgr, depth_mm, intrinsics)
            dets = _merge_obbs(dets, side, iou_thr=0.35)
            # Always fuse vial-live AABB→OBB for upright (Round-2 contract).
            upright = [
                d
                for d in self._upright_aabb_dets(bgr)
                if d.conf >= self.confidence_threshold
            ]
            dets = _merge_obbs(dets, upright, iou_thr=0.35)
            if not dets and self.allow_classical_fallback:
                dets = detect_classical_upright(bgr) or [
                    d
                    for d in detect_classical_obb(bgr)
                    if d.conf >= 0.55 and d.orientation == "upright"
                ]
            backend_name = "yolov8_obb+upright_aabb+sideways"
        elif self.backend == "aabb_refine" and self.aabb_model is not None:
            uprightish = self._upright_aabb_dets(bgr)
            side = detect_sideways_obb(bgr, depth_mm, intrinsics)
            dets = _merge_obbs(uprightish, side, iou_thr=0.35)
            if self.allow_classical_fallback and not any(
                d.orientation == "upright" for d in dets
            ):
                dets = _merge_obbs(dets, detect_classical_upright(bgr), iou_thr=0.35)
            if not dets and self.allow_classical_fallback:
                dets = [
                    d
                    for d in detect_classical_obb(bgr)
                    if d.conf >= 0.55 and d.orientation == "upright"
                ]
            backend_name = "aabb_refine+sideways_obb"
        else:
            side = detect_sideways_obb(bgr, depth_mm, intrinsics)
            dets = side if side else (
                detect_classical_obb(bgr) if self.allow_classical_fallback else []
            )
            backend_name = "sideways_obb" if side else self.backend

        ms = (time.perf_counter() - t0) * 1000.0
        stats = InferStats(
            backend=backend_name,
            model_path=str(self.model_path or ""),
            inference_ms=ms,
            num_dets=len(dets),
        )
        return dets, stats

    def _predict_obb(self, bgr: np.ndarray) -> List[OBBDet]:
        results = self.model.predict(
            bgr,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        out: List[OBBDet] = []
        if not results:
            return out
        r0 = results[0]
        if getattr(r0, "obb", None) is None:
            return out
        obb = r0.obb
        try:
            xyxyxyxy = obb.xyxyxyxy.cpu().numpy()
            confs = obb.conf.cpu().numpy()
            clss = obb.cls.cpu().numpy()
        except Exception:  # noqa: BLE001
            return out
        for corners, conf, cls_id in zip(xyxyxyxy, confs, clss):
            det = corners_to_obb(corners.reshape(4, 2), conf=float(conf), source="yolov8_obb")
            det.cls_id = int(cls_id)
            if det.conf >= self.confidence_threshold:
                out.append(det)
        return out

    def _predict_aabb_refine(self, bgr: np.ndarray) -> List[OBBDet]:
        results = self.aabb_model.predict(
            bgr,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        out: List[OBBDet] = []
        if not results or results[0].boxes is None:
            return out
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            if float(conf) < self.confidence_threshold:
                continue
            out.append(refine_aabb_to_obb(bgr, float(x1), float(y1), float(x2), float(y2), float(conf)))
        return out

    def annotate(self, bgr: np.ndarray, dets: Sequence[OBBDet]) -> np.ndarray:
        return annotate_obb(bgr, dets)


def _merge_obbs(primary: List[OBBDet], extra: List[OBBDet], *, iou_thr: float) -> List[OBBDet]:
    """Union detections; skip extras that overlap an existing OBB (IoU ≥ thr)."""
    from .obb_geometry import polygon_iou

    merged = list(primary)
    for e in extra:
        if any(polygon_iou(e, p) >= iou_thr for p in merged):
            continue
        merged.append(e)
    return merged
