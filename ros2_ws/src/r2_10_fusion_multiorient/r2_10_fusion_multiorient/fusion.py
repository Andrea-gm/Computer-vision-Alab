"""End-to-end multi-orientation fusion: YOLO-TTA ∪ classical ∪ refine → NMS."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np

from .classical import propose_sideways, refine_cap_axis
from .nms import orientation_aware_nms
from .types import Det, classify_orientation
from .yolo_tta import YoloTTADetector, annotate, default_camera_root, resolve_model_path


@dataclass
class FusionConfig:
    camera_root: Optional[str] = None
    model_path: Optional[str] = None
    yolo_conf: float = 0.25
    yolo_iou: float = 0.45
    imgsz: int = 640
    device: Optional[str] = None
    use_yolo: bool = True
    use_yolo_tta: bool = True
    tta_angles: Sequence[int] = (0, 90, 180, 270)
    use_classical_upright: bool = True
    use_classical_sideways: bool = True
    # RGB-only Method-I / classical sideways without depth — unsafe on upright stills.
    # Audit ERROR_AUDIT.md: disable in production; keep Method-I when depth is valid.
    enable_rgb_classical_sideways: bool = False
    use_cap_refine: bool = True
    nms_iou: float = 0.45
    max_det: int = 3
    min_conf_final: float = 0.20
    # Floor applied to all YOLO proposals (kills empty-scene clutter ~0.29).
    empty_scene_yolo_floor: float = 0.35
    # After Method-I finds a depth-backed vial, extra non-overlapping YOLO boxes
    # need this conf OR depth/body support (audit: removes 22 hard-sideways FPs).
    extra_yolo_conf_after_method_i: float = 0.60
    # When Method-I is empty with depth, keep upright/top-down YOLO at this conf.
    empty_method_i_keep_conf: float = 0.50
    extras: dict[str, Any] = field(default_factory=dict)


def _align_depth_roi(
    depth_mm: Optional[np.ndarray], w: int, h: int
) -> Optional[np.ndarray]:
    if depth_mm is None or getattr(depth_mm, "size", 0) == 0:
        return None
    d = np.asarray(depth_mm, dtype=np.float32)
    if d.shape[:2] != (h, w):
        try:
            import cv2

            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
        except Exception:  # noqa: BLE001
            return None
    return d


def _box_has_depth_body_support(
    depth_mm: Optional[np.ndarray],
    det: Det,
    frame_w: int,
    frame_h: int,
    *,
    min_invalid: float = 0.15,
) -> bool:
    """True if the box ROI shows clear-vial-like invalid/missing depth."""
    depth = _align_depth_roi(depth_mm, frame_w, frame_h)
    if depth is None:
        return False
    x1 = int(max(0, min(frame_w - 1, round(det.x1))))
    y1 = int(max(0, min(frame_h - 1, round(det.y1))))
    x2 = int(max(x1 + 1, min(frame_w, round(det.x2))))
    y2 = int(max(y1 + 1, min(frame_h, round(det.y2))))
    roi = depth[y1:y2, x1:x2]
    if roi.size < 25:
        return False
    invalid = float((roi <= 1.0).mean())
    return invalid >= min_invalid


def _is_upright_or_topdown(det: Det) -> bool:
    orient = det.orientation if det.orientation != "unknown" else classify_orientation(det)
    return orient in ("upright", "topdown")


class FusionVialDetector:
    """Production multi-orientation vial detector."""

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        self.config = config or FusionConfig()
        root = self.config.camera_root or str(default_camera_root())
        self.camera_root = Path(root)
        angles = self.config.tta_angles if self.config.use_yolo_tta else (0,)
        self.yolo: Optional[YoloTTADetector] = None
        if self.config.use_yolo:
            self.yolo = YoloTTADetector(
                self.config.model_path,
                camera_root=self.camera_root,
                confidence_threshold=self.config.yolo_conf,
                iou_threshold=self.config.yolo_iou,
                imgsz=self.config.imgsz,
                device=self.config.device,
                tta_angles=angles,
            )
        self.model_path = (
            str(self.yolo.model_path)
            if self.yolo and self.yolo.model_path
            else resolve_model_path(self.config.model_path, self.camera_root)
        )
        self.backend = "fusion"
        if self.yolo and self.yolo.available:
            self.backend = "fusion_yolo_tta" if self.config.use_yolo_tta else "fusion_yolo"
        else:
            self.backend = "fusion_classical"

    def detect(
        self,
        bgr: np.ndarray,
        depth_mm: Optional[np.ndarray] = None,
    ) -> List[Det]:
        if bgr is None or getattr(bgr, "size", 0) == 0:
            return []
        cfg = self.config
        proposals: List[Det] = []
        frame_h, frame_w = bgr.shape[:2]

        from .types import box_iou

        yolo_dets: List[Det] = []
        if self.yolo is not None and self.yolo.available:
            yolo_dets = self.yolo.predict(bgr)
            yolo_keep = max(cfg.yolo_conf, cfg.empty_scene_yolo_floor)
            yolo_dets = [d for d in yolo_dets if d.conf >= yolo_keep]

        # Method-I is calibrated with depth (0 FP on synth). Without depth it
        # over-fires on real upright stills and must not override YOLO.
        has_depth = depth_mm is not None and getattr(depth_mm, "size", 0) > 0
        side_dets: List[Det] = []
        if cfg.use_classical_sideways:
            if has_depth:
                side_dets = propose_sideways(bgr, depth_mm)
            elif cfg.enable_rgb_classical_sideways and not yolo_dets:
                side_dets = propose_sideways(bgr, None)

        if side_dets:
            proposals.extend(side_dets)
            # Audit: after a depth-backed Method-I hit, do not keep every
            # non-overlapping YOLO box at conf>=0.35 — require depth/body
            # support or conf >= extra_yolo_conf_after_method_i (default 0.60).
            for yd in yolo_dets:
                if any(box_iou(yd, sd) >= 0.15 for sd in side_dets):
                    continue
                if yd.orientation == "unknown":
                    yd.orientation = classify_orientation(yd)
                depth_ok = _box_has_depth_body_support(
                    depth_mm, yd, frame_w, frame_h
                )
                if depth_ok or yd.conf >= cfg.extra_yolo_conf_after_method_i:
                    proposals.append(yd)
        else:
            if has_depth:
                # Pose-aware empty veto: Method-I-empty vetoes weak *sideways*
                # hypotheses only; keep confident upright / top-down YOLO.
                kept: List[Det] = []
                for d in yolo_dets:
                    if d.orientation == "unknown":
                        d.orientation = classify_orientation(d)
                    if not _is_upright_or_topdown(d):
                        continue
                    if d.conf >= cfg.empty_method_i_keep_conf:
                        kept.append(d)
                proposals.extend(kept)
                # Do NOT run classical upright — it FPs on empty desks with depth.
            else:
                # YOLO only without depth. RGB classical upright on empty YOLO
                # FPs empty desks (ERROR_AUDIT vial_0191 after RGB sideways off).
                proposals.extend(yolo_dets)

        for d in proposals:
            if d.orientation == "unknown":
                d.orientation = classify_orientation(d)

        fused = orientation_aware_nms(
            proposals,
            iou_thr=cfg.nms_iou,
            max_det=cfg.max_det,
            prefer_sources=(
                "classical_sideways",
                "yolo",
                "yolo_tta90",
                "yolo_tta270",
                "yolo_tta180",
                "classical_upright",
            ),
        )

        if cfg.use_cap_refine and fused:
            refined: List[Det] = []
            for d in fused:
                if "classical" in d.source:
                    refined.append(d)
                    continue
                try:
                    refined.append(refine_cap_axis(bgr, d, depth_mm))
                except Exception:  # noqa: BLE001
                    refined.append(d)
            fused = orientation_aware_nms(
                refined,
                iou_thr=cfg.nms_iou,
                max_det=cfg.max_det,
            )

        fused = [d for d in fused if d.conf >= cfg.min_conf_final]
        # Soft shape gate: reject tiny / wall-strip boxes.
        h, w = bgr.shape[:2]
        area = float(h * w)
        cleaned: List[Det] = []
        for d in fused:
            bw = max(1.0, d.x2 - d.x1)
            bh = max(1.0, d.y2 - d.y1)
            frac = (bw * bh) / area
            hw = bh / bw
            if frac < 0.001 or frac > 0.85:
                continue
            if hw < 0.28 or hw > 2.8:
                continue
            cleaned.append(d)
        return cleaned

    def annotate(self, bgr: np.ndarray, dets: Sequence[Det]) -> np.ndarray:
        return annotate(bgr, dets)
