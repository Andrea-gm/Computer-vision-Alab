"""SAHI / tiled + multi-scale YOLO vial detector with rotation TTA.

Modes
-----
- ``baseline``: single full-frame YOLO pass (imgsz fixed).
- ``sahi``: full-frame + overlapping tiles + optional image scales + rotation TTA.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union
import time

import numpy as np

from .tiling import (
    crop_slice,
    generate_slices,
    remap_dets_from_slice,
    rotate_image,
    scale_image,
    unrotate_det,
    unscale_det,
)
from .types import Det, InferStats, annotate, nms
from .weights import default_camera_root, resolve_model_path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class SahiMultiscaleDetector:
    """Ultralytics YOLO wrapped with SAHI tiling, multi-scale, and rotation TTA."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        *,
        camera_root: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        mode: str = "sahi",
        slice_width: int = 320,
        slice_height: int = 320,
        overlap_ratio: float = 0.25,
        image_scales: Optional[Sequence[float]] = None,
        rotation_tta: bool = True,
        rotation_angles: Optional[Sequence[int]] = None,
        include_full_frame: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.mode = (mode or "sahi").strip().lower()
        if self.mode not in ("baseline", "sahi", "single", "tiled"):
            raise ValueError(f"Unknown mode {mode!r}; use baseline or sahi")
        if self.mode in ("single",):
            self.mode = "baseline"
        if self.mode == "tiled":
            self.mode = "sahi"
        self.slice_width = int(slice_width)
        self.slice_height = int(slice_height)
        self.overlap_ratio = float(overlap_ratio)
        self.image_scales = list(image_scales) if image_scales is not None else [1.0, 1.5]
        self.rotation_tta = bool(rotation_tta)
        # 0 = identity, 1 = CCW90 (sideways→upright), 3 = CW90
        self.rotation_angles = (
            list(rotation_angles) if rotation_angles is not None else [0, 1, 3]
        )
        self.include_full_frame = bool(include_full_frame)
        # Tile only on identity (+ optional 90°) at scale 1.0 to cap FPS cost.
        self.tile_rotations: List[int] = [0, 1]
        self.tile_scales: List[float] = [1.0]
        self.device = device
        self.camera_root = Path(camera_root) if camera_root else default_camera_root()
        self.model_path: Optional[Path] = resolve_model_path(model_path, self.camera_root)
        self.model = None
        self.backend = "none"
        self._load()

    def _load(self) -> None:
        if self.model_path is None:
            raise FileNotFoundError(
                "No YOLO weights found under Camera/runs or Camera/*.pt. "
                "Expected vial-live best.pt."
            )
        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))
        self.backend = "yolo"

    def predict(self, bgr: np.ndarray) -> tuple[List[Det], InferStats]:
        t0 = time.perf_counter()
        if self.mode == "baseline":
            dets, n_slices, n_passes = self._predict_baseline(bgr)
            mode_name = "baseline"
        else:
            dets, n_slices, n_passes = self._predict_sahi(bgr)
            mode_name = "sahi"
        ms = (time.perf_counter() - t0) * 1000.0
        stats = InferStats(
            backend=self.backend,
            model_path=str(self.model_path or ""),
            mode=mode_name,
            inference_ms=ms,
            num_dets=len(dets),
            num_slices=n_slices,
            num_passes=n_passes,
        )
        return dets, stats

    def _predict_baseline(self, bgr: np.ndarray) -> tuple[List[Det], int, int]:
        dets = self._infer_array(bgr)
        return dets, 0, 1

    def _predict_sahi(self, bgr: np.ndarray) -> tuple[List[Det], int, int]:
        """Full-frame multi-scale + rotation TTA, plus identity/90° tiles at 1.0×."""
        h0, w0 = bgr.shape[:2]
        all_dets: List[Det] = []
        n_slices = 0
        n_passes = 0

        angles = self.rotation_angles if self.rotation_tta else [0]

        # --- Pass A: full-frame multi-scale × rotation TTA ---
        if self.include_full_frame:
            for k90 in angles:
                rotated = rotate_image(bgr, k90)
                for scale in self.image_scales:
                    scaled = scale_image(rotated, float(scale))
                    full_dets = self._infer_array(scaled)
                    full_dets = [unscale_det(d, float(scale)) for d in full_dets]
                    if k90:
                        full_dets = [unrotate_det(d, k90, w0, h0) for d in full_dets]
                    all_dets.extend(full_dets)
                    n_passes += 1

        # --- Pass B: overlapping tiles (limited rotations/scales for FPS) ---
        tile_angles = [a for a in self.tile_rotations if a in angles] if self.rotation_tta else [0]
        for k90 in tile_angles:
            rotated = rotate_image(bgr, k90)
            for scale in self.tile_scales:
                scaled = scale_image(rotated, float(scale))
                sh, sw = scaled.shape[:2]
                slices = generate_slices(
                    sw, sh, self.slice_width, self.slice_height, self.overlap_ratio
                )
                for sl in slices:
                    if sl.w >= sw and sl.h >= sh and self.include_full_frame:
                        continue  # duplicate of full-frame
                    crop = crop_slice(scaled, sl)
                    tile_dets = self._infer_array(crop)
                    tile_dets = remap_dets_from_slice(tile_dets, sl)
                    tile_dets = [unscale_det(d, float(scale)) for d in tile_dets]
                    if k90:
                        tile_dets = [unrotate_det(d, k90, w0, h0) for d in tile_dets]
                    all_dets.extend(tile_dets)
                    n_slices += 1
                    n_passes += 1

        clamped: List[Det] = []
        for d in all_dets:
            x1 = float(np.clip(d.x1, 0, w0 - 1))
            y1 = float(np.clip(d.y1, 0, h0 - 1))
            x2 = float(np.clip(d.x2, 0, w0 - 1))
            y2 = float(np.clip(d.y2, 0, h0 - 1))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            clamped.append(Det(x1, y1, x2, y2, d.conf, d.cls_id, d.label, d.source))
        # Slightly looser merge IoU so a strong tile box is not killed by a
        # higher-conf but poorer full-frame hypothesis.
        merge_iou = min(0.65, max(self.iou_threshold, 0.5))
        merged = nms(clamped, merge_iou)
        return merged, n_slices, n_passes

    def _infer_array(self, bgr: np.ndarray) -> List[Det]:
        assert self.model is not None
        kwargs = dict(
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(bgr, **kwargs)
        dets: List[Det] = []
        if not results:
            return dets
        r0 = results[0]
        names = getattr(r0, "names", None) or getattr(self.model, "names", {}) or {0: "vial"}
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return dets
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            cls_id = int(clss[i]) if len(clss) else 0
            label = str(names.get(cls_id, "vial") if isinstance(names, dict) else "vial")
            dets.append(
                Det(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    conf=float(confs[i]),
                    cls_id=cls_id,
                    label=label,
                    source="yolo",
                )
            )
        return dets

    def annotate(self, bgr: np.ndarray, dets: Sequence[Det]) -> np.ndarray:
        return annotate(bgr, list(dets), color=(0, 210, 140))


def list_images(folder: Union[str, Path]) -> List[Path]:
    root = Path(folder)
    if not root.exists():
        return []
    files = [
        p
        for p in sorted(root.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    # Prefer *_color.png when present (sideways_race/synth layout).
    color = [p for p in files if p.stem.endswith("_color")]
    return color if color else files


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
        dets.append(
            Det(x1, y1, x2, y2, conf=1.0, cls_id=cls_id, label="vial", source="gt")
        )
    return dets
