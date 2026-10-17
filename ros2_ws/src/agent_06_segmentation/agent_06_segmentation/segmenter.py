"""YOLOv8-seg inference with weight resolution and mask-from-bbox fallback.

Weight strategy (first hit wins):
  1. Explicit ``model_path`` (seg or det checkpoint)
  2. ``runs/vial-seg/weights/best.pt`` (trained via scripts/train_seg.py)
  3. ``yolov8n-seg.pt`` / ``yolov8s-seg.pt`` under camera root (COCO-pretrained)
  4. Existing vial *detection* weights (vial-live → r1 → bon) — boxes only;
     masks synthesized from bbox (ellipse approximating the vial silhouette)
  5. Ultralytics auto-download of ``yolov8n-seg.pt``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CAMERA_ROOT_DEFAULT = Path(r"E:\Desktop\Jessica\robotics")

DET_WEIGHT_CANDIDATES = (
    Path("runs") / "vial-live" / "weights" / "best.pt",
    Path("runs") / "train-race" / "r1-winner" / "weights" / "best.pt",
    Path("runs") / "bon-winner" / "weights" / "best.pt",
    Path("yolov8s.pt"),
    Path("yolov8n.pt"),
)

SEG_WEIGHT_CANDIDATES = (
    Path("runs") / "vial-seg" / "weights" / "best.pt",
    Path("yolov8n-seg.pt"),
    Path("yolov8s-seg.pt"),
)


@dataclass
class InstanceDet:
    """One vial instance: axis-aligned box + optional binary mask."""

    x: int
    y: int
    w: int
    h: int
    conf: float
    class_id: int = 0
    class_name: str = "vial"
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    mask_source: str = "none"  # "network" | "bbox_ellipse" | "bbox_rect" | "none"

    def xyxy(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    def to_dict(self, *, include_mask_rle: bool = False) -> dict:
        d = {
            "x": int(self.x),
            "y": int(self.y),
            "w": int(self.w),
            "h": int(self.h),
            "conf": float(self.conf),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "mask_source": self.mask_source,
            "has_mask": self.mask is not None,
        }
        if include_mask_rle and self.mask is not None:
            d["mask_area"] = int(self.mask.sum())
        return d


def resolve_model_path(
    model_path: str | Path | None,
    camera_root: Path,
    *,
    prefer_seg: bool = True,
) -> Tuple[Path, str]:
    """Return (path, kind) where kind is 'seg' | 'det' | 'seg_pretrained'."""
    if model_path:
        p = Path(model_path)
        if p.is_file():
            kind = "seg" if _looks_like_seg(p) else "det"
            return p, kind

    if prefer_seg:
        for rel in SEG_WEIGHT_CANDIDATES:
            p = camera_root / rel
            if p.is_file():
                kind = "seg" if "seg" in p.name.lower() or "vial-seg" in str(p) else "seg"
                return p, kind

    for rel in DET_WEIGHT_CANDIDATES:
        p = camera_root / rel
        if p.is_file():
            return p, "det"

    # Ultralytics will download this name if absent.
    return Path("yolov8n-seg.pt"), "seg_pretrained"


def _looks_like_seg(path: Path) -> bool:
    name = path.name.lower()
    parts = {p.lower() for p in path.parts}
    return "seg" in name or "vial-seg" in parts or "segmentation" in parts


def bbox_to_ellipse_mask(
    h: int, w: int, x: int, y: int, bw: int, bh: int
) -> np.ndarray:
    """Filled ellipse inside the box — better vial silhouette than a rectangle."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if bw <= 0 or bh <= 0:
        return mask
    cx = int(x + bw / 2)
    cy = int(y + bh / 2)
    ax = max(1, int(bw / 2))
    ay = max(1, int(bh / 2))
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1, thickness=-1)
    return mask


def bbox_to_rect_mask(
    h: int, w: int, x: int, y: int, bw: int, bh: int
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(w, int(x + bw))
    y1 = min(h, int(y + bh))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 1
    return mask


def yolo_bbox_line_to_xywh(
    line: str, img_w: int, img_h: int
) -> Optional[Tuple[int, int, int, int, int]]:
    """Parse YOLO det (5 toks) or seg (class + polygon) → class, x, y, w, h."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    if len(parts) == 5:
        cx, cy, nw, nh = map(float, parts[1:5])
        bw = nw * img_w
        bh = nh * img_h
        x = cx * img_w - bw / 2
        y = cy * img_h - bh / 2
        return cls, int(round(x)), int(round(y)), int(round(bw)), int(round(bh))
    # Segmentation polygon: normalized x,y pairs
    coords = list(map(float, parts[1:]))
    if len(coords) < 6 or len(coords) % 2 != 0:
        return None
    xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
    ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return cls, int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0))


def polygon_mask_from_yolo_line(
    line: str, img_h: int, img_w: int
) -> Optional[np.ndarray]:
    parts = line.strip().split()
    if len(parts) <= 5:
        return None
    coords = list(map(float, parts[1:]))
    if len(coords) < 6 or len(coords) % 2 != 0:
        return None
    pts = np.array(
        [[coords[i] * img_w, coords[i + 1] * img_h] for i in range(0, len(coords), 2)],
        dtype=np.float32,
    )
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


class VialSegmenter:
    """Loads Ultralytics YOLO and returns instance boxes + masks."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        camera_root: str | Path = CAMERA_ROOT_DEFAULT,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        mask_fallback: str = "ellipse",  # ellipse | rect | none
        class_names: Sequence[str] = ("vial",),
    ) -> None:
        self.camera_root = Path(camera_root)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.mask_fallback = mask_fallback
        self.class_names = list(class_names)
        self.resolved_path, self.model_kind = resolve_model_path(
            model_path, self.camera_root, prefer_seg=True
        )
        self._model = None
        self._has_mask_head = self.model_kind in ("seg", "seg_pretrained")
        self._load_error: Optional[str] = None

    def load(self) -> None:
        from ultralytics import YOLO

        try:
            self._model = YOLO(str(self.resolved_path))
            # Probe task if possible
            task = getattr(self._model, "task", None)
            if task == "segment":
                self._has_mask_head = True
                if self.model_kind == "det":
                    self.model_kind = "seg"
            elif task == "detect":
                self._has_mask_head = False
                self.model_kind = "det"
        except Exception as exc:  # noqa: BLE001 — surface to node logger
            self._load_error = str(exc)
            self._model = None
            raise

    @property
    def ready(self) -> bool:
        return self._model is not None

    def predict(self, bgr: np.ndarray) -> List[InstanceDet]:
        if self._model is None:
            raise RuntimeError(f"Model not loaded: {self._load_error}")
        h, w = bgr.shape[:2]
        results = self._model.predict(
            source=bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return []
        r0 = results[0]
        dets: List[InstanceDet] = []
        boxes = getattr(r0, "boxes", None)
        masks = getattr(r0, "masks", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        mask_data = None
        if masks is not None and getattr(masks, "data", None) is not None:
            mask_data = masks.data.cpu().numpy()  # (N, mh, mw) float

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            x, y = int(round(x1)), int(round(y1))
            bw, bh = int(round(x2 - x1)), int(round(y2 - y1))
            cid = int(clss[i])
            cname = (
                self.class_names[cid]
                if 0 <= cid < len(self.class_names)
                else str(cid)
            )
            # Prefer network class names when available
            names = getattr(r0, "names", None) or getattr(self._model, "names", None)
            if isinstance(names, dict) and cid in names:
                cname = str(names[cid])

            mask = None
            source = "none"
            if mask_data is not None and i < len(mask_data):
                m = mask_data[i]
                if m.shape[0] != h or m.shape[1] != w:
                    m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                mask = (m > 0.5).astype(np.uint8)
                source = "network"
            elif self.mask_fallback == "ellipse":
                mask = bbox_to_ellipse_mask(h, w, x, y, bw, bh)
                source = "bbox_ellipse"
            elif self.mask_fallback == "rect":
                mask = bbox_to_rect_mask(h, w, x, y, bw, bh)
                source = "bbox_rect"

            dets.append(
                InstanceDet(
                    x=max(0, x),
                    y=max(0, y),
                    w=max(0, bw),
                    h=max(0, bh),
                    conf=float(confs[i]),
                    class_id=cid,
                    class_name=cname,
                    mask=mask,
                    mask_source=source,
                )
            )
        return dets

    def annotate(
        self,
        bgr: np.ndarray,
        dets: Sequence[InstanceDet],
        *,
        alpha: float = 0.45,
    ) -> np.ndarray:
        out = bgr.copy()
        overlay = out.copy()
        palette = [
            (40, 180, 80),
            (0, 140, 255),
            (220, 80, 60),
            (200, 180, 40),
            (180, 80, 200),
        ]
        for i, d in enumerate(dets):
            color = palette[i % len(palette)]
            if d.mask is not None and d.mask.shape[:2] == out.shape[:2]:
                overlay[d.mask.astype(bool)] = color
            x1, y1, x2, y2 = d.xyxy()
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{d.class_name} {d.conf:.2f} [{d.mask_source}]"
            cv2.putText(
                out,
                label,
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)
        return out


def detections_to_json(dets: Sequence[InstanceDet], **meta: Any) -> str:
    payload = {
        "detections": [d.to_dict() for d in dets],
        "count": len(dets),
        **meta,
    }
    return json.dumps(payload)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_b = a.astype(bool)
    b_b = b.astype(bool)
    inter = np.logical_and(a_b, b_b).sum()
    union = np.logical_or(a_b, b_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def box_iou_xywh(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)
