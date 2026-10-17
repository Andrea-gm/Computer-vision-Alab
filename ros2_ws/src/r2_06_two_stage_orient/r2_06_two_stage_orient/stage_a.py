"""Stage A — robust multi-source vial proposals.

Round-1 miss root causes this stage fixes:
  - Preferring COCO ``yolov8n-seg`` over vial-live det weights (agent_06).
  - Downward-only cap→body expansion (agent_02/04/05 classical paths).
  - No rotation TTA → sideways vials never proposed by upright-trained YOLO.
  - Over-tight conf / aspect gates that dropped easy mid-field upright vials.

Sources merged by NMS:
  1. YOLO vial-live (or r1/bon) at recall-oriented conf, plus 90° rotation TTA.
  2. Cap-seeded axis-agnostic expansions (down / up / left / right).
  3. Optional sideways_race Method-I when depth is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .types import Proposal, clamp_box, nms_xywh

CAMERA_ROOT_DEFAULT = Path(r"E:\Desktop\Jessica\robotics")

DET_WEIGHT_CANDIDATES = (
    Path("runs") / "vial-live" / "weights" / "best.pt",
    Path("runs") / "train-race" / "r1-winner" / "weights" / "best.pt",
    Path("runs") / "bon-winner" / "weights" / "best.pt",
    Path("yolov8s.pt"),
    Path("yolov8n.pt"),
)


def resolve_det_weights(
    explicit: Optional[Union[str, Path]],
    camera_root: Path,
) -> Optional[Path]:
    """Prefer vial *detection* weights. Never auto-pick COCO-seg."""
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
    for rel in DET_WEIGHT_CANDIDATES:
        p = camera_root / rel
        if p.is_file():
            return p
    return None


def white_cap_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Slightly looser than round-1 classical defaults to recover easy upright lids.
    mask = ((hsv[:, :, 2] >= 175) & (hsv[:, :, 1] <= 60)).astype(np.uint8) * 255
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_m = (
        (lab[:, :, 0] >= 185)
        & (np.abs(lab[:, :, 1].astype(np.int16) - 128) <= 14)
        & (np.abs(lab[:, :, 2].astype(np.int16) - 128) <= 14)
    ).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask, lab_m)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def _body_support(gray: np.ndarray, edges: np.ndarray, box: Tuple[int, int, int, int]) -> float:
    """How much non-cap body evidence sits inside a candidate AABB."""
    x, y, bw, bh = box
    if bw <= 2 or bh <= 2:
        return 0.0
    roi_e = edges[y : y + bh, x : x + bw]
    roi_g = gray[y : y + bh, x : x + bw]
    if roi_e.size == 0:
        return 0.0
    edge_frac = float(cv2.countNonZero(roi_e)) / float(roi_e.size)
    # Mild gradient energy (clear cup often has rim contrast).
    gx = cv2.Sobel(roi_g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_g, cv2.CV_32F, 0, 1, ksize=3)
    mag = float(np.mean(np.sqrt(gx * gx + gy * gy))) / 255.0
    return float(np.clip(0.55 * edge_frac * 8.0 + 0.45 * mag * 3.0, 0.0, 1.0))


def _cap_proposals(bgr: np.ndarray) -> List[Proposal]:
    """Expand each white-cap blob; keep the best 1–2 axis directions (not all four)."""
    h, w = bgr.shape[:2]
    mask = white_cap_mask(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = float(max(1, h * w))
    out: List[Proposal] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        frac = area / area_img
        # Caps are compact; large white walls/desk slabs are the empty-frame FP source.
        if frac < 0.0005 or frac > 0.045:
            continue
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        if cw < 6 or ch < 6:
            continue
        if max(cw, ch) > 0.22 * min(h, w):
            continue
        roi = mask[cy : cy + ch, cx : cx + cw]
        fill = float(cv2.countNonZero(roi)) / float(max(1, cw * ch))
        if fill < 0.40:
            continue

        diam = float(max(cw, ch))
        body = max(diam * 1.05, min(cw, ch) * 1.7)
        # Physical prior: full vial AABB should stay mid-scale in these frames.
        max_box = 0.18 * area_img
        circ = 4.0 * np.pi * area / max(1.0, cv2.arcLength(cnt, True) ** 2)
        base = float(
            np.clip(0.20 + 0.30 * fill + 0.20 * min(1.0, circ) + 0.08 * min(1.0, frac * 250), 0.12, 0.70)
        )

        dirs = (
            ("cap_down", cx - int(0.15 * cw), cy - int(0.08 * ch), int(cw * 1.3), int(ch + body)),
            ("cap_up", cx - int(0.15 * cw), cy - int(body), int(cw * 1.3), int(ch + body)),
            ("cap_right", cx - int(0.08 * cw), cy - int(0.2 * ch), int(cw + body), int(ch * 1.4)),
            ("cap_left", cx - int(body), cy - int(0.2 * ch), int(cw + body), int(ch * 1.4)),
        )
        scored_dirs: List[Proposal] = []
        for src, x, y, bw, bh in dirs:
            x, y, bw, bh = clamp_box(x, y, bw, bh, w, h)
            if bw * bh > max_box:
                continue
            aspect = bh / float(max(1, bw))
            if aspect < 0.40 or aspect > 2.8:
                continue
            support = _body_support(gray, edges, (x, y, bw, bh))
            if support < 0.12:
                continue
            score = base + 0.40 * support
            if src in ("cap_down", "cap_up") and aspect >= 0.9:
                score += 0.04
            if src in ("cap_left", "cap_right") and aspect <= 0.9:
                score += 0.04
            if cw >= ch * 1.2 and src in ("cap_left", "cap_right"):
                score += 0.06
            if ch >= cw * 1.15 and src in ("cap_down", "cap_up"):
                score += 0.06
            scored_dirs.append(Proposal(x, y, bw, bh, float(min(0.82, score)), source=src))

        scored_dirs.sort(key=lambda p: p.score, reverse=True)
        # Keep only the single best axis per cap (cuts empty FP duplication).
        if scored_dirs:
            out.append(scored_dirs[0])
    return out


def _map_box_from_rotated(
    x1: float, y1: float, x2: float, y2: float, angle: int, w: int, h: int
) -> Tuple[int, int, int, int]:
    """Map xyxy from a cv2.rotate'd image back to the original frame as xywh.

    OpenCV conventions (x=col, y=row), original size (w, h):
      90 CW:  (x,y) -> (h-1-y, x)     rotated size (h, w)
      180:    (x,y) -> (w-1-x, h-1-y)
      90 CCW: (x,y) -> (y, w-1-x)     (== 270 CW)
    """
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    mapped = []
    if angle == 0:
        mapped = corners
    elif angle == 90:  # ROTATE_90_CLOCKWISE inverse
        for ox, oy in corners:
            # forward x'=h-1-y, y'=x  →  x=oy, y=h-1-ox
            mapped.append((oy, h - 1 - ox))
    elif angle == 180:
        for ox, oy in corners:
            mapped.append((w - 1 - ox, h - 1 - oy))
    elif angle == 270:  # ROTATE_90_COUNTERCLOCKWISE inverse
        for ox, oy in corners:
            # forward x'=y, y'=w-1-x  →  x=w-1-oy, y=ox
            mapped.append((w - 1 - oy, ox))
    else:
        mapped = corners
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    bx1, by1 = min(xs), min(ys)
    bx2, by2 = max(xs), max(ys)
    return clamp_box(
        int(round(bx1)),
        int(round(by1)),
        int(round(bx2 - bx1)),
        int(round(by2 - by1)),
        w,
        h,
    )


class StageAProposer:
    """Generate recall-oriented vial box proposals."""

    def __init__(
        self,
        *,
        camera_root: Union[str, Path] = CAMERA_ROOT_DEFAULT,
        model_path: Optional[Union[str, Path]] = None,
        conf: float = 0.12,
        iou: float = 0.45,
        imgsz: int = 640,
        use_yolo: bool = True,
        use_rotation_tta: bool = True,
        use_cap_proposals: bool = True,
        use_sideways_method: bool = True,
        nms_iou: float = 0.50,
        max_proposals: int = 12,
        device: Optional[str] = None,
    ) -> None:
        self.camera_root = Path(camera_root)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.use_yolo = bool(use_yolo)
        self.use_rotation_tta = bool(use_rotation_tta)
        self.use_cap_proposals = bool(use_cap_proposals)
        self.use_sideways_method = bool(use_sideways_method)
        self.nms_iou = float(nms_iou)
        self.max_proposals = int(max_proposals)
        self.device = device
        self.weights = resolve_det_weights(model_path, self.camera_root)
        self._model = None
        self._yolo_error: Optional[str] = None
        self.backend = "cap_only"

    def load(self) -> None:
        if not self.use_yolo:
            self.backend = "cap_only"
            return
        if self.weights is None:
            self.backend = "cap_only"
            self._yolo_error = "no vial det weights found"
            return
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights))
            self.backend = "yolo+cap"
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self._yolo_error = str(exc)
            self.backend = "cap_only"

    def _yolo_on(self, bgr: np.ndarray) -> List[Proposal]:
        if self._model is None:
            return []
        kwargs = dict(
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        if self.device:
            kwargs["device"] = self.device
        results = self._model.predict(bgr, **kwargs)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        h, w = bgr.shape[:2]
        out: List[Proposal] = []
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            x, y, bw, bh = clamp_box(
                int(round(x1)),
                int(round(y1)),
                int(round(x2 - x1)),
                int(round(y2 - y1)),
                w,
                h,
            )
            # YOLO on real desk imagery must beat classical caps in NMS.
            out.append(Proposal(x, y, bw, bh, float(min(0.99, 0.86 + 0.13 * c)), source="yolo"))
        return out

    def _yolo_tta(self, bgr: np.ndarray) -> List[Proposal]:
        h, w = bgr.shape[:2]
        props = list(self._yolo_on(bgr))
        if not self.use_rotation_tta or self._model is None:
            return props
        # Native hit usually means upright desk vial — skip costly TTA.
        # Run rotations only when native YOLO misses (sideways / odd pose).
        if props:
            return props
        rot_flags = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        for angle, flag in rot_flags.items():
            rotated = cv2.rotate(bgr, flag)
            for p in self._yolo_on(rotated):
                x1, y1, x2, y2 = p.as_xyxy()
                x, y, bw, bh = _map_box_from_rotated(x1, y1, x2, y2, angle, w, h)
                props.append(
                    Proposal(x, y, bw, bh, float(p.score) * 0.88, source=f"yolo_rot{angle}")
                )
        return props

    def _sideways_method(
        self, bgr: np.ndarray, depth_mm: Optional[np.ndarray], intrinsics: Optional[dict]
    ) -> List[Proposal]:
        if not self.use_sideways_method:
            return []
        try:
            import sys

            methods = self.camera_root / "sideways_race" / "methods"
            race_root = self.camera_root / "sideways_race"
            for p in (str(methods), str(race_root), str(self.camera_root)):
                if p not in sys.path:
                    sys.path.insert(0, p)
            from method_i_hard_robust import detect_sideways  # type: ignore
        except Exception:
            return []
        # Method-I is a sideways RGB-D specialist. On RGB-only desk stills (no
        # depth / all-zero depth) it floods upright frames with false boxes and
        # crowds out YOLO — only run when depth has real metric content.
        if depth_mm is None:
            return []
        depth = depth_mm.astype(np.float32)
        valid = (depth > 80.0) & (depth < 5000.0)
        if int(valid.sum()) < 500:
            return []
        try:
            boxes = detect_sideways(bgr, depth, intrinsics)
        except Exception:
            return []
        out: List[Proposal] = []
        for b in boxes or []:
            x, y, bw, bh = [int(v) for v in b[:4]]
            x, y, bw, bh = clamp_box(x, y, bw, bh, bgr.shape[1], bgr.shape[0])
            # Race winner — must beat directional cap proposals in NMS.
            out.append(Proposal(x, y, bw, bh, 0.93, source="sideways_I"))
        return out

    def propose(
        self,
        bgr: np.ndarray,
        depth_mm: Optional[np.ndarray] = None,
        intrinsics: Optional[dict] = None,
    ) -> List[Proposal]:
        if self.use_yolo and self._model is None and self._yolo_error is None:
            self.load()

        props: List[Proposal] = []
        if self.use_yolo:
            props.extend(self._yolo_tta(bgr))
        yolo_hit = any(p.source == "yolo" for p in props)
        if self.use_cap_proposals:
            caps = _cap_proposals(bgr)
            if yolo_hit:
                # Keep classical caps only when they look sideways (YOLO covers upright).
                caps = [
                    p
                    for p in caps
                    if p.source in ("cap_left", "cap_right") and p.h < p.w * 0.95
                ]
            props.extend(caps)
        props.extend(self._sideways_method(bgr, depth_mm, intrinsics))

        if not props:
            return []
        boxes = [p.as_xywh() for p in props]
        scores = [p.score for p in props]
        keep = nms_xywh(boxes, scores, iou_thr=self.nms_iou, max_dets=self.max_proposals)
        return [props[i] for i in keep]
