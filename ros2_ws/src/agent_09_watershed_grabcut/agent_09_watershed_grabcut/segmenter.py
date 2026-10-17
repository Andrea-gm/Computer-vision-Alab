"""Automated watershed + GrabCut vial instance segmentation.

Pipeline (no manual clicks):
  1. Adaptive white-cap mask (opaque screw lid: high V, low S).
  2. Distance-transform peaks → watershed markers (instance seeds).
  3. Grow clear-cup extent from each seed using edge / saturation cues.
  4. GrabCut refine each ROI (rect then mask init with sure-FG on the lid).
  5. Geometric + lid-top scoring → NMS → instance boxes.

Suited to single upright vials with a white lid. Weak on: cluttered desks with
many bright objects, transparent body vs light background, empty/no-lid frames.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Det:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int = 0
    label: str = "vial"
    source: str = "watershed_grabcut"


@dataclass
class InferStats:
    backend: str
    inference_ms: float
    n_seeds: int = 0
    n_markers: int = 0


def det_to_dict(d: Det) -> Dict:
    return {
        "x1": d.x1,
        "y1": d.y1,
        "x2": d.x2,
        "y2": d.y2,
        "confidence": d.conf,
        "class_id": d.cls_id,
        "label": d.label,
        "source": d.source,
    }


def list_images(folder: Path) -> List[Path]:
    if not folder or not Path(folder).exists():
        return []
    root = Path(folder)
    imgs = [p for p in sorted(root.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if imgs:
        return imgs
    # YOLO split layout: images/{split}
    nested: List[Path] = []
    for sub in ("images/test", "images/val", "images/train", "images"):
        d = root / sub
        if d.is_dir():
            nested.extend(p for p in sorted(d.rglob("*")) if p.suffix.lower() in IMAGE_EXTS)
    return nested


def box_iou(a: Det, b: Det) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    bb = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    return float(inter / (aa + bb - inter + 1e-9))


def load_yolo_label(path: Path, width: int, height: int) -> List[Det]:
    if not path.exists():
        return []
    dets: List[Det] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height
        dets.append(Det(x1, y1, x2, y2, 1.0, cls_id=cls_id, source="gt"))
    return dets


def _white_mask(value: np.ndarray, saturation: np.ndarray, vmin: int) -> np.ndarray:
    return ((value >= vmin) & (saturation <= 55)).astype(np.uint8) * 255


def _adaptive_cap_mask(value: np.ndarray, saturation: np.ndarray) -> np.ndarray:
    """Pick a white threshold whose fill fraction looks like lids, not the whole desk."""
    best = _white_mask(value, saturation, 200)
    best_frac = float(best.mean() / 255.0)
    if 0.002 <= best_frac <= 0.22:
        return best
    for vmin in (185, 170, 210, 220, 230):
        trial = _white_mask(value, saturation, vmin)
        frac = float(trial.mean() / 255.0)
        if 0.002 <= frac <= 0.22 and abs(frac - 0.06) < abs(best_frac - 0.06):
            best = trial
            best_frac = frac
    return best


def _nms(dets: Sequence[Det], iou_thr: float, max_dets: int) -> List[Det]:
    ordered = sorted(dets, key=lambda d: d.conf, reverse=True)
    keep: List[Det] = []
    for d in ordered:
        if any(box_iou(d, k) > iou_thr for k in keep):
            continue
        keep.append(d)
        if len(keep) >= max_dets:
            break
    return keep


def _grow_cup(
    edges: np.ndarray,
    saturation: np.ndarray,
    value: np.ndarray,
    x: int,
    y: int,
    bw: int,
    bh: int,
) -> Tuple[int, int, int, int]:
    """Expand from lid box along the clearer cup (usually below)."""
    h, w = edges.shape[:2]
    cx = x + bw * 0.5
    max_cup = int(max(bh * 3.0, bw * 2.4, 0.22 * h))

    def row_ok(r: int) -> bool:
        x0 = max(0, int(cx - bw * 0.6))
        x1 = min(w, int(cx + bw * 0.6))
        if x1 <= x0:
            return False
        e = float((edges[r, x0:x1] > 0).mean())
        sat = float(saturation[r, x0:x1].mean())
        return e > 0.035 or sat < 95

    below = 0
    gap = 0
    for r in range(y + bh, min(h, y + bh + max_cup)):
        if row_ok(r):
            below += 1
            gap = 0
        else:
            gap += 1
            if below > 0 and gap <= 2:
                below += 1
            elif below >= max(8, int(bw * 0.28)):
                break
            elif below == 0 and gap > 1:
                break

    above = 0
    gap = 0
    for r in range(y - 1, max(-1, y - max_cup), -1):
        if row_ok(r):
            above += 1
            gap = 0
        else:
            gap += 1
            if above > 0 and gap <= 2:
                above += 1
            elif above >= max(8, int(bw * 0.28)):
                break
            elif above == 0 and gap > 1:
                break

    pad = max(2, int(bw * 0.1))
    if below >= above and below >= max(6, int(bw * 0.25)):
        return (
            max(0, x - pad),
            max(0, y - pad),
            min(w - 1, x + bw + pad),
            min(h - 1, y + bh + below + pad),
        )
    if above > below and above >= max(6, int(bw * 0.25)):
        return (
            max(0, x - pad),
            max(0, y - above - pad),
            min(w - 1, x + bw + pad),
            min(h - 1, y + bh + pad),
        )
    # Geometric prior: lid ~ top third of upright vial.
    return (
        max(0, x - pad),
        max(0, y - pad),
        min(w - 1, x + bw + pad),
        min(h - 1, y + int(max(bh * 2.5, bw * 1.35)) + pad),
    )


def _grabcut_refine(
    bgr: np.ndarray,
    rect: Tuple[int, int, int, int],
    sure_fg_mask: np.ndarray,
    iters: int,
) -> Optional[Tuple[int, int, int, int]]:
    h, w = bgr.shape[:2]
    rx, ry, rw, rh = rect
    if rw < 10 or rh < 12:
        return None
    gc = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc, (rx, ry, rw, rh), bgd, fgd, max(1, iters), cv2.GC_INIT_WITH_RECT)
        if sure_fg_mask is not None and sure_fg_mask.any():
            gc[sure_fg_mask > 0] = cv2.GC_FGD
            cv2.grabCut(
                bgr, gc, None, bgd, fgd, max(1, iters - 1), cv2.GC_INIT_WITH_MASK
            )
    except Exception:  # noqa: BLE001
        return None
    binm = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    n, _labs, stats, _ = cv2.connectedComponentsWithStats(binm)
    best = None
    best_a = 0
    for i in range(1, n):
        x, y, ww, hh, aa = stats[i]
        if aa < 0.0015 * h * w or aa > 0.45 * h * w:
            continue
        ox1 = max(x, rx)
        oy1 = max(y, ry)
        ox2 = min(x + ww, rx + rw)
        oy2 = min(y + hh, ry + rh)
        if max(0, ox2 - ox1) * max(0, oy2 - oy1) <= 0:
            continue
        ar = ww / float(hh + 1e-6)
        if ar > 2.0 or ar < 0.2:
            continue
        if aa > best_a:
            best_a = aa
            best = (x, y, x + ww, y + hh)
    return best


class WatershedGrabCutDetector:
    """Configurable automated watershed / GrabCut vial finder."""

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.40,
        iou_threshold: float = 0.35,
        max_side: int = 640,
        max_dets: int = 1,
        grabcut_iters: int = 2,
        use_center_prior: bool = True,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_side = int(max_side)
        self.max_dets = int(max_dets)
        self.grabcut_iters = int(grabcut_iters)
        self.use_center_prior = bool(use_center_prior)
        self.backend = "watershed_grabcut"

    def predict(self, bgr: np.ndarray) -> Tuple[List[Det], InferStats]:
        t0 = time.perf_counter()
        dets, n_seeds, n_markers = self._detect(bgr)
        ms = (time.perf_counter() - t0) * 1000.0
        return dets, InferStats(self.backend, ms, n_seeds=n_seeds, n_markers=n_markers)

    def annotate(self, bgr: np.ndarray, dets: Sequence[Det]) -> np.ndarray:
        out = bgr.copy()
        for d in dets:
            p1 = (int(d.x1), int(d.y1))
            p2 = (int(d.x2), int(d.y2))
            cv2.rectangle(out, p1, p2, (40, 200, 80), 2)
            tag = f"{d.label} {d.conf:.2f}"
            cv2.putText(
                out,
                tag,
                (p1[0], max(16, p1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 200, 80),
                2,
                cv2.LINE_AA,
            )
        return out

    def _detect(self, bgr: np.ndarray) -> Tuple[List[Det], int, int]:
        if bgr is None or bgr.size == 0:
            return [], 0, 0
        H0, W0 = bgr.shape[:2]
        scale = 1.0
        img = bgr
        if max(H0, W0) > self.max_side:
            scale = self.max_side / float(max(H0, W0))
            img = cv2.resize(bgr, (int(W0 * scale), int(H0 * scale)))
        h, w = img.shape[:2]

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 35, 100)

        cap = _adaptive_cap_mask(val, sat)
        # Blown-out / empty frames: discard near-full white mask.
        if float(cap.mean()) > 200:
            cap = np.zeros_like(cap)
        cap = cv2.morphologyEx(
            cap, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        cap = cv2.morphologyEx(
            cap, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3))
        )
        cap = cv2.morphologyEx(
            cap, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )

        n, labels, stats, cents = cv2.connectedComponentsWithStats(cap)
        filtered = np.zeros_like(cap)
        seeds: List[dict] = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < max(40, int(0.00045 * h * w)) or area > int(0.16 * h * w):
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            if bw > 0.72 * w or bh > 0.72 * h:
                continue
            ar = bw / float(bh + 1e-6)
            side_on = bw >= max(14, int(0.03 * w)) and 1.55 <= ar <= 10.0 and bh >= 3
            roundish = (
                bw >= max(10, int(0.02 * min(h, w)))
                and bh >= max(8, int(0.02 * min(h, w)))
                and 0.55 <= ar < 1.55
            )
            if not (side_on or roundish):
                continue
            fill = area / float(bw * bh + 1e-6)
            if fill < 0.25:
                continue
            filtered[labels == i] = 255
            seeds.append(
                {
                    "x": x,
                    "y": y,
                    "w": bw,
                    "h": bh,
                    "cx": float(cents[i, 0]),
                    "cy": float(cents[i, 1]),
                    "area": area,
                    "fill": fill,
                }
            )

        used_center = False
        if not seeds and self.use_center_prior:
            used_center = True
            rw, rh = int(0.5 * w), int(0.55 * h)
            x = (w - rw) // 2
            y = int(0.12 * h)
            seeds.append(
                {
                    "x": x,
                    "y": y,
                    "w": rw,
                    "h": max(12, rh // 4),
                    "cx": w / 2.0,
                    "cy": y + rh * 0.15,
                    "area": rw * max(12, rh // 4),
                    "fill": 0.5,
                }
            )
            filtered[y : y + max(12, rh // 4), x : x + rw] = 255

        # Distance-transform peaks → watershed markers.
        dist = cv2.distanceTransform((filtered > 0).astype(np.uint8), cv2.DIST_L2, 5)
        markers = np.zeros((h, w), np.int32)
        n_markers = 0
        if dist.max() > 1.0:
            ker = max(5, int(0.04 * min(h, w)) | 1)
            dil = cv2.dilate(dist, np.ones((ker, ker), np.uint8))
            peaks = (dist == dil) & (dist >= max(2.0, 0.35 * float(dist.max())))
            ys, xs = np.where(peaks)
            pts = sorted(zip(dist[ys, xs].tolist(), xs.tolist(), ys.tolist()), reverse=True)[
                :12
            ]
            for k, (_, px, py) in enumerate(pts, start=2):
                cv2.circle(markers, (int(px), int(py)), 3, k, -1)
                n_markers += 1
        if n_markers == 0:
            for k, sd in enumerate(seeds[:8], start=2):
                r = max(2, int(0.2 * min(sd["w"], sd["h"])))
                cv2.circle(markers, (int(sd["cx"]), int(sd["cy"])), r, k, -1)
                n_markers += 1

        sure_fg = (markers > 1).astype(np.uint8) * 255
        sure_bg = cv2.bitwise_not(cv2.dilate(filtered, np.ones((7, 7), np.uint8), 2))
        sure_bg[:2, :] = 255
        sure_bg[-2:, :] = 255
        sure_bg[:, :2] = 255
        sure_bg[:, -2:] = 255
        unknown = cv2.subtract(cv2.bitwise_not(sure_bg), sure_fg)
        markers_ws = markers.copy()
        markers_ws[sure_bg > 0] = 1
        markers_ws[unknown > 0] = 0
        try:
            cv2.watershed(img, markers_ws)
        except Exception:  # noqa: BLE001
            markers_ws = markers

        raw_dets: List[Det] = []
        for lab in range(2, int(markers_ws.max()) + 1):
            mask = (markers_ws == lab).astype(np.uint8)
            if int(mask.sum()) < 20:
                continue
            ys, xs = np.where(mask > 0)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            bw, bh = x2 - x1 + 1, y2 - y1 + 1
            if bw * bh > 0.35 * h * w:
                continue
            bx1, by1, bx2, by2 = _grow_cup(edges, sat, val, x1, y1, bw, bh)
            rw, rh = bx2 - bx1, by2 - by1
            if rw < 8 or rh < 10:
                continue
            ar = rw / float(rh)
            if ar > 1.95 or ar < 0.2:
                continue

            refined = _grabcut_refine(
                img, (bx1, by1, rw, rh), mask, self.grabcut_iters
            )
            if refined is not None:
                bx1, by1, bx2, by2 = refined

            tw = max(1, int(0.4 * (by2 - by1)))
            top = (val[by1 : by1 + tw, bx1:bx2] >= 175) & (
                sat[by1 : by1 + tw, bx1:bx2] <= 65
            )
            top_white = float(top.mean()) if top.size else 0.0
            aspect_h = (by2 - by1) / float(max(1, bx2 - bx1))
            aspect_score = 1.0 - min(1.0, abs(aspect_h - 1.25) / 1.0)
            border_pen = 0.0
            if by1 <= 1 and (bx2 - bx1) > 0.25 * w:
                border_pen = 0.28
            elif bx1 <= 1 or by1 <= 1 or bx2 >= w - 2:
                border_pen = 0.12
            cxn = 0.5 * (bx1 + bx2) / w
            cyn = 0.5 * (by1 + by2) / h
            center = 1.0 - min(1.0, (((cxn - 0.5) ** 2 + (cyn - 0.55) ** 2) ** 0.5) / 0.75)
            score = (
                0.25
                + 0.45 * top_white
                + 0.2 * aspect_score
                + 0.15 * center
                - border_pen
            )
            if used_center:
                score = max(score, 0.35 + 0.25 * aspect_score)
            score = float(np.clip(score, 0.05, 0.99))
            if score < self.confidence_threshold:
                continue
            raw_dets.append(
                Det(
                    bx1 / scale,
                    by1 / scale,
                    bx2 / scale,
                    by2 / scale,
                    score,
                    source="watershed_grabcut",
                )
            )

        return _nms(raw_dets, self.iou_threshold, self.max_dets), len(seeds), n_markers
