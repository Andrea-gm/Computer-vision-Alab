"""Stage-1 proposal generators: white-cap HSV, edge contours, optional YOLO."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .types import Detection


def _white_mask(bgr: np.ndarray, v_min: int = 170, s_max: int = 60) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] >= v_min) & (hsv[:, :, 1] <= s_max)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return mask


def _expand_from_cap(
    gray: np.ndarray,
    edges: np.ndarray,
    cx: int,
    cy: int,
    cw: int,
    ch: int,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Grow a lid blob into a full vial box using vertical/horizontal edge support."""
    # Prefer upright growth (cap on top) — dominant pose in this dataset.
    max_down = int(max(ch * 4.5, cw * 1.6))
    max_up = int(ch * 1.2)
    max_side = int(max(cw * 3.5, ch * 1.8))

    def edge_row_score(row: int, x0: int, x1: int) -> float:
        if row < 0 or row >= img_h or x1 <= x0:
            return 0.0
        strip = edges[row, x0:x1]
        return float(strip.mean()) / 255.0

    def edge_col_score(col: int, y0: int, y1: int) -> float:
        if col < 0 or col >= img_w or y1 <= y0:
            return 0.0
        strip = edges[y0:y1, col]
        return float(strip.mean()) / 255.0

    x0, x1 = cx, cx + cw
    y0, y1 = cy, cy + ch

    # Grow downward (cup below cap).
    gap = 0
    for dy in range(1, max_down + 1):
        row = y1 + dy - 1
        if row >= img_h:
            break
        score = edge_row_score(row, max(0, x0 - 2), min(img_w, x1 + 2))
        bright = float(gray[row, max(0, x0) : min(img_w, x1)].mean()) if x1 > x0 else 0.0
        if score >= 0.04 or bright >= 40:
            gap = 0
            y1 = row + 1
        else:
            gap += 1
            if gap >= 6:
                break

    # Grow upward a little (top-down / partial cap).
    gap = 0
    for dy in range(1, max_up + 1):
        row = y0 - dy
        if row < 0:
            break
        score = edge_row_score(row, max(0, x0 - 2), min(img_w, x1 + 2))
        if score >= 0.04:
            gap = 0
            y0 = row
        else:
            gap += 1
            if gap >= 4:
                break

    # Sideways: grow left/right from end-cap.
    aspect = (x1 - x0) / max(1, y1 - y0)
    if aspect > 0.95 or cw >= ch * 0.85:
        for direction, sign in (("left", -1), ("right", 1)):
            gap = 0
            limit = max_side
            for step in range(1, limit + 1):
                col = (x0 - step) if sign < 0 else (x1 + step - 1)
                if col < 0 or col >= img_w:
                    break
                score = edge_col_score(col, max(0, y0 - 2), min(img_h, y1 + 2))
                if score >= 0.035:
                    gap = 0
                    if sign < 0:
                        x0 = col
                    else:
                        x1 = col + 1
                else:
                    gap += 1
                    if gap >= 6:
                        break

    # Pad slightly so clear rim is inside the box (GT boxes are full lid→cup).
    pad_x = max(2, int(0.04 * (x1 - x0)))
    pad_y = max(2, int(0.03 * (y1 - y0)))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(img_w, x1 + pad_x)
    y1 = min(img_h, y1 + pad_y)
    return x0, y0, x1, y1


def propose_cap(bgr: np.ndarray, max_proposals: int = 12) -> list[Detection]:
    """White-cap HSV blobs expanded into lid+cup boxes."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 110)
    mask = _white_mask(bgr)
    # Suppress huge near-full-frame white (table/wall).
    if float(mask.mean()) / 255.0 > 0.45:
        mask = _white_mask(bgr, v_min=200, s_max=45)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dets: list[Detection] = []
    area_img = float(w * h)
    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        if area < 40 or area > 0.35 * area_img:
            continue
        if bw < 6 or bh < 4:
            continue
        # Cap should be compact; reject huge wallpaper blobs.
        fill = area / float(max(1, bw * bh))
        if fill < 0.25:
            continue
        x0, y0, x1, y1 = _expand_from_cap(gray, edges, x, y, bw, bh, w, h)
        box_area = (x1 - x0) * (y1 - y0)
        if box_area < 80 or box_area > 0.92 * area_img:
            continue
        # Score: white density in top/end band + expansion growth.
        growth = box_area / float(max(1, bw * bh))
        white_frac = float(mask[y : y + bh, x : x + bw].mean()) / 255.0
        score = float(np.clip(0.35 * white_frac + 0.15 * min(growth, 6.0) / 6.0, 0.05, 0.95))
        dets.append(
            Detection(
                x1=float(x0),
                y1=float(y0),
                x2=float(x1),
                y2=float(y1),
                score=score,
                source="cap",
                extras={"cap_area": float(area), "growth": float(growth)},
            )
        )
    dets.sort(key=lambda d: d.score, reverse=True)
    return dets[:max_proposals]


def propose_edges(bgr: np.ndarray, max_proposals: int = 10) -> list[Detection]:
    """Edge / contour proposals with vial-like aspect and rim support."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 35, 100)
    # Dilate to close clear-body gaps.
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] <= 70) & (hsv[:, :, 2] >= 150)).astype(np.uint8)
    dets: list[Detection] = []
    area_img = float(w * h)
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 120 or area > 0.85 * area_img:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 10 or bh < 12:
            continue
        aspect = bw / float(bh)
        # Dataset GT aspect roughly 0.47–1.20; allow sideways slack.
        if aspect < 0.35 or aspect > 1.55:
            continue
        peri = cv2.arcLength(cnt, True)
        circularity = 4.0 * np.pi * area / (peri * peri + 1e-6)
        roi_white = float(white[y : y + bh, x : x + bw].mean())
        if roi_white < 0.02:
            continue
        # Prefer elongated upright vials with some white (cap).
        score = float(
            np.clip(
                0.25 * roi_white
                + 0.2 * (1.0 - abs(aspect - 0.65))
                + 0.15 * min(circularity, 1.0)
                + 0.1 * min(area / area_img, 0.4) / 0.4,
                0.05,
                0.9,
            )
        )
        dets.append(
            Detection(
                x1=float(x),
                y1=float(y),
                x2=float(x + bw),
                y2=float(y + bh),
                score=score,
                source="edge",
                extras={"circularity": float(circularity), "white": roi_white},
            )
        )
    dets.sort(key=lambda d: d.score, reverse=True)
    return dets[:max_proposals]


class YoloProposer:
    """Optional ultralytics YOLO proposer (high-recall low-conf)."""

    def __init__(self, weights: str | None, conf: float = 0.10, device: str | None = None):
        self.weights = weights
        self.conf = conf
        self.device = device
        self._model: Any = None
        self.available = False
        if weights:
            try:
                from ultralytics import YOLO

                self._model = YOLO(weights)
                self.available = True
            except Exception:
                self._model = None
                self.available = False

    def propose(self, bgr: np.ndarray, max_proposals: int = 10) -> list[Detection]:
        if not self.available or self._model is None:
            return []
        kwargs: dict[str, Any] = dict(conf=self.conf, verbose=False)
        if self.device is not None:
            kwargs["device"] = self.device
        results = self._model.predict(bgr, **kwargs)
        dets: list[Detection] = []
        if not results:
            return dets
        boxes = results[0].boxes
        if boxes is None:
            return dets
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            dets.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=float(c),
                    source="yolo",
                    extras={"yolo_conf": float(c)},
                )
            )
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets[:max_proposals]


def shape_filter(
    detections: list[Detection],
    width: int,
    height: int,
    *,
    min_aspect: float = 0.38,
    max_aspect: float = 1.45,
    min_area_frac: float = 0.0015,
    max_area_frac: float = 0.95,
    min_side: int = 10,
) -> list[Detection]:
    """Stage-2 geometric filter tuned to this dataset's GT aspect distribution."""
    kept: list[Detection] = []
    area_img = float(max(1, width * height))
    for d in detections:
        d = d.clip(width, height)
        x1, y1, x2, y2 = d.xyxy
        bw = x2 - x1
        bh = y2 - y1
        if bw < min_side or bh < min_side:
            continue
        aspect = bw / float(bh)
        if aspect < min_aspect or aspect > max_aspect:
            continue
        area_frac = (bw * bh) / area_img
        if area_frac < min_area_frac or area_frac > max_area_frac:
            continue
        kept.append(d)
    return kept
