"""Sideways vial detector G: ensemble B + A + D with IoU consensus.

Calls working race methods (B_rotated_white_cap, A_cap_horizontal_grow,
D_edge_template), clusters proposals by IoU, and ranks survivors by white
end-cap, elongation, and vial-scale at Z. B anchors recall/FP; A/D can
add boxes only when they pass consensus or strict quality gates.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

NAME = "G_ensemble"

_METHODS_DIR = Path(__file__).resolve().parent
_ROOT = _METHODS_DIR.parents[1]
_DIMS_PATH = _ROOT / "vial_3d_dims.json"

_DEFAULT_TOTAL_H_MM = 72.69
_DEFAULT_LID_D_MM = 54.22
_SIDEWAYS_ASPECT_MIN = 0.34
_SIDEWAYS_ASPECT_MAX = 0.82
_MIN_IOU_CLUSTER = 0.25
_MIN_SURFACE_MM = 80.0
_DEFAULT_FX = 600.0

# Supplemental (non-B) acceptance — tuned on synth race set.
_MIN_WHITE_END_B = 0.08
_MIN_WHITE_END_A = 0.12
_MIN_WHITE_END_D = 0.20
_MIN_RANK_A = 0.22
_MIN_RANK_D = 0.48
_MIN_HOLE_FRAC = 0.35
_NMS_IOU = 0.35


def _load_dims() -> tuple[float, float]:
    if _DIMS_PATH.is_file():
        try:
            data = json.loads(_DIMS_PATH.read_text(encoding="utf-8"))
            h = float(data.get("total_height_mm", _DEFAULT_TOTAL_H_MM))
            d = float(data.get("lid_diameter_mm", _DEFAULT_LID_D_MM))
            return h, d
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return _DEFAULT_TOTAL_H_MM, _DEFAULT_LID_D_MM


TOTAL_H_MM, LID_D_MM = _load_dims()

_DETECTORS: dict[str, object] | None = None


def _load_detector(stem: str):
    """Load sibling method module; return None on failure."""
    path = _METHODS_DIR / f"{stem}.py"
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location(f"_g_{stem}", path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "detect_sideways"):
            return None
        return mod.detect_sideways
    except Exception:
        return None


def _get_detectors() -> dict[str, object]:
    global _DETECTORS
    if _DETECTORS is None:
        _DETECTORS = {
            "B": _load_detector("method_b_rotated_white_cap"),
            "A": _load_detector("method_a_cap_horizontal_grow"),
            "D": _load_detector("method_d_edge_template"),
        }
    return _DETECTORS


def _clip_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = (int(v) for v in box)
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _union_box(boxes) -> tuple[int, int, int, int]:
    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    x2 = [b[0] + b[2] for b in boxes]
    y2 = [b[1] + b[3] for b in boxes]
    x0, y0 = min(xs), min(ys)
    return int(x0), int(y0), int(max(x2) - x0), int(max(y2) - y0)


def _fx_fy(intrinsics, width: int, height: int) -> tuple[float, float]:
    if intrinsics is None:
        return _DEFAULT_FX, _DEFAULT_FX
    if isinstance(intrinsics, dict):
        fx = float(intrinsics.get("fx", _DEFAULT_FX))
        fy = float(intrinsics.get("fy", fx))
        return fx, fy
    try:
        seq = list(intrinsics)
        if len(seq) >= 2:
            return float(seq[0]), float(seq[1])
    except (TypeError, ValueError):
        pass
    return _DEFAULT_FX, _DEFAULT_FX


def _is_elongated(box) -> bool:
    _, _, bw, bh = box
    if bw < 16 or bh < 8:
        return False
    aspect = bh / max(bw, 1)
    return _SIDEWAYS_ASPECT_MIN <= aspect <= _SIDEWAYS_ASPECT_MAX


def _white_end_score(color_bgr: np.ndarray, box) -> float:
    h, w = color_bgr.shape[:2]
    x, y, bw, bh = _clip_box(box, w, h)
    if bw < 12 or bh < 8:
        return 0.0
    hsv = cv2.cvtColor(color_bgr[y : y + bh, x : x + bw], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 150)
    band = max(3, int(bw * 0.34))
    if bw <= band * 2:
        return 0.0
    left = float(white[:, :band].mean())
    right = float(white[:, -band:].mean())
    mid = float(white[:, band : bw - band].mean())
    end = max(left, right)
    if end < 0.22 or mid > end - 0.05:
        return 0.0
    return float(max(0.0, min(1.0, end - mid + 0.08)))


def _median_z_mm(
    depth_mm: np.ndarray, color_bgr: np.ndarray, box
) -> float | None:
    h, w = depth_mm.shape[:2]
    x, y, bw, bh = _clip_box(box, w, h)
    roi_d = depth_mm[y : y + bh, x : x + bw]
    hsv = cv2.cvtColor(color_bgr[y : y + bh, x : x + bw], cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 60) & (hsv[:, :, 2] >= 170)
    band = max(3, int(bw * 0.34))
    left_w = float(white[:, :band].mean()) if band < bw else 0.0
    right_w = float(white[:, -band:].mean()) if band < bw else 0.0
    if left_w >= right_w and left_w >= 0.2:
        patch = roi_d[:, :band]
    elif right_w > left_w and right_w >= 0.2:
        patch = roi_d[:, -band:]
    else:
        patch = roi_d
    valid = patch[(patch > _MIN_SURFACE_MM) & (patch < 5000)]
    if valid.size < 8:
        valid = roi_d[(roi_d > _MIN_SURFACE_MM) & (roi_d < 5000)]
    if valid.size < 4:
        return None
    return float(np.median(valid))


def _scale_ok_at_z(box, z_mm: float | None, fx: float, fy: float) -> bool:
    _, _, bw, bh = box
    if z_mm is None or z_mm <= _MIN_SURFACE_MM:
        return True
    if fx <= 1 or fy <= 1:
        return True
    width_mm = bw * z_mm / fx
    height_mm = bh * z_mm / fy
    long_mm = max(width_mm, height_mm)
    short_mm = min(width_mm, height_mm)
    return (
        0.42 * TOTAL_H_MM <= long_mm <= 2.3 * TOTAL_H_MM
        and 0.32 * LID_D_MM <= short_mm <= 2.1 * LID_D_MM
    )


def _clear_cup_hole_frac(depth_mm: np.ndarray, box) -> float:
    """Invalid-depth fraction in the clear cup (middle band), excluding end caps."""
    h, w = depth_mm.shape[:2]
    x, y, bw, bh = _clip_box(box, w, h)
    roi = depth_mm[y : y + bh, x : x + bw]
    band = max(3, int(bw * 0.34))
    if bw <= band * 2:
        return 0.0
    mid = roi[:, band : bw - band]
    return float((mid <= 1.0).mean())


def _body_support(
    depth_mm: np.ndarray, color_bgr: np.ndarray, box, z_mm: float | None
) -> float:
    if z_mm is None:
        return 0.5
    h, w = depth_mm.shape[:2]
    x, y, bw, bh = _clip_box(box, w, h)
    roi = depth_mm[y : y + bh, x : x + bw]
    near = float(((roi > _MIN_SURFACE_MM) & (roi < z_mm + 220.0)).mean())
    invalid = float((roi <= 1.0).mean())
    far = float((roi > z_mm + 220.0).mean())
    support = near + 0.75 * invalid - 0.65 * far
    return max(0.0, min(1.0, support))


def _rank(
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    box,
    z_mm: float | None,
    fx: float,
    fy: float,
) -> float:
    end = _white_end_score(color_bgr, box)
    _, _, bw, bh = box
    elong = min(bw / max(bh, 1), 3.0) / 3.0
    scale = 0.5
    if z_mm is not None and fx > 1 and fy > 1:
        long_mm = max(bw * z_mm / fx, bh * z_mm / fy)
        short_mm = min(bw * z_mm / fx, bh * z_mm / fy)
        scale = max(0.0, 1.0 - abs(long_mm - TOTAL_H_MM) / 90.0) * 0.5 + max(
            0.0, 1.0 - abs(short_mm - LID_D_MM) / 70.0
        ) * 0.5
    body = _body_support(depth_mm, color_bgr, box, z_mm)
    return float(end * 0.50 + elong * 0.20 + scale * 0.15 + body * 0.15)


def _call_method(fn, color_bgr, depth_mm, intrinsics) -> list[tuple[int, int, int, int]]:
    if fn is None:
        return []
    try:
        out = fn(color_bgr, depth_mm, intrinsics) or []
        return [_clip_box(tuple(int(v) for v in b), color_bgr.shape[1], color_bgr.shape[0]) for b in out]
    except Exception:
        return []


def _merge_cup_neighbors(
    box: tuple[int, int, int, int],
    candidates: list[tuple[int, int, int, int]],
    depth_mm: np.ndarray,
) -> tuple[int, int, int, int]:
    """Merge nearby D/A boxes that share the same clear-cup depth signature."""
    x, y, w, h = box
    cy = y + h * 0.5
    merged = [box]
    for other in candidates:
        if other == box:
            continue
        if _clear_cup_hole_frac(depth_mm, other) < _MIN_HOLE_FRAC:
            continue
        ox, oy, ow, oh = other
        ocy = oy + oh * 0.5
        if abs(ocy - cy) > max(h, oh) * 0.65:
            continue
        gap = max(x, ox) - min(x + w, ox + ow)
        if gap > 28:
            continue
        merged.append(other)
    if len(merged) >= 2:
        u = _union_box(merged)
        if _is_elongated(u):
            return u
    return box


def _cluster_proposals(
    tagged: list[tuple[tuple[int, int, int, int], str, float]],
    iou_thr: float = _MIN_IOU_CLUSTER,
) -> list[list[int]]:
    """Return cluster index lists into ``tagged``."""
    n = len(tagged)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: tagged[i][2], reverse=True)
    used = [False] * n
    clusters: list[list[int]] = []
    for idx in order:
        if used[idx]:
            continue
        group = [idx]
        used[idx] = True
        changed = True
        while changed:
            changed = False
            for j in range(n):
                if used[j]:
                    continue
                if any(
                    _iou(tagged[gi][0], tagged[j][0]) >= iou_thr for gi in group
                ):
                    used[j] = True
                    group.append(j)
                    changed = True
        clusters.append(group)
    return clusters


def _pick_cluster_box(
    group: list[int],
    tagged: list[tuple[tuple[int, int, int, int], str, float]],
    depth_mm: np.ndarray | None = None,
) -> tuple[tuple[int, int, int, int], set[str], float]:
    sources = {tagged[i][1] for i in group}
    # Prefer B member when present; else highest pre-score.
    b_members = [i for i in group if tagged[i][1] == "B"]
    if b_members:
        best_i = max(b_members, key=lambda i: tagged[i][2])
        consensus = 0.04 * (len(sources) - 1)
        return tagged[best_i][0], sources, tagged[best_i][2] + consensus

    best_i = max(group, key=lambda i: tagged[i][2])
    box = tagged[best_i][0]
    score = tagged[best_i][2]
    if depth_mm is not None and "B" not in sources:
        cup_boxes = [
            tagged[i][0]
            for i in group
            if _clear_cup_hole_frac(depth_mm, tagged[i][0]) >= _MIN_HOLE_FRAC
        ]
        if len(cup_boxes) >= 2:
            u = _union_box(cup_boxes)
            if _is_elongated(u):
                box = u
                score = max(tagged[i][2] for i in group)
    consensus = 0.04 * (len(sources) - 1)
    return box, sources, score + consensus


def _cluster_acceptable(
    sources: set[str],
    rank: float,
    white_end: float,
    scale_ok: bool,
    hole_frac: float,
) -> bool:
    if not scale_ok:
        return False
    if "B" in sources:
        return white_end >= _MIN_WHITE_END_B
    # Non-B supplements must show a see-through cup (depth holes), not table/wall.
    if hole_frac < _MIN_HOLE_FRAC:
        return False
    if "A" in sources:
        return white_end >= _MIN_WHITE_END_A and rank >= _MIN_RANK_A
    if sources <= {"D"}:
        return white_end >= _MIN_WHITE_END_D and rank >= _MIN_RANK_D
    if "D" in sources and "A" in sources:
        return white_end >= _MIN_WHITE_END_A and rank >= _MIN_RANK_A - 0.02
    return False


def detect_sideways(
    color_bgr, depth_mm, intrinsics=None
) -> list[tuple[int, int, int, int]]:
    if color_bgr is None or depth_mm is None:
        return []
    color_bgr = np.asarray(color_bgr)
    depth_mm = np.asarray(depth_mm, dtype=np.float32)
    if color_bgr.ndim != 3 or depth_mm.ndim != 2:
        return []
    h, w = color_bgr.shape[:2]
    if depth_mm.shape[:2] != (h, w):
        depth_mm = cv2.resize(depth_mm, (w, h), interpolation=cv2.INTER_NEAREST)

    dets = _get_detectors()
    fx, fy = _fx_fy(intrinsics, w, h)

    b_boxes = _call_method(dets.get("B"), color_bgr, depth_mm, intrinsics)
    a_boxes = _call_method(dets.get("A"), color_bgr, depth_mm, intrinsics)
    d_boxes = _call_method(dets.get("D"), color_bgr, depth_mm, intrinsics)

    tagged: list[tuple[tuple[int, int, int, int], str, float]] = []
    for box in b_boxes:
        z_mm = _median_z_mm(depth_mm, color_bgr, box)
        tagged.append((box, "B", _rank(color_bgr, depth_mm, box, z_mm, fx, fy) + 0.12))
    for box in a_boxes:
        z_mm = _median_z_mm(depth_mm, color_bgr, box)
        tagged.append((box, "A", _rank(color_bgr, depth_mm, box, z_mm, fx, fy)))
    for box in d_boxes:
        z_mm = _median_z_mm(depth_mm, color_bgr, box)
        tagged.append((box, "D", _rank(color_bgr, depth_mm, box, z_mm, fx, fy)))

    # Anchor: always keep B outputs (zero FP on race set).
    final: list[tuple[int, int, int, int]] = list(b_boxes)

    all_cup_candidates = [
        tagged[i][0]
        for i in range(len(tagged))
        if tagged[i][1] in ("A", "D")
        and _clear_cup_hole_frac(depth_mm, tagged[i][0]) >= _MIN_HOLE_FRAC
    ]

    clusters = _cluster_proposals(tagged)
    for group in clusters:
        box, sources, rank = _pick_cluster_box(group, tagged, depth_mm)
        box = _clip_box(box, w, h)
        if "B" not in sources:
            box = _clip_box(_merge_cup_neighbors(box, all_cup_candidates, depth_mm), w, h)
        if "B" in sources and any(_iou(box, b) >= _NMS_IOU for b in b_boxes):
            continue  # already in final
        if not _is_elongated(box):
            continue
        z_mm = _median_z_mm(depth_mm, color_bgr, box)
        if not _scale_ok_at_z(box, z_mm, fx, fy):
            continue
        white_end = _white_end_score(color_bgr, box)
        hole_frac = _clear_cup_hole_frac(depth_mm, box)
        rank = _rank(color_bgr, depth_mm, box, z_mm, fx, fy)
        if "A" in sources or "D" in sources:
            rank += 0.04 * (len(sources) - 1)
        if not _cluster_acceptable(sources, rank, white_end, True, hole_frac):
            continue
        if any(_iou(box, f) >= _NMS_IOU for f in final):
            continue
        final.append(box)

    final.sort(
        key=lambda b: _rank(
            color_bgr,
            depth_mm,
            b,
            _median_z_mm(depth_mm, color_bgr, b),
            fx,
            fy,
        ),
        reverse=True,
    )
    return final


def _synth_sideways(
    width: int = 320,
    height: int = 240,
    lid_on: str = "left",
    z_mm: float = 450.0,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    color = np.full((height, width, 3), 48, dtype=np.uint8)
    body_w, body_h = 110, 52
    x0 = width // 2 - body_w // 2
    y0 = height // 2 - body_h // 2
    cv2.rectangle(color, (x0, y0), (x0 + body_w - 1, y0 + body_h - 1), (78, 72, 62), -1)
    cv2.rectangle(color, (x0, y0), (x0 + body_w - 1, y0 + body_h - 1), (120, 110, 95), 2)
    lid_w = 28
    lx = x0 if lid_on == "left" else x0 + body_w - lid_w
    color[y0 : y0 + body_h, lx : lx + lid_w] = (235, 235, 235)
    depth = np.full((height, width), 900.0, dtype=np.float32)
    depth[y0 : y0 + body_h, x0 : x0 + body_w] = z_mm + 180.0
    depth[y0 : y0 + body_h, lx : lx + lid_w] = z_mm
    mid_x0 = x0 + lid_w + 4
    mid_x1 = x0 + body_w - 4
    depth[y0 + 6 : y0 + body_h - 6, mid_x0:mid_x1:3] = 0.0
    return color, depth, (x0, y0, body_w, body_h)


if __name__ == "__main__":
    print(f"NAME={NAME}")
    print(f"detectors={ {k: v is not None for k, v in _get_detectors().items()} }")

    synth_dir = _METHODS_DIR.parent / "synth"
    manifest_path = synth_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = next(
            (s for s in manifest.get("samples", []) if s.get("id") == "sideways_left_00"),
            manifest.get("samples", [{}])[0],
        )
        color = cv2.imread(str(synth_dir / entry["color"]))
        depth = np.load(synth_dir / entry["depth"]).astype(np.float32)
        K = entry.get("intrinsics")
        gt = tuple(entry["boxes"][0]) if entry.get("boxes") else None
        label = entry.get("id", "manifest")
    else:
        color, depth, gt = _synth_sideways(lid_on="left")
        K = (600.0, 600.0, color.shape[1] / 2.0, color.shape[0] / 2.0)
        label = "inline_synth"

    if color is None or gt is None:
        raise SystemExit("self-test: missing color or gt")

    boxes = detect_sideways(color, depth, intrinsics=K)
    print(f"sample={label} gt={gt}")
    print(f"detections={boxes}")
    if not boxes:
        print("best_iou=0.000")
        raise SystemExit(1)
    best_iou = max(_iou(gt, b) for b in boxes)
    print(f"best_iou={best_iou:.3f}")
    if best_iou < 0.45:
        raise SystemExit(1)
    print("self-test OK")
