"""Generate the original and hard synthetic RGB-D sideways-vial race sets."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SYNTH = ROOT / "synth"
CAMERA = ROOT.parent
DATASET = CAMERA / "vial_dataset"

# Default pinhole-ish intrinsics matching 640x480 Orbbec-ish FOV.
DEFAULT_K = {
    "fx": 575.0,
    "fy": 575.0,
    "cx": 320.0,
    "cy": 240.0,
    "width": 640,
    "height": 480,
}


def _desk(w: int, h: int, rng: np.random.Generator, base: int = 88) -> tuple[np.ndarray, np.ndarray]:
    color = np.full((h, w, 3), base, dtype=np.uint8)
    noise = rng.integers(-12, 13, size=color.shape, dtype=np.int16)
    color = np.clip(color.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Mottled wall / clutter patches
    for _ in range(int(rng.integers(2, 5))):
        x0 = int(rng.integers(0, max(1, w - 80)))
        y0 = int(rng.integers(0, max(1, h - 60)))
        ww = int(rng.integers(40, 100))
        hh = int(rng.integers(30, 80))
        shade = int(rng.integers(70, 170))
        color[y0 : y0 + hh, x0 : x0 + ww] = (shade, shade, shade)
    depth = np.full((h, w), 1100.0, dtype=np.float32)
    depth += rng.normal(0, 8, size=depth.shape).astype(np.float32)
    return color, depth


def _paint_sideways(
    color: np.ndarray,
    depth: np.ndarray,
    *,
    lid_left: bool,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    h, w = color.shape[:2]
    vh = int(rng.integers(34, 48))
    vw = int(rng.integers(int(vh / 0.68), int(vh / 0.45)))  # h/w in ~0.45–0.68
    lid_w = max(10, int(vw * float(rng.uniform(0.18, 0.28))))
    vx = int(rng.integers(40, max(41, w - vw - 40)))
    vy = int(rng.integers(60, max(61, h - vh - 60)))
    body = (int(rng.integers(55, 85)), int(rng.integers(55, 80)), int(rng.integers(50, 75)))
    # Clear-ish body (darker) + white end-cap
    color[vy : vy + vh, vx : vx + vw] = body
    if lid_left:
        color[vy : vy + vh, vx : vx + lid_w] = (245, 245, 245)
        # Horizontal cup edges
        color[vy, vx + lid_w : vx + vw] = (35, 35, 35)
        color[vy + vh - 1, vx + lid_w : vx + vw] = (35, 35, 35)
        depth[vy : vy + vh, vx : vx + lid_w] = float(rng.uniform(480, 560))
        body_d = depth[vy : vy + vh, vx + lid_w : vx + vw]
        # Holes / far through clear cup
        body_d[:] = 0.0
        if rng.random() < 0.45:
            body_d[:, ::3] = float(rng.uniform(900, 1400))
    else:
        color[vy : vy + vh, vx + vw - lid_w : vx + vw] = (245, 245, 245)
        color[vy, vx : vx + vw - lid_w] = (35, 35, 35)
        color[vy + vh - 1, vx : vx + vw - lid_w] = (35, 35, 35)
        depth[vy : vy + vh, vx + vw - lid_w : vx + vw] = float(rng.uniform(480, 560))
        body_d = depth[vy : vy + vh, vx : vx + vw - lid_w]
        body_d[:] = 0.0
        if rng.random() < 0.45:
            body_d[:, ::3] = float(rng.uniform(900, 1400))
    # Bright false-positive slab elsewhere
    if rng.random() < 0.7:
        sx = int(rng.integers(10, max(11, w - 90)))
        sy = int(rng.integers(20, max(21, h - 70)))
        color[sy : sy + 50, sx : sx + 70] = (230, 230, 230)
        depth[sy : sy + 50, sx : sx + 70] = float(rng.uniform(500, 700))
    return (vx, vy, vw, vh)


def _paint_topdown(color: np.ndarray, depth: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int, int]:
    h, w = color.shape[:2]
    side = int(rng.integers(40, 56))
    lx = int(rng.integers(80, max(81, w - side - 80)))
    ly = int(rng.integers(70, max(71, h - side - 70)))
    color[ly : ly + side, lx : lx + side] = (242, 242, 242)
    color[ly, lx : lx + side] = (200, 200, 200)
    color[ly + side - 1, lx : lx + side] = (200, 200, 200)
    color[ly : ly + side, lx] = (200, 200, 200)
    color[ly : ly + side, lx + side - 1] = (200, 200, 200)
    depth[ly : ly + side, lx : lx + side] = float(rng.uniform(420, 580))
    return (lx, ly, side, side)


def _paint_upright(color: np.ndarray, depth: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int, int]:
    h, w = color.shape[:2]
    bw = int(rng.integers(28, 42))
    bh = int(rng.integers(int(bw * 1.15), int(bw * 1.55)))
    lid_h = max(8, int(bh * 0.22))
    bx = int(rng.integers(80, max(81, w - bw - 80)))
    by = int(rng.integers(60, max(61, h - bh - 60)))
    color[by : by + lid_h, bx : bx + bw] = (245, 245, 245)
    color[by + lid_h : by + bh, bx + 2 : bx + bw - 2] = (70, 70, 65)
    color[by + lid_h, bx + 2 : bx + bw - 2] = (40, 40, 40)
    color[by + bh - 1, bx + 2 : bx + bw - 2] = (40, 40, 40)
    depth[by : by + lid_h, bx : bx + bw] = float(rng.uniform(480, 560))
    cup = depth[by + lid_h : by + bh, bx + 2 : bx + bw - 2]
    cup[:] = 0.0
    return (bx, by, bw, bh)


def _paint_empty_clutter(color: np.ndarray, depth: np.ndarray, rng: np.random.Generator) -> None:
    h, w = color.shape[:2]
    # Bright rectangle (window / paper) — not a vial
    x0 = int(rng.integers(20, max(21, w // 2)))
    y0 = int(rng.integers(20, max(21, h // 2)))
    color[y0 : y0 + 90, x0 : x0 + 120] = (220, 220, 220)
    depth[y0 : y0 + 90, x0 : x0 + 120] = float(rng.uniform(700, 900))
    # Dark tool-ish blob
    x1 = int(rng.integers(w // 2, max(w // 2 + 1, w - 80)))
    y1 = int(rng.integers(h // 2, max(h // 2 + 1, h - 60)))
    color[y1 : y1 + 40, x1 : x1 + 55] = (40, 45, 50)
    depth[y1 : y1 + 40, x1 : x1 + 55] = float(rng.uniform(350, 500))


def _rotated_box(
    center: tuple[float, float], size: tuple[float, float], angle: float
) -> np.ndarray:
    return np.rint(cv2.boxPoints((center, size, angle))).astype(np.int32)


def _paint_hard_sideways(
    color: np.ndarray,
    depth: np.ndarray,
    *,
    lid_left: bool,
    angle: float,
    variant: int,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    """Paint an angled, degraded vial and return its full oriented AABB."""
    h, w = color.shape[:2]
    diameter = int(rng.integers(34, 46))
    length = int(rng.integers(max(64, int(1.55 * diameter)), int(2.15 * diameter)))
    cap_len = int(round(length * rng.uniform(0.20, 0.28)))
    cx = int(rng.integers(length, w - length))
    cy = int(rng.integers(length, h - length))
    theta = np.deg2rad(angle)
    axis = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    body_len = length - cap_len
    cap_sign = -1.0 if lid_left else 1.0
    cap_c = np.array([cx, cy], dtype=np.float32) + cap_sign * axis * (body_len / 2)
    body_c = np.array([cx, cy], dtype=np.float32) - cap_sign * axis * (cap_len / 2)

    whole_poly = _rotated_box((cx, cy), (length, diameter), angle)
    body_poly = _rotated_box(tuple(body_c), (body_len, diameter - 2), angle)
    cap_poly = _rotated_box(tuple(cap_c), (cap_len, diameter), angle)
    body_mask = np.zeros((h, w), np.uint8)
    cap_mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(body_mask, body_poly, 255)
    cv2.fillConvexPoly(cap_mask, cap_poly, 255)

    dim = variant in {1, 5, 7, 11, 13, 15}
    body_value = int(rng.integers(35, 68) if dim else rng.integers(58, 95))
    cap_value = int(rng.integers(150, 190) if dim else rng.integers(215, 248))
    color[body_mask > 0] = (body_value, body_value, max(0, body_value - 5))
    color[cap_mask > 0] = (cap_value, cap_value, cap_value)
    cv2.polylines(color, [body_poly], True, (28, 28, 28), 2, cv2.LINE_AA)
    cv2.polylines(color, [cap_poly], True, (max(80, cap_value - 28),) * 3, 1, cv2.LINE_AA)

    # Cycle through normal, weak, and completely missing depth.
    mode = variant % 4
    if mode != 3:
        depth[body_mask > 0] = 0.0
        depth[cap_mask > 0] = float(rng.uniform(480, 610))
        if mode == 1:  # sparse/weak cap and cup measurements
            yy, xx = np.where((body_mask > 0) | (cap_mask > 0))
            keep = rng.random(yy.size) < 0.24
            depth[yy[~keep], xx[~keep]] = 0.0
        elif mode == 2:  # clear-plastic stripes resembling sensor leakage
            stripe = ((np.indices((h, w))[1] + variant) % 5) == 0
            depth[(body_mask > 0) & stripe] = float(rng.uniform(900, 1350))
    else:
        depth[(body_mask > 0) | (cap_mask > 0)] = 0.0

    # Partial occlusion across one body edge or the body/cap seam.
    if variant % 3 != 0:
        occ_c = np.array([cx, cy], dtype=np.float32) + axis * rng.uniform(-0.18, 0.20) * length
        occ_w = int(rng.integers(5, 12))
        occ_poly = _rotated_box(tuple(occ_c), (occ_w, diameter + 12), angle)
        shade = int(rng.integers(75, 125))
        cv2.fillConvexPoly(color, occ_poly, (shade, shade, shade))
        cv2.fillConvexPoly(depth, occ_poly, float(rng.uniform(780, 1150)))

    # Distractor white blobs; several deliberately resemble detached lids.
    for _ in range(int(rng.integers(2, 5))):
        # Keep distractors detached. Overpainting the labeled vial would turn
        # a hard case into ambiguous/corrupt ground truth rather than clutter.
        for _attempt in range(20):
            bx = int(rng.integers(20, w - 20))
            by = int(rng.integers(20, h - 20))
            if np.hypot(bx - cx, by - cy) > 1.25 * length:
                break
        bw = int(rng.integers(8, 30))
        bh = int(rng.integers(8, 38))
        value = int(rng.integers(175, 246))
        cv2.ellipse(color, (bx, by), (bw, bh), float(rng.integers(0, 180)), 0, 360, (value,) * 3, -1)
        cv2.ellipse(depth, (bx, by), (bw, bh), 0, 0, 360, float(rng.uniform(450, 900)), -1)

    x, y, bw, bh = cv2.boundingRect(whole_poly)
    return x, y, bw, bh


def _paint_hard_negative(
    color: np.ndarray, depth: np.ndarray, variant: int, rng: np.random.Generator
) -> None:
    """Paint cap-like clutter without an attached clear horizontal vial."""
    h, w = color.shape[:2]
    for j in range(3 + variant % 3):
        cx = int(rng.integers(45, w - 45))
        cy = int(rng.integers(45, h - 45))
        angle = float(rng.choice([-30, -20, -15, 15, 20, 30, 75, 90]))
        if j == 0 and variant % 2 == 0:
            size = (int(rng.integers(45, 90)), int(rng.integers(12, 24)))
        else:
            size = (int(rng.integers(14, 45)), int(rng.integers(14, 45)))
        poly = _rotated_box((cx, cy), size, angle)
        value = int(rng.integers(150, 245))
        cv2.fillConvexPoly(color, poly, (value, value, value))
        cv2.fillConvexPoly(depth, poly, float(rng.uniform(430, 950)))
    # Dark object is spatially separated from the white clutter and has solid depth.
    x = int(rng.integers(30, w - 150))
    y = int(rng.integers(30, h - 70))
    color[y : y + 35, x : x + 110] = (45, 48, 50)
    depth[y : y + 35, x : x + 110] = float(rng.uniform(500, 850))


def _write_sample(
    out_dir: Path,
    sample_id: str,
    color: np.ndarray,
    depth: np.ndarray,
    *,
    role: str,
    kind: str,
    boxes: list[list[int]],
    source: str = "synth",
    note: str | None = None,
) -> dict:
    color_name = f"{sample_id}_color.png"
    depth_name = f"{sample_id}_depth.npy"
    meta_name = f"{sample_id}.json"
    cv2.imwrite(str(out_dir / color_name), color)
    np.save(out_dir / depth_name, depth.astype(np.float32))
    meta = {
        "id": sample_id,
        "role": role,
        "kind": kind,
        "source": source,
        "boxes": boxes,
        "color": color_name,
        "depth": depth_name,
        "intrinsics": DEFAULT_K,
        "image_width": int(color.shape[1]),
        "image_height": int(color.shape[0]),
    }
    if note:
        meta["note"] = note
    (out_dir / meta_name).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _wide_box_aspect(box) -> float | None:
    if not box or len(box) != 4:
        return None
    _, _, w, h = [float(v) for v in box]
    if w <= 0 or h <= 0:
        return None
    return h / w


def collect_real_wide(max_n: int = 8) -> list[dict]:
    """Pull widest labeled stills from vial_dataset (usually squat top-down, not true sideways)."""
    found: list[tuple[float, Path, dict]] = []
    for folder in (DATASET / "raw", DATASET / "review"):
        if not folder.is_dir():
            continue
        for jp in folder.rglob("*.json"):
            try:
                meta = json.loads(jp.read_text(encoding="utf-8"))
            except Exception:
                continue
            box = meta.get("box")
            aspect = _wide_box_aspect(box)
            if aspect is None:
                continue
            # Wide: width clearly larger than height (h/w < 0.90)
            if aspect >= 0.90:
                continue
            img = jp.with_suffix(".jpg")
            if not img.exists():
                img = jp.with_suffix(".png")
            if not img.exists():
                continue
            found.append((aspect, img, meta))
    found.sort(key=lambda t: t[0])  # smallest h/w first (= widest)
    out = []
    for aspect, img, meta in found[:max_n]:
        out.append({"aspect": aspect, "image": img, "meta": meta})
    return out


def generate(seed: int = 42, wipe: bool = True) -> dict:
    if wipe and SYNTH.exists():
        # Keep only generated content; do not touch sibling race dirs.
        for child in SYNTH.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir() and child.name in {"images", "real_wide"}:
                shutil.rmtree(child)
    SYNTH.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w, h = 640, 480
    samples: list[dict] = []

    # ~12 left + ~12 right sideways positives
    for i in range(12):
        color, depth = _desk(w, h, rng)
        box = _paint_sideways(color, depth, lid_left=True, rng=rng)
        sid = f"sideways_left_{i:02d}"
        samples.append(
            _write_sample(SYNTH, sid, color, depth, role="positive", kind="sideways_left", boxes=[list(box)])
        )
    for i in range(12):
        color, depth = _desk(w, h, rng)
        box = _paint_sideways(color, depth, lid_left=False, rng=rng)
        sid = f"sideways_right_{i:02d}"
        samples.append(
            _write_sample(SYNTH, sid, color, depth, role="positive", kind="sideways_right", boxes=[list(box)])
        )

    # Hard negatives: top-down, upright, empty clutter
    for i in range(6):
        color, depth = _desk(w, h, rng)
        box = _paint_topdown(color, depth, rng)
        sid = f"topdown_{i:02d}"
        samples.append(
            _write_sample(
                SYNTH,
                sid,
                color,
                depth,
                role="negative",
                kind="topdown",
                boxes=[],
                note=f"control GT upright/top-down box={list(box)}; sideways detector must return []",
            )
        )
    for i in range(6):
        color, depth = _desk(w, h, rng)
        box = _paint_upright(color, depth, rng)
        sid = f"upright_{i:02d}"
        samples.append(
            _write_sample(
                SYNTH,
                sid,
                color,
                depth,
                role="negative",
                kind="upright",
                boxes=[],
                note=f"control GT upright box={list(box)}; sideways detector must return []",
            )
        )
    for i in range(8):
        color, depth = _desk(w, h, rng, base=int(rng.integers(70, 100)))
        _paint_empty_clutter(color, depth, rng)
        sid = f"empty_{i:02d}"
        samples.append(
            _write_sample(SYNTH, sid, color, depth, role="negative", kind="empty", boxes=[])
        )

    # Round-3 hard set: angled/degraded positives plus cap-like clutter negatives.
    # These IDs are additive, so old-set scores can be reproduced by excluding
    # entries whose kind starts with "hard_".
    hard_angles = (-30, -25, -20, -15, 15, 20, 25, 30)
    for side, lid_left in (("left", True), ("right", False)):
        for i, angle in enumerate(hard_angles):
            variant = i + (0 if lid_left else 8)
            base = int(rng.integers(42, 76)) if variant in {1, 5, 7, 11, 13, 15} else int(rng.integers(70, 100))
            color, depth = _desk(w, h, rng, base=base)
            box = _paint_hard_sideways(
                color,
                depth,
                lid_left=lid_left,
                angle=float(angle),
                variant=variant,
                rng=rng,
            )
            sid = f"hard_sideways_{side}_{i:02d}"
            samples.append(
                _write_sample(
                    SYNTH,
                    sid,
                    color,
                    depth,
                    role="positive",
                    kind=f"hard_sideways_{side}",
                    boxes=[list(box)],
                    note=(
                        f"Round-3 hard positive: angle={angle:+d}; partial occlusion, "
                        "dim/clutter and weak or missing depth vary by sample"
                    ),
                )
            )
    for i in range(8):
        color, depth = _desk(w, h, rng, base=int(rng.integers(48, 98)))
        _paint_hard_negative(color, depth, i, rng)
        sid = f"hard_negative_{i:02d}"
        samples.append(
            _write_sample(
                SYNTH,
                sid,
                color,
                depth,
                role="negative",
                kind="hard_negative",
                boxes=[],
                note="Round-3 cap-like white clutter with no attached clear sideways vial",
            )
        )

    # Real wide boxes from vial_dataset (hard negatives — none are true sideways aspect)
    real_dir = SYNTH / "real_wide"
    real_dir.mkdir(exist_ok=True)
    real_items = collect_real_wide()
    for i, item in enumerate(real_items):
        img_path: Path = item["image"]
        meta = item["meta"]
        color = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if color is None:
            continue
        hh, ww = color.shape[:2]
        depth = np.full((hh, ww), 900.0, dtype=np.float32)
        # Approximate lid depth on the labeled box (no real depth available).
        bx, by, bw, bh = [int(v) for v in meta["box"]]
        bx = max(0, min(bx, ww - 1))
        by = max(0, min(by, hh - 1))
        bw = max(1, min(bw, ww - bx))
        bh = max(1, min(bh, hh - by))
        depth[by : by + bh, bx : bx + bw] = float(meta.get("distance_mm") or 500.0)
        sid = f"real_wide_{i:02d}"
        # Store under synth/ but also copy jpg reference name into meta
        entry = _write_sample(
            SYNTH,
            sid,
            color,
            depth,
            role="negative",
            kind="real_wide",
            boxes=[],
            source=str(img_path.relative_to(CAMERA)).replace("\\", "/"),
            note=(
                f"vial_dataset wide box h/w={item['aspect']:.3f} gt={ [bx, by, bw, bh] }; "
                "not in sideways band (0.40–0.71) — treated as hard negative"
            ),
        )
        # Keep a copy of the original jpg for inspection
        shutil.copy2(img_path, real_dir / f"{sid}.jpg")
        samples.append(entry)

    manifest = {
        "seed": seed,
        "n_total": len(samples),
        "n_positive": sum(1 for s in samples if s["role"] == "positive"),
        "n_negative": sum(1 for s in samples if s["role"] == "negative"),
        "old_set_count": sum(1 for s in samples if not s["kind"].startswith("hard_")),
        "hard_set_count": sum(1 for s in samples if s["kind"].startswith("hard_")),
        "counts_by_kind": {},
        "samples": samples,
        "intrinsics_default": DEFAULT_K,
        "api": "detect_sideways(color_bgr, depth_mm, intrinsics=None) -> list[tuple[x,y,w,h]]",
        "iou_threshold": 0.5,
    }
    for s in samples:
        k = s["kind"]
        manifest["counts_by_kind"][k] = manifest["counts_by_kind"].get(k, 0) + 1
    (SYNTH / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    m = generate()
    print(
        f"Wrote {m['n_total']} samples -> {SYNTH} "
        f"(pos={m['n_positive']} neg={m['n_negative']}) kinds={m['counts_by_kind']}"
    )


if __name__ == "__main__":
    main()
