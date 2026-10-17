"""3D-prior identify helpers: score boxes vs fitted vial; live gate + soft rank.

Score candidate boxes against the fitted parametric vial, or lift a still's
box+distance and reproject into the peer camera. Uses ``vial_3d_model`` +
``vial_3d_dims.json``.

Live ``depthcamera.py`` compares YOLO/cap proposals to this prior at the
measured white-cap depth (default-on mild gate; disable with
``--no-3d-model-score``). Cap-first also uses lid diameter@Z for top-down
recovery without inventing upright cup boxes. Dual reproject assist remains
separate and still needs real stereo extrinsics (``--3d-reproject-assist``).

    cd E:\\Desktop\\Jessica\\Camera
    python vial_3d_identify.py score --box 100,120,48,70 --distance-mm 400
    python vial_3d_identify.py score-stills --limit 20
    python vial_3d_identify.py reproject-still --meta path\\to\\sidecar.json
    python depthcamera.py --dual-independent
    python depthcamera.py --dual-independent --no-3d-model-score
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable, Sequence

from vial_3d_model import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DIMS_PATH,
    EXTRINSICS_PATH,
    Intrinsics,
    VialDimensions,
    default_intrinsics,
    is_topdown_box_aspect,
    lid_width_match_score,
    lift_detection_to_3d,
    load_dims,
    load_extrinsics,
    reproject_vial_to_camera,
    score_box_vs_model,
    write_extrinsics_template,
    iter_metric_samples,
)

ROOT = Path(__file__).resolve().parent

# Mild live floor: reject only clear size outliers at measured Z.
# Softer than median GT (~0.30) so real vials at awkward / top-down angles
# still pass; ranking still prefers higher scores among survivors.
# Recall-oriented: keep low — filter never wipes to empty if all fail.
DEFAULT_LIVE_SCORE_MIN = 0.05
# Below this, treat score as "no useful prior" for ranking boost only.
RANK_SCORE_FLOOR = 0.0
# Lid width@Z used by cap-first top-down assist (fraction of lid Ø).
DEFAULT_LID_WIDTH_TOL = 0.45


def _parse_box(text: str) -> tuple[int, int, int, int]:
    parts = [int(v) for v in text.split(",")]
    if len(parts) != 4:
        raise SystemExit("--box must be x,y,w,h")
    return parts[0], parts[1], parts[2], parts[3]


def score_detection_vs_3d(
    box: tuple[int, int, int, int],
    distance_mm: float | None,
    K: Intrinsics | tuple[float, float, float, float] | None,
    dims: VialDimensions | None = None,
    *,
    image_size: tuple[int, int] | None = None,
) -> float | None:
    """Return 0..1 size match vs ``vial_3d_dims.json``, or None if Z/K missing."""
    if distance_mm is None or distance_mm <= 80:
        return None
    if K is None:
        if image_size is None:
            return None
        K = default_intrinsics(image_size[0], image_size[1])
    return float(score_box_vs_model(box, float(distance_mm), K, dims))


def lid_diameter_match_at_z(
    width_px: float,
    distance_mm: float | None,
    K: Intrinsics | tuple[float, float, float, float] | None,
    dims: VialDimensions | None = None,
    *,
    tol: float = DEFAULT_LID_WIDTH_TOL,
    image_size: tuple[int, int] | None = None,
) -> tuple[bool, float]:
    """Whether a lid blob width at Z matches fitted lid Ø; returns (ok, 0..1 score).

    Used by live cap-first top-down assist — does **not** invent a tall cup box.
    """
    if distance_mm is None or distance_mm <= 80 or width_px <= 0:
        return False, 0.0
    if K is None:
        if image_size is None:
            return False, 0.0
        K = default_intrinsics(image_size[0], image_size[1])
    if not isinstance(K, Intrinsics):
        from vial_3d_model import intrinsics_from_tuple

        K = intrinsics_from_tuple(K) or default_intrinsics(
            *(image_size or (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT))
        )
    dims = dims or load_dims()
    width_mm = float(width_px) * float(distance_mm) / K.fx
    score = lid_width_match_score(width_mm, dims)
    diam = dims.lid_diameter_mm
    ok = abs(width_mm - diam) / max(diam, 1.0) <= float(tol)
    return ok, float(score)


def passes_3d_model_prior(
    box: tuple[int, int, int, int],
    distance_mm: float | None,
    K: Intrinsics | tuple[float, float, float, float] | None,
    *,
    dims: VialDimensions | None = None,
    min_score: float = DEFAULT_LIVE_SCORE_MIN,
    image_size: tuple[int, int] | None = None,
) -> bool:
    """True when score is unavailable (skip gate) or meets the mild floor."""
    score = score_detection_vs_3d(
        box, distance_mm, K, dims, image_size=image_size
    )
    if score is None:
        return True
    return score >= float(min_score)


def filter_boxes_by_3d_model(
    boxes: Sequence[tuple[int, int, int, int]],
    *,
    distance_for_box: Callable[[tuple[int, int, int, int]], float | None],
    K: Intrinsics | tuple[float, float, float, float] | None,
    dims: VialDimensions | None = None,
    min_score: float = DEFAULT_LIVE_SCORE_MIN,
    image_size: tuple[int, int] | None = None,
) -> list[tuple[int, int, int, int]]:
    """Keep boxes that match the 3D prior; mild fallback if all fail.

    - No distance → keep (cannot compare).
    - Score ≥ min_score → keep.
    - If every scored box fails, keep the single best-scoring box (down-rank
      instead of wiping the frame — preserves recall for the mild default).
    """
    if not boxes:
        return []
    dims = dims or load_dims()
    scored: list[tuple[tuple[int, int, int, int], float | None]] = []
    for box in boxes:
        scored.append(
            (
                box,
                score_detection_vs_3d(
                    box, distance_for_box(box), K, dims, image_size=image_size
                ),
            )
        )
    kept = [
        box
        for box, score in scored
        if score is None or score >= float(min_score)
    ]
    if kept:
        return kept
    with_scores = [(box, score) for box, score in scored if score is not None]
    if not with_scores:
        return list(boxes)
    best_box, _ = max(with_scores, key=lambda item: item[1])
    return [best_box]


def rank_key_3d_model(
    box: tuple[int, int, int, int],
    *,
    distance_mm: float | None,
    K: Intrinsics | tuple[float, float, float, float] | None,
    dims: VialDimensions | None = None,
    image_size: tuple[int, int] | None = None,
) -> float:
    """Higher is better for ``max(..., key=)`` among vial candidates."""
    score = score_detection_vs_3d(
        box, distance_mm, K, dims, image_size=image_size
    )
    if score is None:
        return RANK_SCORE_FLOOR
    return float(score)


def cmd_score(args: argparse.Namespace) -> int:
    dims = load_dims()
    box = _parse_box(args.box)
    K = default_intrinsics(args.width, args.height)
    score = score_box_vs_model(box, args.distance_mm, K, dims)
    print(f"dims file: {DIMS_PATH}")
    print(
        f"template: lid diam {dims.lid_diameter_mm} x H{dims.lid_height_mm} | "
        f"cup diam {dims.cup_diameter_mm} x H{dims.cup_height_mm} | "
        f"total H {dims.total_height_mm} mm (n={dims.sample_count})"
    )
    print(f"box={box}  Z={args.distance_mm} mm")
    print(f"3D prior score: {score:.3f}  (1.0 ~= perfect size match at this Z)")
    if score < 0.35:
        print("hint: low score - wrong object, bad Z, or box not full lid->cup")
    return 0


def cmd_score_stills(args: argparse.Namespace) -> int:
    dims = load_dims()
    samples = iter_metric_samples()
    if not samples:
        print("No stills with box + distance_mm found under vial_dataset/")
        return 1
    limit = args.limit if args.limit and args.limit > 0 else len(samples)
    scores: list[float] = []
    print(f"scoring up to {limit} of {len(samples)} metric stills vs {DIMS_PATH.name}")
    for i, sample in enumerate(samples[:limit]):
        K = default_intrinsics(sample["image_width"], sample["image_height"])
        s = score_box_vs_model(sample["box"], sample["distance_mm"], K, dims)
        scores.append(s)
        if args.verbose:
            print(
                f"  {s:.3f}  Z={sample['distance_mm']:.0f}  "
                f"box={sample['box']}  {Path(sample['path']).name}"
            )
    print(
        f"scores: n={len(scores)}  median={statistics.median(scores):.3f}  "
        f"mean={statistics.mean(scores):.3f}  "
        f"min={min(scores):.3f}  max={max(scores):.3f}"
    )
    print(
        "interpretation: high median on GT positives => size prior matches labels; "
        "use score CLI on suspect YOLO boxes to reject size outliers."
    )
    return 0


def cmd_reproject_still(args: argparse.Namespace) -> int:
    meta_path = Path(args.meta)
    if not meta_path.exists():
        raise SystemExit(f"missing meta: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    box = meta.get("box")
    dist = meta.get("distance_mm")
    if box is None or dist is None or len(box) != 4:
        raise SystemExit("sidecar needs numeric box[4] and distance_mm")
    box_t = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    distance_mm = float(dist)
    width = int(meta.get("image_width") or DEFAULT_IMAGE_WIDTH)
    height = int(meta.get("image_height") or DEFAULT_IMAGE_HEIGHT)
    src_cam = int(meta.get("camera") if meta.get("camera") is not None else args.from_camera)

    dims = load_dims()
    write_extrinsics_template()
    ext = load_extrinsics()
    if ext is None:
        print(f"No extrinsics at {EXTRINSICS_PATH}")
        return 1

    K0 = ext.K0 or default_intrinsics(width, height)
    K1 = ext.K1 or default_intrinsics(width, height)
    dst_cam = args.to_camera
    hyp = lift_detection_to_3d(
        box_t, distance_mm, K0 if src_cam == 0 else K1, dims, source_camera=src_cam
    )
    if hyp is None:
        print("Could not lift detection (check distance_mm)")
        return 1
    target_K = K1 if dst_cam == 1 else K0
    proj = reproject_vial_to_camera(
        hyp,
        ext,
        target_K,
        (width, height),
        source_camera=src_cam,
        target_camera=dst_cam,
    )
    print(f"source cam{src_cam} box={box_t} Z={distance_mm} mm")
    print(f"cap_xyz_mm={hyp.cap_xyz_mm}")
    print(f"reprojected cam{dst_cam} box={proj}")
    print(
        f"source 3D score={score_box_vs_model(box_t, distance_mm, K0 if src_cam == 0 else K1, dims):.3f}"
    )
    notes = ""
    if EXTRINSICS_PATH.exists():
        data = json.loads(EXTRINSICS_PATH.read_text(encoding="utf-8"))
        notes = str(data.get("notes", ""))
    if "PLACEHOLDER" in notes.upper() or "placeholder" in notes.lower():
        print(
            "WARNING: stereo_extrinsics.json is still a PLACEHOLDER "
            "(identity R + ~120mm t). Reprojected box is illustrative only - "
            "run: python stereo_calib_stub.py howto"
        )
    return 0 if proj is not None else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score / reproject with the parametric 3D vial prior (standalone)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_score = sub.add_parser("score", help="Score one x,y,w,h + Z against vial_3d_dims.json")
    p_score.add_argument("--box", required=True, help="x,y,w,h pixels")
    p_score.add_argument("--distance-mm", type=float, required=True)
    p_score.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH)
    p_score.add_argument("--height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    p_score.set_defaults(func=cmd_score)

    p_stills = sub.add_parser(
        "score-stills",
        help="Score labeled stills that have box + distance_mm (sanity-check the prior)",
    )
    p_stills.add_argument("--limit", type=int, default=30, help="0 = all")
    p_stills.add_argument("-v", "--verbose", action="store_true")
    p_stills.set_defaults(func=cmd_score_stills)

    p_rep = sub.add_parser(
        "reproject-still",
        help="Lift box+distance from a JSON sidecar and reproject to the other cam",
    )
    p_rep.add_argument("--meta", required=True, help="path to *.json sidecar")
    p_rep.add_argument("--from-camera", type=int, default=0)
    p_rep.add_argument("--to-camera", type=int, default=1)
    p_rep.set_defaults(func=cmd_reproject_still)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
