"""Parametric 3D vial model (white lid + clear cup) from labeled stills.

Fits millimeter dimensions from YOLO boxes + white-cap distance sidecars, then
lifts a strong-camera detection into 3D and reprojects into another camera when
stereo extrinsics are available.

This is orthogonal to YOLO fine-tunes. Live locate compares detections to the
fitted size prior by default (``--no-3d-model-score`` to disable). Dual
reproject assist stays opt-in via ``--3d-reproject-assist`` (needs real extrinsics).

``stereo_extrinsics.json`` ships as a PLACEHOLDER (identity R + ~120 mm t) until
you run a ChArUco/chessboard stereo calibrate — see ``stereo_calib_stub.py howto``
and ``VIAL_3D_README.md``.

    cd E:\\Desktop\\Jessica\\Camera
    python vial_3d_model.py fit
    python vial_3d_model.py show
    python vial_3d_model.py preview
    python vial_3d_model.py view          # interactive 3D (drag to rotate)
    python vial_3d_model.py export-mesh   # OBJ + USDA for Isaac Sim / Omniverse
    python vial_3d_model.py reproject-demo --box 100,120,48,70 --distance-mm 400
    python vial_3d_identify.py score --box 100,120,48,70 --distance-mm 400
    python vial_3d_identify.py score-stills --limit 20
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW_DATASET = ROOT / "vial_dataset" / "raw"
REVIEW_ROOT = ROOT / "vial_dataset" / "review"
DIMS_PATH = ROOT / "vial_3d_dims.json"
EXTRINSICS_PATH = ROOT / "stereo_extrinsics.json"
PREVIEW_PATH = ROOT / "vial_3d_preview.png"
PREVIEW_SIDE_PATH = ROOT / "vial_3d_preview_side.png"
ASSET_DIR = ROOT / "assets" / "vial"
OBJ_PATH = ASSET_DIR / "vial.obj"
MTL_PATH = ASSET_DIR / "vial.mtl"
USDA_PATH = ASSET_DIR / "vial.usda"

# Typical Gemini 335L color FOV prior when OpenCV intrinsics are unavailable.
DEFAULT_HFOV_DEG = 90.0
DEFAULT_IMAGE_WIDTH = 1280
DEFAULT_IMAGE_HEIGHT = 800

# Seed prior for a small white-capped clear sample vial (overwritten by fit).
DEFAULT_LID_DIAMETER_MM = 28.0
DEFAULT_LID_HEIGHT_MM = 12.0
DEFAULT_CUP_DIAMETER_MM = 25.0
DEFAULT_CUP_HEIGHT_MM = 45.0

# Full-vial YOLO box: lid occupies roughly this fraction of height (white band).
LID_HEIGHT_FRAC = 0.22


@dataclass
class VialDimensions:
    """Axis-aligned stacked cylinders in millimeters."""

    lid_diameter_mm: float = DEFAULT_LID_DIAMETER_MM
    lid_height_mm: float = DEFAULT_LID_HEIGHT_MM
    cup_diameter_mm: float = DEFAULT_CUP_DIAMETER_MM
    cup_height_mm: float = DEFAULT_CUP_HEIGHT_MM
    sample_count: int = 0
    notes: str = ""

    @property
    def total_height_mm(self) -> float:
        return float(self.lid_height_mm + self.cup_height_mm)

    @property
    def max_diameter_mm(self) -> float:
        return float(max(self.lid_diameter_mm, self.cup_diameter_mm))


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.fx, self.fy, self.cx, self.cy

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class VialHypothesis3D:
    """Cap-center pose in a camera frame + template dimensions."""

    cap_xyz_mm: tuple[float, float, float]
    dims: VialDimensions
    source_camera: int | None = None
    # Image-up is -Y in the pinhole camera frame used by position_mm.
    axis_cam: tuple[float, float, float] = (0.0, -1.0, 0.0)


@dataclass
class StereoExtrinsics:
    """Maps points from camera ``from_camera`` into ``to_camera``: X_to = R @ X_from + t."""

    R: np.ndarray
    t_mm: np.ndarray
    from_camera: int = 0
    to_camera: int = 1
    K0: Intrinsics | None = None
    K1: Intrinsics | None = None

    def transform_point(self, xyz_mm: np.ndarray) -> np.ndarray:
        return self.R @ np.asarray(xyz_mm, dtype=np.float64).reshape(3) + self.t_mm


def default_intrinsics(
    width: int = DEFAULT_IMAGE_WIDTH, height: int = DEFAULT_IMAGE_HEIGHT
) -> Intrinsics:
    hfov = math.radians(DEFAULT_HFOV_DEG)
    fx = width / (2.0 * math.tan(hfov / 2.0))
    fy = fx
    return Intrinsics(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0)


def intrinsics_from_tuple(values) -> Intrinsics | None:
    if values is None:
        return None
    fx, fy, cx, cy = values
    if fx <= 1 or fy <= 1:
        return None
    return Intrinsics(float(fx), float(fy), float(cx), float(cy))


def pixel_to_cam(
    u: float, v: float, z_mm: float, K: Intrinsics
) -> tuple[float, float, float]:
    x = (u - K.cx) * z_mm / K.fx
    y = (v - K.cy) * z_mm / K.fy
    return float(x), float(y), float(z_mm)


def cam_to_pixel(xyz_mm: np.ndarray, K: Intrinsics) -> tuple[float, float] | None:
    x, y, z = [float(v) for v in xyz_mm]
    if z <= 1e-3:
        return None
    return K.fx * x / z + K.cx, K.fy * y / z + K.cy


def yolo_txt_to_xywh(line: str, width: int, height: int) -> tuple[int, int, int, int] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    _, xc, yc, bw, bh = parts[:5]
    xc_f, yc_f, bw_f, bh_f = map(float, (xc, yc, bw, bh))
    w = max(1, int(round(bw_f * width)))
    h = max(1, int(round(bh_f * height)))
    x = int(round(xc_f * width - w / 2))
    y = int(round(yc_f * height - h / 2))
    return x, y, w, h


def load_dims(path: Path | None = None) -> VialDimensions:
    path = path or DIMS_PATH
    if not path.exists():
        return VialDimensions(notes="defaults (run: python vial_3d_model.py fit)")
    data = json.loads(path.read_text(encoding="utf-8"))
    return VialDimensions(
        lid_diameter_mm=float(data.get("lid_diameter_mm", DEFAULT_LID_DIAMETER_MM)),
        lid_height_mm=float(data.get("lid_height_mm", DEFAULT_LID_HEIGHT_MM)),
        cup_diameter_mm=float(data.get("cup_diameter_mm", DEFAULT_CUP_DIAMETER_MM)),
        cup_height_mm=float(data.get("cup_height_mm", DEFAULT_CUP_HEIGHT_MM)),
        sample_count=int(data.get("sample_count", 0)),
        notes=str(data.get("notes", "")),
    )


def save_dims(dims: VialDimensions, path: Path | None = None) -> Path:
    path = path or DIMS_PATH
    payload = asdict(dims)
    payload["total_height_mm"] = dims.total_height_mm
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def is_extrinsics_placeholder(path: Path | None = None) -> bool:
    """True if file is missing, notes say PLACEHOLDER, or R is identity/near-identity.

    Live ``--3d-reproject-assist`` must refuse placeholder R,t (would mis-aim).
    Offline demos may still load the file for illustration.
    """
    path = path or EXTRINSICS_PATH
    if not path.exists():
        return True
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    notes = str(data.get("notes", "")).upper()
    if "PLACEHOLDER" in notes:
        return True
    R = np.asarray(data.get("R", []), dtype=np.float64).reshape(-1)
    if R.size == 9:
        eye = np.eye(3).reshape(-1)
        if float(np.linalg.norm(R - eye)) < 1e-6:
            return True
    return False


def extrinsics_reproject_state(path: Path | None = None) -> str:
    """Return ``missing`` | ``placeholder`` | ``ready`` for live cross-cam assist."""
    path = path or EXTRINSICS_PATH
    if not path.exists():
        return "missing"
    if is_extrinsics_placeholder(path):
        return "placeholder"
    return "ready"


def load_extrinsics(path: Path | None = None) -> StereoExtrinsics | None:
    path = path or EXTRINSICS_PATH
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    R = np.asarray(data["R"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(data["t_mm"], dtype=np.float64).reshape(3)
    K0 = K1 = None
    if data.get("K0") is not None:
        K0 = Intrinsics(*[float(v) for v in data["K0"]])
    if data.get("K1") is not None:
        K1 = Intrinsics(*[float(v) for v in data["K1"]])
    return StereoExtrinsics(
        R=R,
        t_mm=t,
        from_camera=int(data.get("from_camera", 0)),
        to_camera=int(data.get("to_camera", 1)),
        K0=K0,
        K1=K1,
    )


def write_extrinsics_template(path: Path | None = None) -> Path:
    """Identity stub so the file format is discoverable before ChArUco calib."""
    path = path or EXTRINSICS_PATH
    if path.exists():
        return path
    K = default_intrinsics()
    payload = {
        "from_camera": 0,
        "to_camera": 1,
        "R": np.eye(3).tolist(),
        "t_mm": [120.0, 0.0, 0.0],
        "K0": [K.fx, K.fy, K.cx, K.cy],
        "K1": [K.fx, K.fy, K.cx, K.cy],
        "notes": (
            "PLACEHOLDER — replace R,t after ChArUco/chessboard stereoCalibrate. "
            "t_mm is a rough baseline guess along +X; identity R is wrong for angled mounts."
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def iter_metric_samples() -> list[dict]:
    """Collect box + distance_mm samples from review sidecars and enriched raw JSON."""
    samples: list[dict] = []
    roots = [REVIEW_ROOT, RAW_DATASET]
    for root in roots:
        if not root.exists():
            continue
        for meta_path in root.rglob("*.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            dist = meta.get("distance_mm")
            box = meta.get("box")
            if dist is None or box is None or len(box) != 4:
                continue
            try:
                distance_mm = float(dist)
                x, y, w, h = [int(v) for v in box]
            except (TypeError, ValueError):
                continue
            if distance_mm <= 80 or w < 8 or h < 8:
                continue
            width = int(meta.get("image_width") or DEFAULT_IMAGE_WIDTH)
            height = int(meta.get("image_height") or DEFAULT_IMAGE_HEIGHT)
            # Prefer paired image size when present.
            img = meta_path.with_suffix(".jpg")
            if img.exists():
                im = cv2.imread(str(img), cv2.IMREAD_COLOR)
                if im is not None:
                    height, width = im.shape[:2]
            samples.append(
                {
                    "path": str(meta_path),
                    "box": (x, y, w, h),
                    "distance_mm": distance_mm,
                    "image_width": width,
                    "image_height": height,
                    "camera": meta.get("camera"),
                }
            )
    return samples


def _aspect_samples_from_raw() -> list[float]:
    """Full-vial box aspects from raw YOLO labels (no depth required)."""
    aspects: list[float] = []
    if not RAW_DATASET.exists():
        return aspects
    for txt in RAW_DATASET.glob("*.txt"):
        text = txt.read_text(encoding="utf-8").strip()
        if not text:
            continue
        img = txt.with_suffix(".jpg")
        if not img.exists():
            continue
        im = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if im is None:
            continue
        h, w = im.shape[:2]
        box = yolo_txt_to_xywh(text.splitlines()[0], w, h)
        if box is None:
            continue
        _, _, bw, bh = box
        aspect = bh / max(bw, 1)
        if 0.9 <= aspect <= 1.8:
            aspects.append(aspect)
    return aspects


def fit_dimensions_from_stills(
    *,
    out_path: Path | None = None,
    min_samples: int = 5,
) -> VialDimensions:
    """Estimate lid/cup mm sizes from labeled stills with white-cap distance."""
    samples = iter_metric_samples()
    widths: list[float] = []
    heights: list[float] = []
    for sample in samples:
        x, y, w, h = sample["box"]
        z = sample["distance_mm"]
        K = default_intrinsics(sample["image_width"], sample["image_height"])
        widths.append(w * z / K.fx)
        heights.append(h * z / K.fy)

    aspects = _aspect_samples_from_raw()
    median_aspect = statistics.median(aspects) if aspects else 1.35

    if len(widths) >= min_samples:
        diam = float(statistics.median(widths))
        total_h = float(statistics.median(heights))
        # Clamp to physical vial band used by vial_scale.
        diam = min(max(diam, 14.0), 110.0)
        total_h = min(max(total_h, 22.0), 170.0)
        lid_h = max(6.0, total_h * LID_HEIGHT_FRAC)
        cup_h = max(12.0, total_h - lid_h)
        cup_d = max(12.0, diam * 0.92)
        dims = VialDimensions(
            lid_diameter_mm=round(diam, 2),
            lid_height_mm=round(lid_h, 2),
            cup_diameter_mm=round(cup_d, 2),
            cup_height_mm=round(cup_h, 2),
            sample_count=len(widths),
            notes=(
                f"fit from {len(widths)} distance sidecars; "
                f"raw aspect median={median_aspect:.2f} (n={len(aspects)})"
            ),
        )
    else:
        # Aspect-only: keep default diameter, scale height from median aspect.
        total_h = DEFAULT_LID_DIAMETER_MM * median_aspect
        lid_h = max(6.0, total_h * LID_HEIGHT_FRAC)
        dims = VialDimensions(
            lid_diameter_mm=DEFAULT_LID_DIAMETER_MM,
            lid_height_mm=round(lid_h, 2),
            cup_diameter_mm=DEFAULT_CUP_DIAMETER_MM,
            cup_height_mm=round(max(12.0, total_h - lid_h), 2),
            sample_count=len(widths),
            notes=(
                f"few metric samples ({len(widths)}); used defaults + raw aspect "
                f"median={median_aspect:.2f}. Re-run after R-accept keeps distance_mm."
            ),
        )

    save_dims(dims, out_path)
    write_extrinsics_template()
    return dims


def lift_detection_to_3d(
    box: tuple[int, int, int, int],
    distance_mm: float,
    K: Intrinsics | tuple[float, float, float, float],
    dims: VialDimensions | None = None,
    *,
    source_camera: int | None = None,
    cap_center_uv: tuple[float, float] | None = None,
) -> VialHypothesis3D | None:
    """Build a 3D vial hypothesis from a 2D box + white-cap depth."""
    if distance_mm is None or distance_mm <= 80:
        return None
    if not isinstance(K, Intrinsics):
        K = intrinsics_from_tuple(K) or default_intrinsics()
    dims = dims or load_dims()
    x, y, w, h = [int(v) for v in box]
    if cap_center_uv is None:
        # Prefer upper white-cap band center (matches measure_cap_mm bias).
        u = x + w / 2.0
        v = y + max(2.0, h * LID_HEIGHT_FRAC * 0.5)
    else:
        u, v = cap_center_uv
    xyz = pixel_to_cam(u, v, float(distance_mm), K)
    return VialHypothesis3D(
        cap_xyz_mm=xyz, dims=dims, source_camera=source_camera
    )


def _vial_corners_cam(hyp: VialHypothesis3D) -> np.ndarray:
    """8-ish silhouette corners of lid+cup AABB in the hypothesis camera frame."""
    cx, cy, cz = hyp.cap_xyz_mm
    axis = np.asarray(hyp.axis_cam, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-6)
    # Build a local frame with axis as "up" along the vial (lid → cup is +axis? 
    # Cap center is at the lid; cup extends opposite to image-up when lid is on top
    # in the image, i.e. along +Y if axis is -Y... Wait:
    # axis_cam default (0,-1,0) points image-up. Lid sits at cap; cup extends
    # toward image-down = -axis.
    down = -axis
    # Arbitrary orthonormal basis for radial directions.
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    r1 = np.cross(axis, helper)
    r1 = r1 / max(np.linalg.norm(r1), 1e-6)
    r2 = np.cross(axis, r1)

    d = hyp.dims
    lid_r = d.lid_diameter_mm / 2.0
    cup_r = d.cup_diameter_mm / 2.0
    points: list[np.ndarray] = []
    # Lid disk at cap plane and lid bottom.
    for along in (0.0, d.lid_height_mm):
        center = np.array([cx, cy, cz]) + down * along
        for ang in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
            radial = math.cos(ang) * r1 + math.sin(ang) * r2
            points.append(center + lid_r * radial)
    # Cup from lid bottom to cup bottom.
    for along in (d.lid_height_mm, d.total_height_mm):
        center = np.array([cx, cy, cz]) + down * along
        for ang in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
            radial = math.cos(ang) * r1 + math.sin(ang) * r2
            points.append(center + cup_r * radial)
    return np.asarray(points, dtype=np.float64)


def project_points_to_box(
    points_mm: np.ndarray,
    K: Intrinsics,
    image_size: tuple[int, int],
    *,
    pad_px: int = 6,
) -> tuple[int, int, int, int] | None:
    """Axis-aligned pixel box from 3D points; None if behind camera / empty."""
    width, height = image_size
    us: list[float] = []
    vs: list[float] = []
    for p in points_mm:
        uv = cam_to_pixel(p, K)
        if uv is None:
            continue
        us.append(uv[0])
        vs.append(uv[1])
    if len(us) < 3:
        return None
    x1 = int(math.floor(min(us))) - pad_px
    y1 = int(math.floor(min(vs))) - pad_px
    x2 = int(math.ceil(max(us))) + pad_px
    y2 = int(math.ceil(max(vs))) + pad_px
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    bw, bh = x2 - x1, y2 - y1
    if bw < 8 or bh < 8:
        return None
    return x1, y1, bw, bh


def reproject_vial_to_camera(
    hyp: VialHypothesis3D,
    extrinsics: StereoExtrinsics,
    target_K: Intrinsics | tuple[float, float, float, float],
    image_size: tuple[int, int],
    *,
    source_camera: int | None = None,
    target_camera: int | None = None,
) -> tuple[int, int, int, int] | None:
    """Project a 3D vial hypothesis into another camera's pixel box.

    ``extrinsics`` must map ``from_camera`` → ``to_camera``. If the hypothesis
    was built in ``to_camera``, this returns None (call with inverted extrinsics
    or swap roles).
    """
    if not isinstance(target_K, Intrinsics):
        target_K = intrinsics_from_tuple(target_K) or default_intrinsics(*image_size)

    src = source_camera if source_camera is not None else hyp.source_camera
    dst = target_camera
    if src is not None and src != extrinsics.from_camera:
        # Only identity-invert when R is orthogonal; for MVP require matching direction.
        if src == extrinsics.to_camera and dst == extrinsics.from_camera:
            R_inv = extrinsics.R.T
            t_inv = -R_inv @ extrinsics.t_mm
            corners = _vial_corners_cam(hyp)
            moved = (R_inv @ corners.T).T + t_inv
            return project_points_to_box(moved, target_K, image_size)
        if src != extrinsics.from_camera:
            return None

    corners = _vial_corners_cam(hyp)
    moved = np.asarray([extrinsics.transform_point(p) for p in corners])
    return project_points_to_box(moved, target_K, image_size)


def is_topdown_box_aspect(box: tuple[int, int, int, int]) -> bool:
    """Nearly square box → likely top-down / near-straight-up lid face (not tall cup)."""
    _, _, w, h = box
    if w < 1 or h < 1:
        return False
    aspect = h / float(w)
    return 0.72 <= aspect <= 1.28


def lid_width_match_score(
    width_mm: float,
    dims: VialDimensions | None = None,
    *,
    diameter_mm: float | None = None,
) -> float:
    """0..1: how well an observed width at Z matches the fitted lid diameter."""
    dims = dims or load_dims()
    diam = float(diameter_mm if diameter_mm is not None else dims.lid_diameter_mm)
    if diam <= 1.0 or width_mm <= 0:
        return 0.0
    err = abs(float(width_mm) - diam) / diam
    return float(math.exp(-3.0 * err))


def score_box_vs_model(
    box: tuple[int, int, int, int],
    distance_mm: float,
    K: Intrinsics | tuple[float, float, float, float],
    dims: VialDimensions | None = None,
) -> float:
    """0..1 score: how well box width/height at Z match the fitted template.

    Top-down / near-straight-up boxes (nearly square) are scored mainly on
    **lid diameter**, not full upright cup height — otherwise a true lid face
    is punished for missing the invented tall lid→cup silhouette.
    """
    dims = dims or load_dims()
    if not isinstance(K, Intrinsics):
        K = intrinsics_from_tuple(K) or default_intrinsics()
    if distance_mm is None or distance_mm <= 80:
        return 0.0
    _, _, w, h = box
    w_mm = w * distance_mm / K.fx
    h_mm = h * distance_mm / K.fy
    diam = dims.max_diameter_mm
    diam_err = abs(w_mm - diam) / max(diam, 1.0)
    # Face-on lid: expect ~Ø×Ø (or slightly taller with rim), not total H.
    if is_topdown_box_aspect(box) and h_mm < dims.total_height_mm * 0.88:
        face_err = abs(h_mm - diam) / max(diam, 1.0)
        return float(math.exp(-2.5 * (0.8 * diam_err + 0.2 * face_err)))
    h_err = abs(h_mm - dims.total_height_mm) / max(dims.total_height_mm, 1.0)
    # Soft exponential: within ~25% → high score.
    return float(math.exp(-2.5 * (diam_err + h_err)))


def expected_lid_width_px(
    distance_mm: float,
    K: Intrinsics | tuple[float, float, float, float],
    dims: VialDimensions | None = None,
) -> float:
    """Expected white-cap diameter in pixels at depth Z."""
    dims = dims or load_dims()
    if not isinstance(K, Intrinsics):
        K = intrinsics_from_tuple(K) or default_intrinsics()
    z = max(float(distance_mm), 1.0)
    return float(dims.lid_diameter_mm * K.fx / z)


def expected_wh_px(
    distance_mm: float,
    K: Intrinsics | tuple[float, float, float, float],
    dims: VialDimensions | None = None,
) -> tuple[float, float]:
    """Expected full-vial pixel width/height at white-cap depth Z."""
    dims = dims or load_dims()
    if not isinstance(K, Intrinsics):
        K = intrinsics_from_tuple(K) or default_intrinsics()
    z = max(float(distance_mm), 1.0)
    w_px = dims.max_diameter_mm * K.fx / z
    h_px = dims.total_height_mm * K.fy / z
    return float(w_px), float(h_px)


def box_from_cap_center(
    u: float,
    v: float,
    distance_mm: float,
    K: Intrinsics | tuple[float, float, float, float],
    dims: VialDimensions | None = None,
    *,
    image_size: tuple[int, int],
    pad_px: int = 4,
) -> tuple[int, int, int, int] | None:
    """Project the parametric vial into a pixel box given white-cap center + Z.

    This is the single-cam CV primitive: depth finds the lid, the 3D model
    supplies the full lid→cup box at that pose (no stereo needed).
    """
    if distance_mm is None or distance_mm <= 80:
        return None
    if not isinstance(K, Intrinsics):
        K = intrinsics_from_tuple(K) or default_intrinsics(*image_size)
    dims = dims or load_dims()
    xyz = pixel_to_cam(float(u), float(v), float(distance_mm), K)
    hyp = VialHypothesis3D(cap_xyz_mm=xyz, dims=dims)
    return project_points_to_box(
        _vial_corners_cam(hyp), K, image_size, pad_px=pad_px
    )


def _nearest_valid_depth_mm(depth_mm: np.ndarray, u: int, v: int, radius: int = 4) -> float | None:
    """Median of finite positive depths in a small window around (u, v)."""
    height, width = depth_mm.shape[:2]
    x0 = max(0, u - radius)
    y0 = max(0, v - radius)
    x1 = min(width, u + radius + 1)
    y1 = min(height, v + radius + 1)
    patch = depth_mm[y0:y1, x0:x1].astype(np.float32).ravel()
    finite = patch[np.isfinite(patch) & (patch > 80.0) & (patch < 2500.0)]
    if finite.size < 3:
        return None
    return float(np.median(finite))


def propose_boxes_from_3d_model(
    image: np.ndarray,
    depth_mm: np.ndarray,
    K: Intrinsics | tuple[float, float, float, float] | None,
    dims: VialDimensions | None = None,
    *,
    max_proposals: int = 6,
    lid_width_tol: float = 0.45,
) -> list[tuple[int, int, int, int]]:
    """Propose full-vial boxes from white lids + depth using the fitted 3D size.

    Pipeline (single-cam, no stereo):
      1. Find bright low-saturation lid blobs.
      2. Read white-cap depth under each blob.
      3. Keep lids whose width in mm is near ``lid_diameter_mm``.
      4. Project the parametric lid+cup cylinders into a pixel box at that UVZ.

    Caller must still run clear-vial / soft-clear / scale gates — this only proposes.
    """
    dims = dims or load_dims()
    height, width = image.shape[:2]
    if depth_mm.shape[0] != height or depth_mm.shape[1] != width:
        depth_mm = cv2.resize(
            depth_mm.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    if K is None:
        K = default_intrinsics(width, height)
    elif not isinstance(K, Intrinsics):
        K = intrinsics_from_tuple(K) or default_intrinsics(width, height)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    cap_mask = ((val >= 165) & (sat <= 55)).astype(np.uint8) * 255
    cap_mask = cv2.morphologyEx(
        cap_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)),
    )
    contours, _ = cv2.findContours(cap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_w = max(14, int(width * 0.03))
    ranked: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < min_w or bh < 2 or bw > 0.65 * width:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 1.4 or aspect > 14.0:
            continue
        u = x + bw * 0.5
        v = y + bh * 0.5
        z = _nearest_valid_depth_mm(depth_mm, int(u), int(v), radius=max(3, bw // 8))
        if z is None:
            continue
        width_mm = bw * z / K.fx
        diam = dims.lid_diameter_mm
        if abs(width_mm - diam) / max(diam, 1.0) > lid_width_tol:
            continue
        box = box_from_cap_center(
            u, v, z, K, dims, image_size=(width, height), pad_px=4
        )
        if box is None:
            continue
        score = score_box_vs_model(box, z, K, dims)
        # Prefer lids that match diameter tightly and score high at Z.
        diam_match = 1.0 - min(1.0, abs(width_mm - diam) / max(diam, 1.0))
        ranked.append((0.55 * score + 0.45 * diam_match, box))

    if not ranked:
        return []
    ranked.sort(key=lambda item: item[0], reverse=True)
    # Deduplicate overlapping proposals (keep highest score).
    kept: list[tuple[int, int, int, int]] = []
    for _, box in ranked:
        bx, by, bw, bh = box
        overlap = False
        for ox, oy, ow, oh in kept:
            x1, y1 = max(bx, ox), max(by, oy)
            x2, y2 = min(bx + bw, ox + ow), min(by + bh, oy + oh)
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            union = bw * bh + ow * oh - inter
            if union > 0 and inter / union >= 0.45:
                overlap = True
                break
        if not overlap:
            kept.append(box)
        if len(kept) >= max_proposals:
            break
    return kept


def _cylinder_solid_mesh(
    radius: float,
    z0: float,
    z1: float,
    *,
    n_theta: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed cylinder mesh (side + top + bottom). Vertices in mm, Z-up."""
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    bottom = np.column_stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.full(n_theta, z0)]
    )
    top = np.column_stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.full(n_theta, z1)]
    )
    center_b = np.array([[0.0, 0.0, z0]])
    center_t = np.array([[0.0, 0.0, z1]])
    verts = np.vstack([bottom, top, center_b, center_t])
    cb = 2 * n_theta
    ct = cb + 1
    faces: list[list[int]] = []
    for i in range(n_theta):
        j = (i + 1) % n_theta
        faces.append([i, j, n_theta + j])
        faces.append([i, n_theta + j, n_theta + i])
        faces.append([cb, j, i])
        faces.append([ct, n_theta + i, n_theta + j])
    return verts, np.asarray(faces, dtype=np.int32)


def export_vial_mesh(
    dims: VialDimensions | None = None,
    *,
    out_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Write OBJ+MTL (meters) and USDA for Isaac Sim / Omniverse import.

    Returns ``(obj_path, mtl_path, usda_path)``.
    """
    dims = dims or load_dims()
    out_dir = out_dir or ASSET_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / "vial.obj"
    mtl_path = out_dir / "vial.mtl"
    usda_path = out_dir / "vial.usda"

    cup_r = dims.cup_diameter_mm * 0.5
    lid_r = dims.lid_diameter_mm * 0.5
    cup_verts, cup_faces = _cylinder_solid_mesh(cup_r, 0.0, dims.cup_height_mm)
    lid_verts, lid_faces = _cylinder_solid_mesh(
        lid_r, dims.cup_height_mm, dims.total_height_mm
    )

    # OBJ uses meters (Isaac Lab MeshConverter expects SI).
    scale = 0.001
    mtl_path.write_text(
        "\n".join(
            [
                "newmtl cup_clear",
                "Kd 0.55 0.72 0.82",
                "d 0.35",
                "illum 2",
                "",
                "newmtl lid_white",
                "Kd 0.96 0.96 0.97",
                "d 1.0",
                "illum 2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    def _write_obj_part(lines: list[str], verts: np.ndarray, faces: np.ndarray, v_offset: int) -> int:
        for x, y, z in verts:
            lines.append(f"v {x * scale:.6f} {y * scale:.6f} {z * scale:.6f}")
        for a, b, c in faces:
            lines.append(f"f {a + 1 + v_offset} {b + 1 + v_offset} {c + 1 + v_offset}")
        return v_offset + len(verts)

    lines = [
        "# Parametric clear vial + white screw cap",
        f"# lid Ø{dims.lid_diameter_mm:.2f}×{dims.lid_height_mm:.2f} mm",
        f"# cup Ø{dims.cup_diameter_mm:.2f}×{dims.cup_height_mm:.2f} mm",
        f"# total H {dims.total_height_mm:.2f} mm",
        f"mtllib {mtl_path.name}",
        "o vial_cup",
        "usemtl cup_clear",
    ]
    offset = _write_obj_part(lines, cup_verts, cup_faces, 0)
    lines.extend(["o vial_lid", "usemtl lid_white"])
    _write_obj_part(lines, lid_verts, lid_faces, offset)
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Native USDA (mm units) — open directly in Isaac Sim / Omniverse.
    usda = f"""#usda 1.0
(
    defaultPrim = "Vial"
    metersPerUnit = 0.001
    upAxis = "Z"
    doc = "Parametric vial from vial_3d_dims.json — lid Ø{dims.lid_diameter_mm:.2f} cup Ø{dims.cup_diameter_mm:.2f} total H {dims.total_height_mm:.2f} mm"
)

def Xform "Vial" (
    kind = "component"
)
{{
    def Cylinder "Cup"
    {{
        double height = {dims.cup_height_mm:.4f}
        double radius = {cup_r:.4f}
        float3[] extent = [({-cup_r:.4f}, {-cup_r:.4f}, {-dims.cup_height_mm * 0.5:.4f}), ({cup_r:.4f}, {cup_r:.4f}, {dims.cup_height_mm * 0.5:.4f})]
        color3f[] primvars:displayColor = [(0.55, 0.72, 0.82)]
        float[] primvars:displayOpacity = [0.35]
        double3 xformOp:translate = (0, 0, {dims.cup_height_mm * 0.5:.4f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}

    def Cylinder "Lid"
    {{
        double height = {dims.lid_height_mm:.4f}
        double radius = {lid_r:.4f}
        float3[] extent = [({-lid_r:.4f}, {-lid_r:.4f}, {-dims.lid_height_mm * 0.5:.4f}), ({lid_r:.4f}, {lid_r:.4f}, {dims.lid_height_mm * 0.5:.4f})]
        color3f[] primvars:displayColor = [(0.96, 0.96, 0.97)]
        float[] primvars:displayOpacity = [1.0]
        double3 xformOp:translate = (0, 0, {dims.cup_height_mm + dims.lid_height_mm * 0.5:.4f})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
}}
"""
    usda_path.write_text(usda, encoding="utf-8")
    return obj_path, mtl_path, usda_path


def assist_box_from_peer(
    peer_box: tuple[int, int, int, int],
    peer_distance_mm: float,
    peer_K: Intrinsics | tuple[float, float, float, float],
    local_K: Intrinsics | tuple[float, float, float, float],
    local_image_size: tuple[int, int],
    *,
    peer_camera: int,
    local_camera: int,
    dims: VialDimensions | None = None,
    extrinsics: StereoExtrinsics | None = None,
) -> tuple[int, int, int, int] | None:
    """Opt-in helper: peer detection → reprojected local search box.

    Returns None when extrinsics are missing or the peer pose is invalid.
    Caller must still run clear-vial / soft-clear gates — this never bypasses them.
    """
    extrinsics = extrinsics if extrinsics is not None else load_extrinsics()
    if extrinsics is None:
        return None
    dims = dims or load_dims()
    hyp = lift_detection_to_3d(
        peer_box,
        peer_distance_mm,
        peer_K,
        dims,
        source_camera=peer_camera,
    )
    if hyp is None:
        return None
    return reproject_vial_to_camera(
        hyp,
        extrinsics,
        local_K,
        local_image_size,
        source_camera=peer_camera,
        target_camera=local_camera,
    )


def _cylinder_mesh(
    radius: float,
    z0: float,
    z1: float,
    n_theta: int = 64,
    n_z: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X,Y,Z grids for a vertical cylinder about the Z axis."""
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta)
    z = np.linspace(z0, z1, n_z)
    theta_g, z_g = np.meshgrid(theta, z)
    x = radius * np.cos(theta_g)
    y = radius * np.sin(theta_g)
    return x, y, z_g


def _disk_mesh(
    radius: float,
    z: float,
    n_theta: int = 64,
    n_r: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X,Y,Z grids for a horizontal disk at height z."""
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta)
    r = np.linspace(0.0, radius, n_r)
    theta_g, r_g = np.meshgrid(theta, r)
    x = r_g * np.cos(theta_g)
    y = r_g * np.sin(theta_g)
    z_g = np.full_like(x, z)
    return x, y, z_g


def _draw_vial_3d(ax, dims: VialDimensions, *, elev: float = 18.0, azim: float = -55.0) -> None:
    """Draw stacked lid (opaque white) + cup (translucent) cylinders on ax."""
    lid_r = dims.lid_diameter_mm * 0.5
    cup_r = dims.cup_diameter_mm * 0.5
    cup_h = dims.cup_height_mm
    lid_h = dims.lid_height_mm
    total_h = dims.total_height_mm

    # Cup: clear / translucent cyan-gray walls + faint top rim
    x, y, z = _cylinder_mesh(cup_r, 0.0, cup_h)
    ax.plot_surface(x, y, z, color=(0.55, 0.72, 0.82), alpha=0.28, linewidth=0, shade=True)
    x, y, z = _disk_mesh(cup_r, 0.0)
    ax.plot_surface(x, y, z, color=(0.45, 0.62, 0.72), alpha=0.35, linewidth=0, shade=True)
    # Cup rim ring (top of cup, under lid)
    theta = np.linspace(0.0, 2.0 * math.pi, 80)
    ax.plot(cup_r * np.cos(theta), cup_r * np.sin(theta), np.full_like(theta, cup_h),
            color=(0.35, 0.55, 0.65), lw=1.2, alpha=0.7)

    # Lid: opaque white cylinder stacked on cup
    x, y, z = _cylinder_mesh(lid_r, cup_h, total_h)
    ax.plot_surface(x, y, z, color=(0.96, 0.96, 0.97), alpha=0.95, linewidth=0, shade=True)
    x, y, z = _disk_mesh(lid_r, total_h)
    ax.plot_surface(x, y, z, color=(1.0, 1.0, 1.0), alpha=0.98, linewidth=0, shade=True)
    x, y, z = _disk_mesh(lid_r, cup_h)
    ax.plot_surface(x, y, z, color=(0.88, 0.88, 0.90), alpha=0.9, linewidth=0, shade=True)

    # Outline edges for readability
    for r, z0, z1, color, lw in (
        (cup_r, 0.0, cup_h, (0.25, 0.45, 0.55), 1.0),
        (lid_r, cup_h, total_h, (0.45, 0.45, 0.48), 1.2),
    ):
        for ang in (0.0, math.pi):
            ax.plot(
                [r * math.cos(ang), r * math.cos(ang)],
                [r * math.sin(ang), r * math.sin(ang)],
                [z0, z1],
                color=color,
                lw=lw,
                alpha=0.85,
            )
        ax.plot(r * np.cos(theta), r * np.sin(theta), np.full_like(theta, z0),
                color=color, lw=lw, alpha=0.85)
        ax.plot(r * np.cos(theta), r * np.sin(theta), np.full_like(theta, z1),
                color=color, lw=lw, alpha=0.85)

    # Dimension annotations (mm)
    max_r = max(lid_r, cup_r)
    pad = max_r * 0.18
    # Lid diameter arrow (top)
    y_dim = max_r + pad * 1.6
    ax.plot([-lid_r, lid_r], [y_dim, y_dim], [total_h + pad * 0.3, total_h + pad * 0.3],
            color="#333333", lw=1.0)
    ax.plot([-lid_r, -lid_r], [y_dim - pad * 0.3, y_dim + pad * 0.3],
            [total_h + pad * 0.3, total_h + pad * 0.3], color="#333333", lw=1.0)
    ax.plot([lid_r, lid_r], [y_dim - pad * 0.3, y_dim + pad * 0.3],
            [total_h + pad * 0.3, total_h + pad * 0.3], color="#333333", lw=1.0)
    ax.text(0, y_dim + pad * 0.5, total_h + pad * 0.8,
            f"lid Ø {dims.lid_diameter_mm:.2f} mm",
            ha="center", va="bottom", fontsize=9, color="#222222")

    # Cup diameter
    y_dim2 = -(max_r + pad * 1.4)
    ax.plot([-cup_r, cup_r], [y_dim2, y_dim2], [cup_h * 0.45, cup_h * 0.45],
            color="#1a5f7a", lw=1.0)
    ax.text(0, y_dim2 - pad * 0.4, cup_h * 0.45,
            f"cup Ø {dims.cup_diameter_mm:.2f} mm",
            ha="center", va="top", fontsize=9, color="#1a5f7a")

    # Height stack (right side)
    x_h = max_r + pad * 2.2
    ax.plot([x_h, x_h], [0, 0], [0, total_h], color="#333333", lw=1.0)
    ax.plot([x_h - pad * 0.25, x_h + pad * 0.25], [0, 0], [0, 0], color="#333333", lw=1.0)
    ax.plot([x_h - pad * 0.25, x_h + pad * 0.25], [0, 0], [total_h, total_h],
            color="#333333", lw=1.0)
    # Lid / cup split tick
    ax.plot([x_h - pad * 0.2, x_h + pad * 0.2], [0, 0], [cup_h, cup_h],
            color="#666666", lw=0.9)
    ax.text(x_h + pad * 0.5, 0, cup_h * 0.5,
            f"cup H {cup_h:.2f}",
            ha="left", va="center", fontsize=8, color="#1a5f7a")
    ax.text(x_h + pad * 0.5, 0, cup_h + lid_h * 0.5,
            f"lid H {lid_h:.2f}",
            ha="left", va="center", fontsize=8, color="#444444")
    ax.text(x_h + pad * 0.5, 0, total_h + pad * 0.4,
            f"total H {total_h:.2f} mm",
            ha="left", va="bottom", fontsize=9, color="#222222", fontweight="bold")

    lim = max_r * 1.85
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, total_h * 1.2)
    ax.set_box_aspect((1, 1, total_h / (2 * lim) * 1.15))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("Y (mm)", fontsize=8)
    ax.set_zlabel("Z (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.25)


def _draw_vial_side(ax, dims: VialDimensions) -> None:
    """Orthographic side silhouette (XZ) with dimension callouts."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    lid_r = dims.lid_diameter_mm * 0.5
    cup_r = dims.cup_diameter_mm * 0.5
    cup_h = dims.cup_height_mm
    lid_h = dims.lid_height_mm
    total_h = dims.total_height_mm

    # Cup body
    ax.add_patch(
        Rectangle(
            (-cup_r, 0),
            2 * cup_r,
            cup_h,
            facecolor=(0.55, 0.72, 0.82, 0.35),
            edgecolor=(0.25, 0.45, 0.55),
            linewidth=1.5,
        )
    )
    # Lid
    ax.add_patch(
        FancyBboxPatch(
            (-lid_r, cup_h),
            2 * lid_r,
            lid_h,
            boxstyle="round,pad=0,rounding_size=0.8",
            facecolor=(0.96, 0.96, 0.97),
            edgecolor=(0.4, 0.4, 0.42),
            linewidth=1.6,
        )
    )

    # Dimension lines
    # Widths
    ax.annotate(
        "",
        xy=(-lid_r, total_h + 4),
        xytext=(lid_r, total_h + 4),
        arrowprops=dict(arrowstyle="<->", color="#333", lw=1.1),
    )
    ax.text(0, total_h + 5.5, f"lid Ø {dims.lid_diameter_mm:.2f} mm",
            ha="center", va="bottom", fontsize=10, color="#222")
    ax.annotate(
        "",
        xy=(-cup_r, -5),
        xytext=(cup_r, -5),
        arrowprops=dict(arrowstyle="<->", color="#1a5f7a", lw=1.1),
    )
    ax.text(0, -7, f"cup Ø {dims.cup_diameter_mm:.2f} mm",
            ha="center", va="top", fontsize=10, color="#1a5f7a")

    # Heights (right)
    x_h = max(lid_r, cup_r) + 6
    ax.annotate(
        "",
        xy=(x_h, 0),
        xytext=(x_h, cup_h),
        arrowprops=dict(arrowstyle="<->", color="#1a5f7a", lw=1.0),
    )
    ax.text(x_h + 1.5, cup_h * 0.5, f"{cup_h:.2f}", ha="left", va="center",
            fontsize=9, color="#1a5f7a")
    ax.annotate(
        "",
        xy=(x_h, cup_h),
        xytext=(x_h, total_h),
        arrowprops=dict(arrowstyle="<->", color="#555", lw=1.0),
    )
    ax.text(x_h + 1.5, cup_h + lid_h * 0.5, f"{lid_h:.2f}", ha="left", va="center",
            fontsize=9, color="#444")
    ax.annotate(
        "",
        xy=(x_h + 8, 0),
        xytext=(x_h + 8, total_h),
        arrowprops=dict(arrowstyle="<->", color="#222", lw=1.2),
    )
    ax.text(x_h + 9.5, total_h * 0.5, f"total H {total_h:.2f} mm",
            ha="left", va="center", fontsize=10, color="#222", rotation=90)

    max_r = max(lid_r, cup_r)
    ax.set_xlim(-max_r - 8, x_h + 22)
    ax.set_ylim(-12, total_h + 14)
    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)", fontsize=9)
    ax.set_ylabel("Z (mm)", fontsize=9)
    ax.set_title("Side (orthographic)", fontsize=11, pad=8)
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="#aaa", lw=0.6)


def render_vial_preview(
    dims: VialDimensions | None = None,
    *,
    out_path: Path | None = None,
    side_path: Path | None = None,
) -> tuple[Path, Path]:
    """Render perspective + side views; write PNG(s). Returns (combined, side) paths."""
    import matplotlib.pyplot as plt

    dims = dims or load_dims()
    out_path = out_path or PREVIEW_PATH
    side_path = side_path or PREVIEW_SIDE_PATH

    # Combined figure: perspective (left) + side ortho (right) ~1200px wide
    fig = plt.figure(figsize=(12.0, 7.2), dpi=120, facecolor="white")
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    _draw_vial_3d(ax3d, dims, elev=18.0, azim=-55.0)
    ax3d.set_title("Perspective", fontsize=11, pad=6)

    ax_side = fig.add_subplot(1, 2, 2)
    _draw_vial_side(ax_side, dims)

    fig.suptitle(
        "Parametric vial — white lid on clear cup (stacked cylinders)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        (
            f"lid Ø{dims.lid_diameter_mm:.2f}×{dims.lid_height_mm:.2f} mm  ·  "
            f"cup Ø{dims.cup_diameter_mm:.2f}×{dims.cup_height_mm:.2f} mm  ·  "
            f"total H {dims.total_height_mm:.2f} mm"
            + (f"  ·  n={dims.sample_count}" if dims.sample_count else "")
        ),
        ha="center",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Optional dedicated side PNG
    fig2, ax2 = plt.subplots(figsize=(6.0, 8.0), dpi=120, facecolor="white")
    _draw_vial_side(ax2, dims)
    fig2.suptitle(
        f"Vial side view — total H {dims.total_height_mm:.2f} mm",
        fontsize=12,
        fontweight="bold",
    )
    fig2.tight_layout(rect=(0, 0, 1, 0.96))
    fig2.savefig(side_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    return out_path, side_path


def cmd_preview(_: argparse.Namespace) -> int:
    dims = load_dims()
    combined, side = render_vial_preview(dims)
    print(f"Wrote {combined}")
    print(f"Wrote {side}")
    print(
        f"  stacked cylinders: lid Ø{dims.lid_diameter_mm:.2f}×{dims.lid_height_mm:.2f} mm "
        f"on cup Ø{dims.cup_diameter_mm:.2f}×{dims.cup_height_mm:.2f} mm "
        f"(total H {dims.total_height_mm:.2f} mm)"
    )
    return 0


def cmd_view(_: argparse.Namespace) -> int:
    """Interactive matplotlib 3D window — drag to rotate, close window to exit."""
    import matplotlib.pyplot as plt

    dims = load_dims()
    fig = plt.figure(figsize=(9.0, 7.5), dpi=110, facecolor="white")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _draw_vial_3d(ax, dims, elev=18.0, azim=-55.0)
    ax.set_title(
        f"Interactive vial  ·  Ø{dims.lid_diameter_mm:.2f} × H{dims.total_height_mm:.2f} mm"
        + (f"  ·  n={dims.sample_count}" if dims.sample_count else ""),
        fontsize=12,
        pad=10,
    )
    fig.text(
        0.5,
        0.02,
        "Drag to rotate  ·  scroll to zoom  ·  used live as size prior + 3D box proposals",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    print(
        f"Showing interactive 3D vial (lid Ø{dims.lid_diameter_mm:.2f} × "
        f"H{dims.total_height_mm:.2f} mm). Close the window when done."
    )
    plt.show()
    return 0


def cmd_export_mesh(_: argparse.Namespace) -> int:
    dims = load_dims()
    obj_path, mtl_path, usda_path = export_vial_mesh(dims)
    print(f"Wrote {obj_path}")
    print(f"Wrote {mtl_path}")
    print(f"Wrote {usda_path}")
    print(
        f"Isaac Sim: File -> Import -> "
        f"{usda_path.name}  (or convert {obj_path.name} with MeshConverter)."
    )
    print(
        "Live CV already uses these dims: "
        "python depthcamera.py --dual-independent"
    )
    return 0


def cmd_fit(_: argparse.Namespace) -> int:
    dims = fit_dimensions_from_stills()
    print(f"Wrote {DIMS_PATH}")
    print(
        f"  lid diam {dims.lid_diameter_mm} mm x {dims.lid_height_mm} mm | "
        f"cup diam {dims.cup_diameter_mm} mm x {dims.cup_height_mm} mm | "
        f"total H {dims.total_height_mm} mm"
    )
    print(f"  samples={dims.sample_count}  {dims.notes}")
    print(f"Extrinsics template: {EXTRINSICS_PATH} (replace after ChArUco)")
    return 0


def cmd_show(_: argparse.Namespace) -> int:
    dims = load_dims()
    print(json.dumps(asdict(dims) | {"total_height_mm": dims.total_height_mm}, indent=2))
    if not EXTRINSICS_PATH.exists():
        print("extrinsics: missing (run fit or calibrate)")
        return 0
    state = extrinsics_reproject_state()
    print(
        "extrinsics:",
        EXTRINSICS_PATH.name,
        "| PLACEHOLDER" if state != "ready" else "| looks set",
    )
    if state != "ready":
        print(
            "  WARNING: identity R / placeholder t - dual reproject will mis-aim. "
            "See: python stereo_calib_stub.py howto"
        )
    return 0


def cmd_reproject_demo(args: argparse.Namespace) -> int:
    dims = load_dims()
    write_extrinsics_template()
    ext = load_extrinsics()
    if ext is None:
        print("No extrinsics; wrote template only.")
        return 1
    parts = [int(v) for v in args.box.split(",")]
    if len(parts) != 4:
        raise SystemExit("--box must be x,y,w,h")
    box = (parts[0], parts[1], parts[2], parts[3])
    K0 = ext.K0 or default_intrinsics()
    K1 = ext.K1 or default_intrinsics()
    hyp = lift_detection_to_3d(box, args.distance_mm, K0, dims, source_camera=0)
    if hyp is None:
        print("Could not lift detection")
        return 1
    size = (args.width, args.height)
    proj = reproject_vial_to_camera(
        hyp, ext, K1, size, source_camera=0, target_camera=1
    )
    print("hypothesis cap_xyz_mm:", hyp.cap_xyz_mm)
    print("reprojected cam1 box:", proj)
    print("3D size score (source):", round(score_box_vs_model(box, args.distance_mm, K0, dims), 3))
    print(
        "NOTE: placeholder extrinsics are not accurate - calibrate before live assist."
    )
    return 0 if proj is not None else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parametric 3D vial model tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit", help="Fit lid/cup mm sizes from stills + distance sidecars")
    p_fit.set_defaults(func=cmd_fit)

    p_show = sub.add_parser("show", help="Print current vial_3d_dims.json")
    p_show.set_defaults(func=cmd_show)

    p_preview = sub.add_parser(
        "preview",
        help="Render 3D + side PNG of parametric vial (vial_3d_preview.png)",
    )
    p_preview.set_defaults(func=cmd_preview)

    p_view = sub.add_parser(
        "view",
        help="Interactive 3D window (drag to rotate) — same dims used by live CV",
    )
    p_view.set_defaults(func=cmd_view)

    p_export = sub.add_parser(
        "export-mesh",
        help="Write OBJ+MTL+USDA under assets/vial/ for Isaac Sim / Omniverse",
    )
    p_export.set_defaults(func=cmd_export_mesh)

    p_demo = sub.add_parser(
        "reproject-demo",
        help="Lift a cam0 box and reproject to cam1 (needs stereo_extrinsics.json)",
    )
    p_demo.add_argument("--box", default="100,120,48,70", help="x,y,w,h in cam0 pixels")
    p_demo.add_argument("--distance-mm", type=float, default=400.0)
    p_demo.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH)
    p_demo.add_argument("--height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    p_demo.set_defaults(func=cmd_reproject_demo)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
