"""Load / extract upright + sideways vial templates (gray + edge PNGs)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


@dataclass
class Template:
    name: str
    gray: np.ndarray
    edge: np.ndarray
    source_image: str = ""
    area_frac: float = 0.0
    aspect: float = 1.0
    pose: str = "upright"  # upright | sideways
    cap_side: str = "top"  # top | bottom | left | right


def default_camera_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "vial_dataset").is_dir():
            return parent
    for parent in here.parents:
        if parent.name == "ros2_ws":
            sibling = parent.parent / "robotics"
            if sibling.is_dir():
                return sibling
            break
    return here.parents[4]


def default_templates_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "templates"


def resolve_templates_dir(
    explicit: Optional[Union[str, Path]] = None,
    camera_root: Optional[Union[str, Path]] = None,
) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    pkg = default_templates_dir()
    if pkg.exists() and any(pkg.glob("*_gray.png")):
        return pkg
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("r2_02_rotated_templates")) / "data" / "templates"
        if share.exists() and any(share.glob("*_gray.png")):
            return share
    except Exception:  # noqa: BLE001
        pass
    _ = camera_root
    return pkg


def rotate_bound_gray(img: np.ndarray, angle: float, *, edge: bool = False) -> np.ndarray:
    """Rotate image, expanding canvas so content is not clipped."""
    h, w = img.shape[:2]
    if abs(angle) < 1e-6 or abs(abs(angle) - 360.0) < 1e-6:
        return img.copy()
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = int(h * sin + w * cos)
    nH = int(h * cos + w * sin)
    M[0, 2] += (nW / 2.0) - center[0]
    M[1, 2] += (nH / 2.0) - center[1]
    border = cv2.BORDER_CONSTANT if edge else cv2.BORDER_REPLICATE
    return cv2.warpAffine(
        img,
        M,
        (max(1, nW), max(1, nH)),
        flags=cv2.INTER_LINEAR,
        borderMode=border,
        borderValue=0,
    )


def yolo_box_from_label(label_path: Path, width: int, height: int):
    if not label_path.exists():
        return None
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    line = text.splitlines()[0]
    parts = line.split()
    if len(parts) != 5:
        return None
    _, cx, cy, bw, bh = [float(v) for v in parts]
    x1 = int((cx - bw / 2.0) * width)
    y1 = int((cy - bh / 2.0) * height)
    x2 = int((cx + bw / 2.0) * width)
    y2 = int((cy + bh / 2.0) * height)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _infer_cap_side(crop_bgr: np.ndarray, pose: str) -> str:
    """Estimate which end of the crop holds the white cap."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 2] >= 165) & (hsv[:, :, 1] <= 70)).astype(np.float32)
    h, w = white.shape
    if pose == "sideways" or w >= h:
        left = float(white[:, : max(1, w // 3)].mean())
        right = float(white[:, 2 * w // 3 :].mean())
        return "left" if left >= right else "right"
    top = float(white[: max(1, h // 3), :].mean())
    bot = float(white[2 * h // 3 :, :].mean())
    return "top" if top >= bot else "bottom"


def _write_template(
    out_dir: Path,
    name: str,
    crop_bgr: np.ndarray,
    *,
    canon_long: int,
    source_image: str,
    area: float,
    aspect: float,
    pose: str,
    cap_side: str = "",
) -> Template:
    gray0 = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h0, w0 = gray0.shape[:2]
    # Normalize so the longer side is canon_long (works for upright and sideways).
    long0 = max(h0, w0)
    scale = canon_long / float(max(long0, 1))
    tw = max(12, int(round(w0 * scale)))
    th = max(12, int(round(h0 * scale)))
    gray = cv2.resize(gray0, (tw, th), interpolation=cv2.INTER_AREA)
    edge = cv2.Canny(gray, 40, 120)
    cv2.imwrite(str(out_dir / f"{name}_gray.png"), gray)
    cv2.imwrite(str(out_dir / f"{name}_edge.png"), edge)
    side = cap_side or _infer_cap_side(crop_bgr, pose)
    return Template(
        name=name,
        gray=gray,
        edge=edge,
        source_image=source_image,
        area_frac=float(area),
        aspect=float(aspect),
        pose=pose,
        cap_side=side,
    )


def _extract_upright_from_split(
    split_dir: Path,
    *,
    split: str,
    max_templates: int,
    canon_long: int,
    area_min: float,
    area_max: float,
    aspect_min: float,
    aspect_max: float,
) -> List[Tuple[str, np.ndarray, float, float, str, str]]:
    name = "val" if split in ("val", "validate") else split
    img_dir = split_dir / "images" / name
    lbl_dir = split_dir / "labels" / name
    candidates = []
    for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
        lbl = lbl_dir / f"{img_path.stem}.txt"
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        box = yolo_box_from_label(lbl, w, h)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        area = (bw * bh) / float(max(w * h, 1))
        aspect = bh / float(max(bw, 1))
        if not (area_min <= area <= area_max and aspect_min <= aspect <= aspect_max):
            continue
        crop = bgr[y1:y2, x1:x2]
        if crop.shape[0] < 24 or crop.shape[1] < 16:
            continue
        score = -abs(np.log(area / 0.03)) - 0.5 * abs(aspect - 1.25)
        candidates.append((score, img_path, crop, area, aspect))

    candidates.sort(key=lambda t: t[0], reverse=True)
    selected = []
    used_stems = set()
    for item in candidates:
        stem = item[1].stem.split("_")[0]
        if stem in used_stems and len(selected) < max_templates // 2:
            continue
        used_stems.add(stem)
        selected.append(item)
        if len(selected) >= max_templates:
            break
    if len(selected) < max_templates:
        for item in candidates:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= max_templates:
                break

    out: List[Tuple[str, np.ndarray, float, float, str, str]] = []
    for _, img_path, crop, area, aspect in selected:
        out.append((img_path.name, crop, float(area), float(aspect), "upright", "top"))
    return out


def _extract_sideways_from_synth(
    synth_dir: Path,
    *,
    max_templates: int,
) -> List[Tuple[str, np.ndarray, float, float, str, str]]:
    """Crop sideways GT boxes from sideways_race/synth JSON+PNG.

    Uses only easy `sideways_left/right_*` with even indices (00,02,…) so
    odd indices + hard_sideways remain held-out for evaluate.py.
    """
    if not synth_dir.exists():
        return []
    left: List[Tuple[float, Path, np.ndarray, float, float]] = []
    right: List[Tuple[float, Path, np.ndarray, float, float]] = []
    for meta_path in sorted(synth_dir.glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        kind = str(meta.get("kind", ""))
        # Hold out hard_sideways* entirely from the template bank.
        if kind not in ("sideways_left", "sideways_right"):
            continue
        stem = meta_path.stem  # e.g. sideways_left_00
        try:
            idx = int(stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if idx % 2 == 1:
            continue  # odd indices held out
        color_name = meta.get("color")
        if not color_name:
            continue
        img_path = synth_dir / color_name
        if not img_path.exists():
            continue
        boxes = meta.get("boxes") or []
        if not boxes:
            continue
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        x, y, bw, bh = [int(v) for v in boxes[0][:4]]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + bw), min(h, y + bh)
        if x2 - x1 < 16 or y2 - y1 < 12:
            continue
        crop = bgr[y1:y2, x1:x2]
        area = ((x2 - x1) * (y2 - y1)) / float(max(w * h, 1))
        aspect = (y2 - y1) / float(max(x2 - x1, 1))
        score = -abs(np.log(area / 0.02)) - 0.4 * abs(aspect - 0.55)
        bucket = left if kind.endswith("left") else right
        bucket.append((score, img_path, crop, area, aspect))

    left.sort(key=lambda t: t[0], reverse=True)
    right.sort(key=lambda t: t[0], reverse=True)
    per = max(1, max_templates // 2)
    selected = left[:per] + right[:per]
    selected = selected[:max_templates]
    out: List[Tuple[str, np.ndarray, float, float, str, str]] = []
    for _, img_path, crop, area, aspect in selected:
        cap = "left" if "left" in img_path.stem else "right"
        out.append((img_path.name, crop, float(area), float(aspect), "sideways", cap))
    return out


def extract_templates(
    out_dir: Path,
    *,
    camera_root: Optional[Path] = None,
    split: str = "train",
    max_upright: int = 8,
    max_sideways: int = 8,
    canon_long: int = 80,
) -> List[Template]:
    """Build template bank from upright train GT + sideways synth GT."""
    root = Path(camera_root) if camera_root else default_camera_root()
    split_dir = root / "vial_dataset" / "split_70_10_20"
    synth_dir = root / "sideways_race" / "synth"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*"):
        if old.is_file():
            old.unlink()

    upright_items: List[Tuple[str, np.ndarray, float, float, str, str]] = []
    if split_dir.exists():
        upright_items = _extract_upright_from_split(
            split_dir,
            split=split,
            max_templates=max_upright,
            canon_long=canon_long,
            area_min=0.008,
            area_max=0.14,
            aspect_min=0.95,
            aspect_max=1.70,
        )
    sideways_items = _extract_sideways_from_synth(
        synth_dir, max_templates=max(2, max_sideways // 2)
    )

    # Also rotate real upright crops → sideways photo templates (helps real fallen
    # vials; synth-only banks overfit painted rectangles / desk clutter).
    # ROTATE_90_CLOCKWISE maps upright top-cap → right-cap.
    n_rot = max(2, max_sideways - len(sideways_items))
    for src_name, crop, area, aspect, _pose, _cap in upright_items[:n_rot]:
        rot = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        sideways_items.append(
            (
                f"rot90_{src_name}",
                rot,
                float(area),
                float(1.0 / max(aspect, 1e-6)),
                "sideways",
                "right",
            )
        )

    meta = {
        "canon_long": canon_long,
        "split": split,
        "n_upright": len(upright_items),
        "n_sideways": len(sideways_items),
        "templates": [],
    }
    templates: List[Template] = []
    idx = 0
    for src_name, crop, area, aspect, pose, cap_side in upright_items + sideways_items:
        name_i = f"tmpl_{idx:02d}_{pose}_{Path(src_name).stem}"
        t = _write_template(
            out_dir,
            name_i,
            crop,
            canon_long=canon_long,
            source_image=src_name,
            area=area,
            aspect=aspect,
            pose=pose,
            cap_side=cap_side,
        )
        meta["templates"].append(
            {
                "name": t.name,
                "source_image": t.source_image,
                "area_frac": t.area_frac,
                "aspect": t.aspect,
                "pose": pose,
                "cap_side": t.cap_side,
                "size_wh": [int(t.gray.shape[1]), int(t.gray.shape[0])],
            }
        )
        templates.append(t)
        idx += 1

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return templates


def load_templates(templates_dir: Union[str, Path]) -> List[Template]:
    root = Path(templates_dir)
    if not root.exists():
        return []
    meta_path = root / "meta.json"
    meta_entries = []
    if meta_path.exists():
        try:
            meta_entries = json.loads(meta_path.read_text(encoding="utf-8")).get("templates", [])
        except Exception:  # noqa: BLE001
            meta_entries = []
    by_name = {e.get("name"): e for e in meta_entries if isinstance(e, dict)}

    templates: List[Template] = []
    for png in sorted(root.glob("*_gray.png")):
        name = png.name.replace("_gray.png", "")
        edge_path = root / f"{name}_edge.png"
        gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        if edge_path.exists():
            edge = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
        else:
            edge = cv2.Canny(gray, 40, 120)
        if edge is None:
            continue
        info = by_name.get(name, {})
        pose = str(info.get("pose", "upright"))
        if "sideways" in name:
            pose = "sideways"
        cap_side = str(info.get("cap_side", ""))
        if not cap_side:
            if "left" in name:
                cap_side = "left"
            elif "right" in name or "rot90" in name:
                cap_side = "right"
            elif pose == "upright":
                cap_side = "top"
            else:
                cap_side = "left"
        templates.append(
            Template(
                name=name,
                gray=gray,
                edge=edge,
                source_image=str(info.get("source_image", "")),
                area_frac=float(info.get("area_frac", 0.0)),
                aspect=float(info.get("aspect", gray.shape[0] / max(gray.shape[1], 1))),
                pose=pose,
                cap_side=cap_side,
            )
        )
    return templates


def ensure_templates(
    templates_dir: Optional[Union[str, Path]] = None,
    camera_root: Optional[Union[str, Path]] = None,
    *,
    max_upright: int = 8,
    max_sideways: int = 8,
) -> Sequence[Template]:
    root = Path(camera_root) if camera_root else default_camera_root()
    tdir = resolve_templates_dir(templates_dir, root)
    loaded = load_templates(tdir)
    if loaded:
        # Require at least one sideways template; otherwise rebuild.
        if any(t.pose == "sideways" or "sideways" in t.name for t in loaded):
            return loaded
    out = default_templates_dir()
    return extract_templates(
        out,
        camera_root=root,
        max_upright=max_upright,
        max_sideways=max_sideways,
    )
