"""Load / resolve vial template assets (grayscale + edge PNGs + meta.json)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

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


def default_camera_root() -> Path:
    # .../Camera/ros2_ws/src/agent_03_template_orb/agent_03_template_orb/template_bank.py
    here = Path(__file__).resolve()
    # Prefer climbing to a folder that contains vial_dataset/.
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
    if pkg.exists() and any(pkg.glob("*.png")):
        return pkg
    # Fallback: share install path via ament (optional).
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("agent_03_template_orb")) / "data" / "templates"
        if share.exists() and any(share.glob("*.png")):
            return share
    except Exception:  # noqa: BLE001
        pass
    _ = camera_root  # reserved for future Camera-level caches
    return pkg


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
        templates.append(
            Template(
                name=name,
                gray=gray,
                edge=edge,
                source_image=str(info.get("source_image", "")),
                area_frac=float(info.get("area_frac", 0.0)),
                aspect=float(info.get("aspect", gray.shape[0] / max(gray.shape[1], 1))),
            )
        )
    return templates


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


def extract_templates_from_split(
    split_dir: Path,
    out_dir: Path,
    *,
    split: str = "train",
    max_templates: int = 14,
    canon_height: int = 80,
    area_min: float = 0.008,
    area_max: float = 0.14,
    aspect_min: float = 0.95,
    aspect_max: float = 1.60,
) -> List[Template]:
    """Crop diverse upright vials from YOLO GT and write gray/edge PNGs + meta."""
    name = "val" if split in ("val", "validate") else split
    img_dir = split_dir / "images" / name
    lbl_dir = split_dir / "labels" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*"):
        if old.is_file():
            old.unlink()

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
        # Prefer mid-area, near-canonical aspect.
        score = -abs(np.log(area / 0.03)) - 0.5 * abs(aspect - 1.25)
        candidates.append((score, img_path, crop, area, aspect))

    candidates.sort(key=lambda t: t[0], reverse=True)
    # Diversify by source stem buckets.
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

    meta = {"canon_height": canon_height, "split": split, "templates": []}
    templates: List[Template] = []
    for i, (_, img_path, crop, area, aspect) in enumerate(selected):
        name_i = f"tmpl_{i:02d}_{img_path.stem}"
        scale = canon_height / float(crop.shape[0])
        tw = max(12, int(round(crop.shape[1] * scale)))
        gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (tw, canon_height))
        edge = cv2.Canny(gray, 40, 120)
        cv2.imwrite(str(out_dir / f"{name_i}_gray.png"), gray)
        cv2.imwrite(str(out_dir / f"{name_i}_edge.png"), edge)
        entry = {
            "name": name_i,
            "source_image": img_path.name,
            "area_frac": float(area),
            "aspect": float(aspect),
            "size_wh": [int(tw), int(canon_height)],
        }
        meta["templates"].append(entry)
        templates.append(
            Template(
                name=name_i,
                gray=gray,
                edge=edge,
                source_image=img_path.name,
                area_frac=float(area),
                aspect=float(aspect),
            )
        )

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return templates


def ensure_templates(
    templates_dir: Optional[Union[str, Path]] = None,
    camera_root: Optional[Union[str, Path]] = None,
    *,
    max_templates: int = 14,
) -> Sequence[Template]:
    """Load templates; if missing, extract from Camera split train set."""
    root = Path(camera_root) if camera_root else default_camera_root()
    tdir = resolve_templates_dir(templates_dir, root)
    loaded = load_templates(tdir)
    if loaded:
        return loaded
    split = root / "vial_dataset" / "split_70_10_20"
    out = default_templates_dir()
    out.mkdir(parents=True, exist_ok=True)
    if split.exists():
        return extract_templates_from_split(split, out, max_templates=max_templates)
    # Last resort: raw folder positives.
    raw = root / "vial_dataset" / "raw"
    if not raw.exists():
        return []
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*"):
        if old.is_file():
            old.unlink()
    meta = {"canon_height": 80, "split": "raw", "templates": []}
    templates: List[Template] = []
    for img_path in sorted(raw.glob("*.jpg")):
        lbl = img_path.with_suffix(".txt")
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
        if not (0.008 <= area <= 0.14 and 0.95 <= aspect <= 1.60):
            continue
        crop = bgr[y1:y2, x1:x2]
        scale = 80 / float(crop.shape[0])
        tw = max(12, int(round(crop.shape[1] * scale)))
        gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (tw, 80))
        edge = cv2.Canny(gray, 40, 120)
        name_i = f"tmpl_{len(templates):02d}_{img_path.stem}"
        cv2.imwrite(str(out / f"{name_i}_gray.png"), gray)
        cv2.imwrite(str(out / f"{name_i}_edge.png"), edge)
        meta["templates"].append(
            {
                "name": name_i,
                "source_image": img_path.name,
                "area_frac": float(area),
                "aspect": float(aspect),
                "size_wh": [int(tw), 80],
            }
        )
        templates.append(
            Template(name_i, gray, edge, img_path.name, float(area), float(aspect))
        )
        if len(templates) >= max_templates:
            break
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return templates
