"""Collect photos of vials, draw boxes on them, and train a YOLO detector.

One photo copied with brightness and rotation changes is not a training set.
The box stays put while the vial moves, so the model learns the wrong place.
Take many real photos from the camera you will use later, label each vial,
then train.

Every positive box must cover the FULL vial: top of the white lid through the
bottom of the clear cup (roughly h/w 1.0–1.65). Lid-only boxes are rejected.

Promote discipline (do not skip):
  1. ``python train_vial_model.py train`` writes ``runs/vial-ft/weights/best.pt``
     and does **not** overwrite live weights by default.
  2. Holdout eval: ``python eval_asymmetric_split.py --weights runs\\vial-ft\\weights\\best.pt``
  3. Promote only on a **clear** beat of live (test@0.25 → recall → empty FP).
     Then: ``python train_vial_model.py train --export-live`` (or copy best.pt).
  Live ``find_weights`` order: vial-live → r1-winner → bon-winner (never newest scratch).

    cd E:\\Desktop\\Jessica\\Camera
    python train_vial_model.py cameras
    python train_vial_model.py capture --camera 0
    python train_vial_model.py label
    python train_vial_model.py split
    python train_vial_model.py train
    python eval_asymmetric_split.py --weights runs\\vial-ft\\weights\\best.pt
    python train_vial_model.py train --export-live   # only after clear holdout beat
    python depthcamera.py --dual-independent

Photos you already have:

    python train_vial_model.py import D:\\Photos\\vials
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "vial_dataset"
RAW = DATASET / "raw"
SPLIT_DIR = DATASET / "split_70_10_20"
LIVE_WEIGHTS = ROOT / "runs" / "vial-live" / "weights" / "best.pt"
R1_WINNER_WEIGHTS = ROOT / "runs" / "train-race" / "r1-winner" / "weights" / "best.pt"
BON_WINNER_WEIGHTS = ROOT / "runs" / "bon-winner" / "weights" / "best.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAME = "vial"
MIN_BOX_PX = 4
# Full vial = lid top through clear-cup bottom. Lid-only boxes are flatter than this.
FULL_VIAL_ASPECT_MIN = 1.0
FULL_VIAL_ASPECT_MAX = 1.65
SHORT_LABEL_DIR = DATASET / "raw_short_aspect"
# Tiny Orbbec sets underfit from scratch; fine-tune live / Attempt E when present.
DEFAULT_EPOCHS = 80
DEFAULT_IMGSZ = 640
DEFAULT_PATIENCE = 25
# Freeze backbone early layers (R2-style); 0 = train all.
DEFAULT_FREEZE = 5
DEFAULT_SPLIT_SEED = 42


def next_image_index() -> int:
    nums = []
    for path in RAW.glob("vial_*.jpg"):
        parts = path.stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            nums.append(int(parts[1]))
    return max(nums, default=0) + 1


def image_label_pairs() -> list[tuple[Path, Path]]:
    if not RAW.exists():
        return []
    pairs = []
    for image_path in sorted(RAW.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        pairs.append((image_path, image_path.with_suffix(".txt")))
    return pairs


def count_boxes(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    count = 0
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            count += 1
    return count


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def draw_banner(frame, lines: list[str]) -> None:
    y = 8
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (8, y), (16 + tw, y + th + 10), (0, 0, 0), -1)
        cv2.putText(
            frame,
            line,
            (12, y + th + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += th + 14


def list_cameras(max_index: int = 6) -> None:
    print("Camera indexes OpenCV can open:")
    found = False
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        ok = cap.isOpened()
        if ok:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  {index}: available ({width}x{height})")
            found = True
        cap.release()
    if not found:
        print("  none")
        print("Plug in the camera, close other apps using it, and run this again.")
    else:
        print("Use the index that shows the view you will run detection on.")
        print("Example: python train_vial_model.py capture --camera 0")


def open_orbbec(index: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index, cv2.CAP_OBSENSOR)
    if cap.isOpened():
        return cap
    cap.release()
    return None


def read_orbbec(cap: cv2.VideoCapture):
    if not cap.grab():
        return None
    ok, frame = cap.retrieve(None, cv2.CAP_OBSENSOR_BGR_IMAGE)
    if not ok or frame is None:
        return None
    return frame


def capture(camera_index: int) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    cap = open_orbbec(camera_index)
    orbbec = cap is not None
    if cap is None:
        cap = open_camera(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera {camera_index}.")
        print("Close depthcamera.py if it is using the Orbbec, then try again.")
        sys.exit(1)
    if orbbec:
        print("Using the Orbbec camera, the same view as depthcamera.py.")

    index = next_image_index()
    window = "Capture vials"
    print("Space or C: save a photo. Q: quit.")
    print(f"Saving to {RAW}")

    while True:
        frame = read_orbbec(cap) if orbbec else None
        if orbbec and frame is None:
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
            continue
        if not orbbec:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped returning frames.")
                break
        preview = frame.copy()
        draw_banner(
            preview,
            [
                f"saved {index - 1}   camera {camera_index}",
                "Space/C save    Q quit",
                "Move the vial before every photo",
                "Later: box lid TOP through cup BOTTOM",
            ],
        )
        cv2.imshow(window, preview)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord(" "), ord("c"), ord("C")):
            path = RAW / f"vial_{index:04d}.jpg"
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"Saved {path.name}")
            index += 1
        elif key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    labeled = sum(1 for _, label in image_label_pairs() if label.exists())
    total = len(image_label_pairs())
    print(f"{total} photos in {RAW}. Labeled: {labeled}.")
    print("Next: python train_vial_model.py label")


def import_images(folder: Path) -> None:
    if not folder.is_dir():
        print(f"Folder not found: {folder}")
        sys.exit(1)
    RAW.mkdir(parents=True, exist_ok=True)
    sources = [
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and RAW not in path.parents
    ]
    if not sources:
        print(f"No images found in {folder}")
        sys.exit(1)

    index = next_image_index()
    copied = 0
    for source in sources:
        dest = RAW / f"vial_{index:04d}.jpg"
        image = cv2.imread(str(source))
        if image is None:
            print(f"Skipped unreadable file: {source}")
            continue
        cv2.imwrite(str(dest), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        label_source = source.with_suffix(".txt")
        if label_source.exists():
            shutil.copy2(label_source, dest.with_suffix(".txt"))
        print(f"{source.name} -> {dest.name}")
        index += 1
        copied += 1
    print(f"Imported {copied} images into {RAW}")
    print("Next: python train_vial_model.py label")


def fit_display(image, max_w: int = 1400, max_h: int = 900):
    height, width = image.shape[:2]
    scale = min(max_w / width, max_h / height)
    scale = min(scale, 3.0)
    display = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
    return display, scale


def box_pixel_aspect(x1: float, y1: float, x2: float, y2: float) -> float:
    """Height / width in pixels."""
    return abs(y2 - y1) / max(abs(x2 - x1), 1e-6)


def is_full_vial_box(x1: float, y1: float, x2: float, y2: float) -> bool:
    aspect = box_pixel_aspect(x1, y1, x2, y2)
    return FULL_VIAL_ASPECT_MIN <= aspect <= FULL_VIAL_ASPECT_MAX


def label_has_short_vial_box(label_path: Path, width: int, height: int) -> bool:
    """True if any non-empty box looks lid-only (too short for a full vial)."""
    for x1, y1, x2, y2 in load_boxes(label_path, width, height):
        if not is_full_vial_box(x1, y1, x2, y2):
            return True
    return False


def load_boxes(label_path: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        boxes.append(
            (
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            )
        )
    return boxes


def save_boxes(label_path: Path, boxes, width: int, height: int) -> int:
    lines = []
    skipped_short = 0
    for x1, y1, x2, y2 in boxes:
        left, right = sorted((float(x1), float(x2)))
        top, bottom = sorted((float(y1), float(y2)))
        left = max(0.0, min(width, left))
        right = max(0.0, min(width, right))
        top = max(0.0, min(height, top))
        bottom = max(0.0, min(height, bottom))
        if right - left < MIN_BOX_PX or bottom - top < MIN_BOX_PX:
            continue
        if not is_full_vial_box(left, top, right, bottom):
            skipped_short += 1
            continue
        cx = min(1.0, max(0.0, ((left + right) / 2) / width))
        cy = min(1.0, max(0.0, ((top + bottom) / 2) / height))
        bw = min(1.0, max(0.0, (right - left) / width))
        bh = min(1.0, max(0.0, (bottom - top) / height))
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    if skipped_short and not lines and boxes:
        print(
            f"  refusing save on {label_path.name}: box(es) too short "
            f"(need h/w ~{FULL_VIAL_ASPECT_MIN:.1f}–{FULL_VIAL_ASPECT_MAX:.1f}, "
            "lid TOP through cup BOTTOM). Extend the box."
        )
        return -1
    label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    if skipped_short:
        print(
            f"  skipped {skipped_short} too-short box(es) on {label_path.name} "
            f"(need h/w ~{FULL_VIAL_ASPECT_MIN:.1f}–{FULL_VIAL_ASPECT_MAX:.1f})"
        )
    return len(lines)


class LabelSession:
    def __init__(self, pairs: list[tuple[Path, Path]]):
        self.pairs = pairs
        self.index = 0
        self.boxes: list[tuple[float, float, float, float]] = []
        self.drawing = False
        self.start = (0, 0)
        self.current = (0, 0)
        self.scale = 1.0
        self.image = None
        self.window = "Label vials"
        self._load_current()

    def _load_current(self) -> None:
        image_path, label_path = self.pairs[self.index]
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            self.boxes = []
            self.scale = 1.0
            return
        height, width = self.image.shape[:2]
        self.boxes = load_boxes(label_path, width, height)
        _, self.scale = fit_display(self.image)

    def _save_current(self) -> bool:
        """Save labels. Returns False if short-only boxes blocked the write."""
        if self.image is None:
            return True
        _, label_path = self.pairs[self.index]
        height, width = self.image.shape[:2]
        kept = save_boxes(label_path, self.boxes, width, height)
        if kept < 0:
            print(
                f"NOT saved {self.pairs[self.index][0].name}: extend the box to cover "
                "lid through cup bottom (or clear with C for an empty-bench label)."
            )
            return False
        print(f"Saved {self.pairs[self.index][0].name} ({kept} boxes)")
        return True

    def _to_image(self, x: int, y: int) -> tuple[float, float]:
        return x / self.scale, y / self.scale

    def on_mouse(self, event, x, y, flags, param) -> None:
        if self.image is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start = self._to_image(x, y)
            self.current = self.start
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current = self._to_image(x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            end = self._to_image(x, y)
            left, right = sorted((self.start[0], end[0]))
            top, bottom = sorted((self.start[1], end[1]))
            if right - left >= MIN_BOX_PX and bottom - top >= MIN_BOX_PX:
                if not is_full_vial_box(left, top, right, bottom):
                    aspect = box_pixel_aspect(left, top, right, bottom)
                    print(
                        f"Box h/w={aspect:.2f} looks too short (lid-only?). "
                        f"Extend from lid TOP through cup BOTTOM "
                        f"(~{FULL_VIAL_ASPECT_MIN:.1f}–{FULL_VIAL_ASPECT_MAX:.1f})."
                    )
                self.boxes.append((left, top, right, bottom))

    def _render(self):
        image_path, label_path = self.pairs[self.index]
        if self.image is None:
            blank = cv2.imread(str(image_path))
            return blank if blank is not None else None
        display, _ = fit_display(self.image)

        def draw(box, color) -> None:
            x1, y1, x2, y2 = box
            p1 = (int(x1 * self.scale), int(y1 * self.scale))
            p2 = (int(x2 * self.scale), int(y2 * self.scale))
            cv2.rectangle(display, p1, p2, color, 2)

        for box in self.boxes:
            color = (0, 220, 0) if is_full_vial_box(*box) else (0, 0, 220)
            draw(box, color)
        if self.drawing:
            draw((self.start[0], self.start[1], self.current[0], self.current[1]), (0, 220, 255))

        state = "labeled" if label_path.exists() else "UNLABELED"
        draw_banner(
            display,
            [
                f"{self.index + 1}/{len(self.pairs)}  {image_path.name}  {state}  boxes {len(self.boxes)}",
                "MUST cover FULL vial: lid TOP through clear cup BOTTOM",
                f"Good h/w ~{FULL_VIAL_ASPECT_MIN:.1f}–{FULL_VIAL_ASPECT_MAX:.1f}  (red box = too short)",
                "N next   P previous   Z undo   C clear   Q save and quit",
                "Empty bench: C then N (no box). Do not invent empty scenes.",
            ],
        )
        return display

    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self.on_mouse)
        print("Box every vial from TOP OF LID through BOTTOM OF CLEAR CUP.")
        print(f"Boxes shorter than h/w {FULL_VIAL_ASPECT_MIN:.1f} are rejected (lid-only).")
        while True:
            display = self._render()
            if display is not None:
                cv2.imshow(self.window, display)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                if self._save_current():
                    break
            if key in (ord("z"), ord("Z")) and self.boxes:
                self.boxes.pop()
            elif key in (ord("c"), ord("C")):
                self.boxes = []
            elif key in (ord("n"), ord("N"), 83, 3):  # n, right arrow
                if self._save_current():
                    self.index = min(self.index + 1, len(self.pairs) - 1)
                    self._load_current()
            elif key in (ord("p"), ord("P"), 81, 2):  # p, left arrow
                if self._save_current():
                    self.index = max(self.index - 1, 0)
                    self._load_current()
        cv2.destroyAllWindows()


def label() -> None:
    pairs = image_label_pairs()
    if not pairs:
        print(f"No photos in {RAW}")
        print("Take some first: python train_vial_model.py capture --camera 0")
        sys.exit(1)
    LabelSession(pairs).run()
    print_status()
    print("Next, when every photo is labeled: python train_vial_model.py train")


def print_status() -> None:
    pairs = image_label_pairs()
    labeled = [(image, label_path) for image, label_path in pairs if label_path.exists()]
    boxes = sum(count_boxes(label_path) for _, label_path in labeled)
    empty = sum(1 for _, label_path in labeled if count_boxes(label_path) == 0)
    print(f"Photos: {len(pairs)}")
    print(f"Labeled: {len(labeled)}   still need a box or an empty label: {len(pairs) - len(labeled)}")
    print(f"Vial boxes: {boxes}   empty-bench photos: {empty}")
    print(f"Folder: {RAW}")
    if len(labeled) < 50:
        print("Aim for at least 80 photos, 100 to 200 if the vial moves around or the light changes.")


def quarantine_short_labels(pairs: list[tuple[Path, Path]]) -> list[tuple[Path, Path]]:
    """Drop (and archive) images whose vial boxes are too short — usually lid-only crops."""
    kept: list[tuple[Path, Path]] = []
    moved = 0
    for image_path, label_path in pairs:
        if count_boxes(label_path) == 0:
            kept.append((image_path, label_path))
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable {image_path.name}")
            continue
        height, width = image.shape[:2]
        if not label_has_short_vial_box(label_path, width, height):
            kept.append((image_path, label_path))
            continue
        SHORT_LABEL_DIR.mkdir(parents=True, exist_ok=True)
        dest_img = SHORT_LABEL_DIR / image_path.name
        dest_lbl = SHORT_LABEL_DIR / label_path.name
        if dest_img.exists():
            dest_img = SHORT_LABEL_DIR / f"{image_path.stem}_short{image_path.suffix}"
            dest_lbl = SHORT_LABEL_DIR / f"{image_path.stem}_short.txt"
        shutil.move(str(image_path), str(dest_img))
        if label_path.exists():
            shutil.move(str(label_path), str(dest_lbl))
        moved += 1
        print(f"  quarantined short-aspect {image_path.name} -> {SHORT_LABEL_DIR.name}/")
    if moved:
        print(
            f"Moved {moved} short-box photo(s) out of raw/ "
            f"(need h/w ~{FULL_VIAL_ASPECT_MIN:.1f}–{FULL_VIAL_ASPECT_MAX:.1f}). "
            "Relabel full vial or leave them out of training."
        )
    return kept


def warn_short_aspects(pairs: list[tuple[Path, Path]]) -> list[str]:
    """Report lid-only-looking boxes without moving anything out of raw/."""
    flagged: list[str] = []
    for image_path, label_path in pairs:
        if count_boxes(label_path) == 0:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        if label_has_short_vial_box(label_path, width, height):
            flagged.append(image_path.name)
    if flagged:
        print(
            f"Warning: {len(flagged)} label(s) look short (h/w outside "
            f"{FULL_VIAL_ASPECT_MIN:.1f}–{FULL_VIAL_ASPECT_MAX:.1f}): {', '.join(flagged[:8])}"
            + ("…" if len(flagged) > 8 else "")
        )
        print("Left in raw/ (no wipe). Relabel lid TOP through cup BOTTOM when you can.")
    return flagged


def _count_positives_in_split(split: str) -> int:
    label_dir = SPLIT_DIR / "labels" / split
    if not label_dir.exists():
        return 0
    return sum(count_boxes(path) for path in label_dir.glob("*.txt"))


def validate_split_dir(seed: int | None = None) -> None:
    """Fail loud if split_70_10_20 is incomplete or has no train positives."""
    yaml_path = SPLIT_DIR / "data.yaml"
    if not yaml_path.exists():
        print(f"Missing split yaml: {yaml_path}")
        sys.exit(1)
    for split in ("train", "val", "test"):
        idir = SPLIT_DIR / "images" / split
        ldir = SPLIT_DIR / "labels" / split
        if not idir.is_dir() or not ldir.is_dir():
            print(f"Split incomplete: need images/ and labels/ for '{split}' under {SPLIT_DIR}")
            sys.exit(1)
        images = sorted(
            p for p in idir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not images:
            print(f"Split '{split}' has zero images at {idir}")
            sys.exit(1)
        missing = [p.name for p in images if not (ldir / f"{p.stem}.txt").exists()]
        if missing:
            print(f"Split '{split}' missing labels for: {missing[:8]}{'...' if len(missing) > 8 else ''}")
            sys.exit(1)
    train_pos = _count_positives_in_split("train")
    if train_pos < 1:
        print("Train split has zero positive boxes. Label vials in raw/, then split again.")
        sys.exit(1)
    manifest_path = SPLIT_DIR / "manifest.json"
    if manifest_path.exists():
        import json

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"Corrupt manifest.json: {exc}")
            sys.exit(1)
        if seed is not None and int(manifest.get("seed", -1)) != int(seed):
            print(
                f"WARNING: manifest seed={manifest.get('seed')} != requested seed={seed}. "
                "Rebuild with: python train_vial_model.py split --seed "
                f"{seed}"
            )
        counts = manifest.get("counts") or {}
        for key, split in (("train", "train"), ("validate", "val"), ("test", "test")):
            expected = counts.get(key)
            if expected is None:
                continue
            actual = len(
                [
                    p
                    for p in (SPLIT_DIR / "images" / split).iterdir()
                    if p.suffix.lower() in IMAGE_EXTS
                ]
            )
            if actual != int(expected):
                print(
                    f"Split count mismatch for {split}: manifest {expected}, disk {actual}. "
                    "Rebuild: python train_vial_model.py split"
                )
                sys.exit(1)
    print(
        f"Split OK: train pos boxes={train_pos}, "
        f"val pos={_count_positives_in_split('val')}, "
        f"test pos={_count_positives_in_split('test')}"
    )


def refresh_70_10_20(seed: int = DEFAULT_SPLIT_SEED) -> Path:
    """Rebuild split_70_10_20 from raw/ without deleting raw images or labels."""
    if not RAW.exists():
        print(f"No raw dataset at {RAW}")
        sys.exit(1)

    pairs_meta: list[tuple[str, str, bool]] = []
    labeled_pairs: list[tuple[Path, Path]] = []
    unlabeled_images = 0
    for img in sorted(RAW.glob("*.jpg")):
        txt = img.with_suffix(".txt")
        if not txt.exists():
            unlabeled_images += 1
            continue
        has_box = bool(txt.read_text(encoding="utf-8").strip())
        pairs_meta.append((img.name, txt.name, has_box))
        labeled_pairs.append((img, txt))

    if unlabeled_images:
        print(
            f"WARNING: {unlabeled_images} jpg(s) in raw/ have no .txt — skipped "
            "(run label, or add empty .txt for hard negatives)."
        )

    if len(pairs_meta) < 4:
        print("Need at least 4 labeled photos before training.")
        print("Take more with capture, then draw boxes with label.")
        sys.exit(1)

    boxed = sum(1 for _, _, has in pairs_meta if has)
    if boxed == 0:
        print("Every label is empty. Drag a box around each vial, then train again.")
        sys.exit(1)

    warn_short_aspects(labeled_pairs)
    if len(pairs_meta) < 80:
        print(
            f"Only {len(pairs_meta)} labeled pairs ({boxed} positives). "
            "Fine-tuning helps, but reliable multi-angle detection needs ~100+ Orbbec stills."
        )

    rng = random.Random(seed)
    rng.shuffle(pairs_meta)
    n = len(pairs_meta)
    n_test = max(1, round(n * 0.20))
    n_val = max(1, round(n * 0.10))
    n_train = n - n_test - n_val
    if n_train < 1:
        print(f"Not enough images after split: {n}")
        sys.exit(1)

    test = pairs_meta[:n_test]
    val = pairs_meta[n_test : n_test + n_val]
    train_items = pairs_meta[n_test + n_val :]

    train_pos = sum(1 for _, _, has in train_items if has)
    if train_pos < 1:
        print(
            "Train split landed with zero positives after shuffle "
            f"(seed={seed}). Add more labeled vials, then retry."
        )
        sys.exit(1)

    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)
    SPLIT_DIR.mkdir(parents=True)

    import json

    manifest = {
        "seed": seed,
        "counts": {
            "train": len(train_items),
            "validate": len(val),
            "test": len(test),
            "total": n,
        },
        "train": [p[0] for p in train_items],
        "validate": [p[0] for p in val],
        "test": [p[0] for p in test],
        "positives": boxed,
        "empty_negatives": n - boxed,
        "train_positives": train_pos,
    }
    (SPLIT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def materialize(split: str, items: list[tuple[str, str, bool]]) -> None:
        idir = SPLIT_DIR / "images" / split
        ldir = SPLIT_DIR / "labels" / split
        idir.mkdir(parents=True)
        ldir.mkdir(parents=True)
        for img_name, txt_name, _ in items:
            src_img = RAW / img_name
            src_txt = RAW / txt_name
            if not src_img.exists() or not src_txt.exists():
                print(f"Missing pair during materialize: {img_name} / {txt_name}")
                sys.exit(1)
            shutil.copy2(src_img, idir / img_name)
            shutil.copy2(src_txt, ldir / Path(txt_name).name)

    materialize("train", train_items)
    materialize("val", val)
    materialize("test", test)

    yaml_path = SPLIT_DIR / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {SPLIT_DIR.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                f"  0: {CLASS_NAME}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"70/20/10 seed {seed}: train {len(train_items)} / val {len(val)} / "
        f"test {len(test)}  (pos {boxed}, empty {n - boxed})"
    )
    print(f"Wrote {yaml_path}")
    validate_split_dir(seed=seed)
    return yaml_path


def prepare_split(val_fraction: float = 0.2) -> Path:
    """Legacy 80/20 helper kept for scripts; prefer refresh_70_10_20 for training."""
    del val_fraction
    return refresh_70_10_20(seed=DEFAULT_SPLIT_SEED)


def pick_device():
    try:
        import torch
    except ImportError:
        return "cpu"
    return 0 if torch.cuda.is_available() else "cpu"


def default_pretrained() -> str:
    """Prefer live → r1-winner → bon-winner over training nano from scratch.

    Matches ``depthcamera.find_weights`` so fine-tunes continue from the same
    lineage live already uses (unless the user passes ``--model``).
    """
    for path in (LIVE_WEIGHTS, R1_WINNER_WEIGHTS, BON_WINNER_WEIGHTS):
        if path.exists():
            return str(path)
    return "yolov8s.pt"


def export_live_weights(best: Path) -> Path:
    if not best.exists():
        print(f"Cannot promote missing weights: {best}")
        sys.exit(1)
    LIVE_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, LIVE_WEIGHTS)
    print(f"Exported live weights: {LIVE_WEIGHTS}")
    print(
        "Promote gate reminder: only do this after holdout clearly beats live "
        "(test@0.25 → recall → empty FP). See TRAIN_PROMOTE.md."
    )
    return LIVE_WEIGHTS


def train(
    epochs: int,
    model_name: str,
    imgsz: int,
    *,
    patience: int = DEFAULT_PATIENCE,
    seed: int = DEFAULT_SPLIT_SEED,
    freeze: int = DEFAULT_FREEZE,
    export_live: bool = False,
) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install the trainer first: pip install ultralytics")
        sys.exit(1)

    model_path = Path(model_name)
    if model_path.suffix == ".pt" and not model_path.exists():
        print(f"Weights not found: {model_path}")
        sys.exit(1)

    yaml_path = refresh_70_10_20(seed=seed)
    device = pick_device()
    # Small Orbbec set: keep batch modest on CPU; GPU can go a bit higher.
    batch = 8 if device == 0 else 2
    print(
        f"Training {model_name} on {device} for up to {epochs} epochs "
        f"(patience {patience}, seed {seed}, freeze {freeze})."
    )
    print(
        "Augmentation: mild geometry + HSV-safe color jitter "
        "(keeps white-cap / clear-cup cues; no aggressive hue wipe)."
    )
    if model_path.suffix == ".pt" and model_path.exists():
        print(f"Fine-tuning from existing weights: {model_name}")
    else:
        print(f"Starting from base model: {model_name}")
    if not export_live:
        print(
            "Live export OFF (default). After a clear holdout beat, re-run with "
            "--export-live or copy best.pt → runs/vial-live/weights/."
        )

    model = YOLO(model_name)
    train_kwargs = dict(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=0,
        patience=patience,
        seed=seed,
        project=str(ROOT / "runs"),
        name="vial-ft",
        exist_ok=True,
        # Multi-angle helpful, but avoid Attempt-B strong-aug collapse.
        degrees=15.0,
        translate=0.10,
        scale=0.35,
        shear=0.0,
        perspective=0.0003,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.7,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=15,
        # HSV-safe: small hue, moderate sat/value (Attempt E style).
        hsv_h=0.012,
        hsv_s=0.45,
        hsv_v=0.35,
        erasing=0.0,
    )
    if freeze and freeze > 0:
        train_kwargs["freeze"] = int(freeze)
    results = model.train(**train_kwargs)
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None and isinstance(results, dict):
        save_dir = results.get("save_dir")
    if not save_dir:
        # Ultralytics usually writes here when exist_ok=True / name=vial-ft
        save_dir = ROOT / "runs" / "vial-ft"
    best = Path(save_dir) / "weights" / "best.pt"
    if not best.exists():
        print(f"Training finished, but best weights missing at {best}")
        sys.exit(1)
    print(f"Best weights: {best}")
    print(
        "Holdout eval (do this before promoting):\n"
        f'  python eval_asymmetric_split.py --weights "{best}" '
        "--out training_reliability_metrics.json"
    )
    if export_live:
        live = export_live_weights(best)
        print(f"Live default path: {live}")
        print(f"Track distance: python depthcamera.py --model \"{live}\"")
        print("Dual independent: python depthcamera.py --dual-independent")
    else:
        print(f"Staging only (not live): {best}")
        print(f"Smoke test: python train_vial_model.py test --camera 0 --model \"{best}\"")
        print(f"A/B live without promote: python depthcamera.py --model \"{best}\" --dual-independent")


def latest_weights() -> Path | None:
    """Same preference as depthcamera.find_weights (never newest scratch first)."""
    for preferred in (LIVE_WEIGHTS, R1_WINNER_WEIGHTS, BON_WINNER_WEIGHTS):
        if preferred.exists():
            return preferred
    runs = ROOT / "runs"
    if not runs.exists():
        return None
    candidates = list(runs.rglob("best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def test(camera_index: int, model_path: Path | None) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install the trainer first: pip install ultralytics")
        sys.exit(1)

    weights = model_path or latest_weights()
    if weights is None or not weights.exists():
        print("No trained weights found. Run: python train_vial_model.py train")
        sys.exit(1)

    cap = open_camera(camera_index)
    if not cap.isOpened():
        print(f"Could not open camera {camera_index}.")
        sys.exit(1)

    model = YOLO(str(weights))
    print(f"Testing {weights}. Q quits. Missed poses should become new labeled photos.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        predicted = model.predict(frame, verbose=False, conf=0.35)
        preview = frame
        if predicted is not None:
            for result in predicted:
                plot = getattr(result, "plot", None)
                if callable(plot):
                    drawn = plot()
                    if isinstance(drawn, np.ndarray):
                        preview = drawn
                break
        draw_banner(preview, ["Q quit", "If a pose is missed, photograph it and train again"])
        cv2.imshow("Test vial detector", preview)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect, label, and train a vial detector.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cameras = sub.add_parser("cameras", help="List camera indexes")
    cameras.add_argument("--max-index", type=int, default=6)

    capture_parser = sub.add_parser("capture", help="Save training photos from a camera")
    capture_parser.add_argument("--camera", type=int, default=0)

    import_parser = sub.add_parser("import", help="Copy existing photos into the dataset")
    import_parser.add_argument("folder", type=Path)

    sub.add_parser("label", help="Draw a box around each vial")
    sub.add_parser("status", help="Show how many photos are labeled")

    train_parser = sub.add_parser(
        "train",
        help="Fine-tune on 70/20/10 split (staging runs/vial-ft; promote with --export-live)",
    )
    train_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    train_parser.add_argument(
        "--model",
        default=None,
        help="Weights to fine-tune (default: vial-live -> r1-winner -> bon-winner -> yolov8s.pt)",
    )
    train_parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    train_parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    train_parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    train_parser.add_argument(
        "--freeze",
        type=int,
        default=DEFAULT_FREEZE,
        help=f"Freeze first N backbone layers (default {DEFAULT_FREEZE}; 0 = none)",
    )
    train_parser.add_argument(
        "--export-live",
        action="store_true",
        help="Copy best.pt to runs/vial-live/weights/ (only after clear holdout beat)",
    )
    train_parser.add_argument(
        "--no-export-live",
        action="store_true",
        help=argparse.SUPPRESS,  # legacy alias; export is already off by default
    )

    split_parser = sub.add_parser(
        "split",
        help="Refresh vial_dataset/split_70_10_20 from raw/ (no raw wipe)",
    )
    split_parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    split_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing split_70_10_20 without rebuilding",
    )

    test_parser = sub.add_parser("test", help="Run the trained detector on the camera")
    test_parser.add_argument("--camera", type=int, default=0)
    test_parser.add_argument("--model", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "cameras":
        list_cameras(args.max_index)
    elif args.cmd == "capture":
        capture(args.camera)
    elif args.cmd == "import":
        import_images(args.folder)
    elif args.cmd == "label":
        label()
    elif args.cmd == "status":
        print_status()
    elif args.cmd == "split":
        if args.validate_only:
            validate_split_dir(seed=args.seed)
        else:
            refresh_70_10_20(seed=args.seed)
    elif args.cmd == "train":
        model_name = args.model or default_pretrained()
        export_live = bool(args.export_live) and not bool(args.no_export_live)
        train(
            args.epochs,
            model_name,
            args.imgsz,
            patience=args.patience,
            seed=args.seed,
            freeze=args.freeze,
            export_live=export_live,
        )
    elif args.cmd == "test":
        test(args.camera, args.model)


if __name__ == "__main__":
    main()
