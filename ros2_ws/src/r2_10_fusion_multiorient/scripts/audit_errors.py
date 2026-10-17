#!/usr/bin/env python3
"""Frame-level audit for the production r2_10 fusion configuration.

Unlike evaluate.py, this treats the synthetic top-down/upright controls and
the real-wide controls as vials. It writes every FN, FP, and bad-IoU frame,
including detector source and annotated evidence images.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2_10_fusion_multiorient.fusion import FusionConfig, FusionVialDetector  # noqa: E402
from r2_10_fusion_multiorient.types import Det, box_iou, classify_orientation, xywh_to_det  # noqa: E402

CAMERA_ROOT = ROOT.parents[2]
SPLIT = CAMERA_ROOT / "vial_dataset" / "split_70_10_20_latest2"
SYNTH = CAMERA_ROOT / "sideways_race" / "synth"
OUT = ROOT / "models" / "error_audit.json"
EVIDENCE = ROOT / "models" / "error_frames"
NOTE_BOX = re.compile(r"(?:box|gt)=\[([0-9, ]+)\]", re.IGNORECASE)


def load_yolo(path: Path, width: int, height: int) -> list[Det]:
    result: list[Det] = []
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        _, cx, cy, bw, bh = fields[:5]
        cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        result.append(
            xywh_to_det(
                round((cx - bw / 2) * width),
                round((cy - bh / 2) * height),
                round(bw * width),
                round(bh * height),
                orientation="upright",
            )
        )
    return result


def note_boxes(sample: dict) -> list[list[int]]:
    match = NOTE_BOX.search(str(sample.get("note", "")))
    if not match:
        return []
    values = [int(v.strip()) for v in match.group(1).split(",")]
    return [values] if len(values) == 4 else []


def samples() -> Iterable[dict]:
    for image in sorted((SPLIT / "images" / "test").iterdir()):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        bgr = cv2.imread(str(image))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gt = load_yolo(SPLIT / "labels" / "test" / f"{image.stem}.txt", w, h)
        bucket = "empty" if not gt else classify_orientation(gt[0])
        if bucket == "topdown":
            bucket = "weak_angle"
        yield {
            "set": "upright_holdout",
            "id": image.stem,
            "kind": bucket,
            "image": image,
            "depth": None,
            "gt": gt,
        }

    manifest = json.loads((SYNTH / "manifest.json").read_text(encoding="utf-8"))
    for sample in manifest["samples"]:
        boxes = list(sample.get("boxes") or [])
        kind = str(sample["kind"])
        if not boxes and kind in {"topdown", "upright", "real_wide"}:
            boxes = note_boxes(sample)
        orientation = (
            "sideways"
            if "sideways" in kind or kind == "real_wide"
            else ("topdown" if kind == "topdown" else "upright")
        )
        yield {
            "set": "sideways_synth",
            "id": sample["id"],
            "kind": kind,
            "image": SYNTH / sample["color"],
            "depth": SYNTH / sample["depth"],
            "gt": [xywh_to_det(*box, orientation=orientation) for box in boxes],
        }


def greedy_match(preds: list[Det], gts: list[Det], threshold: float = 0.5) -> dict:
    unmatched_gt = set(range(len(gts)))
    matched: list[dict] = []
    fp: list[int] = []
    for pi in sorted(range(len(preds)), key=lambda i: preds[i].conf, reverse=True):
        candidates = [(box_iou(preds[pi], gts[gi]), gi) for gi in unmatched_gt]
        best_iou, best_gi = max(candidates, default=(0.0, -1))
        if best_iou >= threshold:
            unmatched_gt.remove(best_gi)
            matched.append({"pred": pi, "gt": best_gi, "iou": round(best_iou, 4)})
        else:
            fp.append(pi)
    best_iou = max((box_iou(p, g) for p in preds for g in gts), default=0.0)
    return {
        "matched": matched,
        "fp_indices": fp,
        "fn_indices": sorted(unmatched_gt),
        "best_iou": round(best_iou, 4),
    }


def draw_evidence(bgr: np.ndarray, gts: list[Det], preds: list[Det]) -> np.ndarray:
    out = bgr.copy()
    for gt in gts:
        x, y, w, h = gt.xywh
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 0), 2)
        cv2.putText(out, "GT", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)
    for pred in preds:
        x, y, w, h = pred.xywh
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 230), 2)
        text = f"{pred.source} {pred.conf:.2f}"
        cv2.putText(out, text, (x, min(out.shape[0] - 5, y + h + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 230), 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument(
        "--extra-yolo-conf",
        type=float,
        default=None,
        help="Override FusionConfig.extra_yolo_conf_after_method_i",
    )
    parser.add_argument(
        "--yolo-floor",
        type=float,
        default=None,
        help="Override FusionConfig.empty_scene_yolo_floor",
    )
    args = parser.parse_args()
    cfg = FusionConfig(
        camera_root=str(CAMERA_ROOT),
        model_path=str(args.model) if args.model else None,
        yolo_conf=0.25,
        enable_rgb_classical_sideways=False,
    )
    if args.extra_yolo_conf is not None:
        cfg.extra_yolo_conf_after_method_i = args.extra_yolo_conf
    if args.yolo_floor is not None:
        cfg.empty_scene_yolo_floor = args.yolo_floor
    detector = FusionVialDetector(cfg)
    errors: list[dict] = []
    totals = Counter()
    by_kind: dict[str, Counter] = defaultdict(Counter)
    source_counts = Counter()
    args.evidence.mkdir(parents=True, exist_ok=True)

    for sample in samples():
        bgr = cv2.imread(str(sample["image"]), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        depth = None
        if sample["depth"] and Path(sample["depth"]).is_file():
            depth = np.load(str(sample["depth"])).astype(np.float32)
        preds = detector.detect(bgr, depth)
        match = greedy_match(preds, sample["gt"])
        wrong = bool(match["fp_indices"] or match["fn_indices"])
        totals["frames"] += 1
        totals["gt"] += len(sample["gt"])
        totals["predictions"] += len(preds)
        totals["tp"] += len(match["matched"])
        totals["fp"] += len(match["fp_indices"])
        totals["fn"] += len(match["fn_indices"])
        totals["wrong_frames"] += int(wrong)
        bucket = by_kind[sample["kind"]]
        bucket["frames"] += 1
        bucket["gt"] += len(sample["gt"])
        bucket["tp"] += len(match["matched"])
        bucket["fp"] += len(match["fp_indices"])
        bucket["fn"] += len(match["fn_indices"])
        bucket["wrong_frames"] += int(wrong)
        source_counts.update(p.source for p in preds)
        if not wrong:
            continue
        evidence = args.evidence / f"{sample['set']}__{sample['id']}.jpg"
        cv2.imwrite(str(evidence), draw_evidence(bgr, sample["gt"], preds))
        errors.append(
            {
                "set": sample["set"],
                "id": sample["id"],
                "kind": sample["kind"],
                "image": str(sample["image"]),
                "evidence": str(evidence),
                "gt": [g.to_dict() for g in sample["gt"]],
                "predictions": [p.to_dict() for p in preds],
                **match,
            }
        )

    payload = {
        "scope": {
            "production_model": str(detector.model_path),
            "iou_threshold": 0.5,
            "upright_holdout": str(SPLIT / "images" / "test"),
            "synth_manifest": str(SYNTH / "manifest.json"),
            "important_difference_from_evaluate_py": (
                "topdown, upright, and real_wide synth controls are scored as vial positives"
            ),
            "fusion_gates": {
                "extra_yolo_conf_after_method_i": cfg.extra_yolo_conf_after_method_i,
                "empty_scene_yolo_floor": cfg.empty_scene_yolo_floor,
                "empty_method_i_keep_conf": cfg.empty_method_i_keep_conf,
                "enable_rgb_classical_sideways": cfg.enable_rgb_classical_sideways,
            },
        },
        "totals": dict(totals),
        "by_kind": {key: dict(value) for key, value in sorted(by_kind.items())},
        "prediction_sources": dict(source_counts),
        "errors": errors,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["totals"], indent=2))
    print(f"wrong frames: {len(errors)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
