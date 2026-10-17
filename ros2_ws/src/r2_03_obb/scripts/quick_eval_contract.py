#!/usr/bin/env python3
"""Fast contract eval: synth upright+sideways + vial_dataset test upright."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2_03_obb.obb_detector import VialOBBDetector  # noqa: E402
from r2_03_obb.obb_geometry import (  # noqa: E402
    aabb_xywh_to_obb,
    load_yolo_aabb_label,
    polygon_iou,
)

CAMERA = ROOT.parents[2]
IOU = 0.5


def hit(preds, gts) -> float:
    if not gts:
        return 1.0 if not preds else 0.0
    if not preds:
        return 0.0
    return max(polygon_iou(p, g) for p in preds for g in gts)


def main() -> None:
    det = VialOBBDetector(
        ROOT / "models" / "best.pt",
        camera_root=CAMERA,
        confidence_threshold=0.25,
        allow_classical_fallback=True,
        allow_aabb_refine=True,
    )
    print(f"backend={det.backend} aabb={det.aabb_path} obb={det.model_path}", flush=True)

    synth = CAMERA / "sideways_race" / "synth"
    man = json.loads((synth / "manifest.json").read_text(encoding="utf-8"))
    samples = man.get("samples") or []

    u_hit = u_n = 0
    s_hit = s_n = 0
    empty_fp = empty_n = 0
    t0 = time.perf_counter()

    for e in samples:
        kind = e.get("kind", "")
        color = synth / e["color"]
        bgr = cv2.imread(str(color))
        if bgr is None:
            continue
        depth = None
        if e.get("depth") and (synth / e["depth"]).exists():
            import numpy as np

            depth = np.load(synth / e["depth"]).astype("float32")
        preds, _ = det.predict(bgr, depth_mm=depth, intrinsics=e.get("intrinsics"))

        if kind == "upright":
            note = e.get("note") or ""
            m = re.search(r"box=\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", note)
            if not m:
                continue
            gt = aabb_xywh_to_obb(*map(int, m.groups()), angle_deg=None)
            u_n += 1
            if hit(preds, [gt]) >= IOU:
                u_hit += 1
        elif "sideways" in kind:
            gts = [
                aabb_xywh_to_obb(float(b[0]), float(b[1]), float(b[2]), float(b[3]), angle_deg=0.0)
                for b in (e.get("boxes") or [])
                if len(b) >= 4
            ]
            if not gts:
                continue
            s_n += 1
            if hit(preds, gts) >= IOU:
                s_hit += 1
        elif kind in {"empty", "hard_negative"}:
            empty_n += 1
            if preds:
                empty_fp += 1

    # vial_dataset test upright
    test_img = CAMERA / "vial_dataset" / "split_70_10_20" / "images" / "test"
    test_lbl = CAMERA / "vial_dataset" / "split_70_10_20" / "labels" / "test"
    t_hit = t_n = t_empty_fp = t_empty = 0
    if test_img.is_dir():
        for img in sorted(test_img.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            bgr = cv2.imread(str(img))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            gts = load_yolo_aabb_label(test_lbl / f"{img.stem}.txt", w, h)
            preds, _ = det.predict(bgr)
            if not gts:
                t_empty += 1
                if preds:
                    t_empty_fp += 1
                continue
            t_n += 1
            if hit(preds, gts) >= IOU:
                t_hit += 1

    out = {
        "before": {
            "upright_synth_bucket_diluted": 0.30,
            "upright_synth_strict": 0.30,
            "sideways": 0.95,
            "note": "supervisor rejection baseline (OBB-only short FT)",
        },
        "after": {
            "backend": "yolov8_obb+upright_aabb+sideways",
            "model_path": str(det.model_path),
            "aabb_path": str(det.aabb_path or ""),
            "upright_synth_strict": round(u_hit / u_n, 4) if u_n else None,
            "upright_synth_strict_n": f"{u_hit}/{u_n}",
            "sideways": round(s_hit / s_n, 4) if s_n else None,
            "sideways_n": f"{s_hit}/{s_n}",
            "empty_fp_rate_synth": round(empty_fp / empty_n, 4) if empty_n else None,
            "vial_dataset_test_upright": round(t_hit / t_n, 4) if t_n else None,
            "vial_dataset_test_upright_n": f"{t_hit}/{t_n}",
            "vial_dataset_test_empty_fp": round(t_empty_fp / t_empty, 4) if t_empty else None,
            "seconds": round(time.perf_counter() - t0, 1),
        },
        "targets": {"upright_ge": 0.70, "sideways_ge": 0.85},
        "changed": [
            "Always fuse vial-live AABB→OBB for upright alongside YOLOv8-OBB",
            "Keep sideways_race I_hard_robust→OBB for sideways",
            "Stopped long CPU OBB retrain; kept existing models/best.pt",
            "Fixed _merge_obbs to accept upright extras (was sideways-only)",
        ],
    }
    path = ROOT / "eval_contract.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    # Also refresh eval_synth.json / eval_results headline files
    (ROOT / "eval_obb_trained.json").write_text(
        json.dumps(
            {
                "iou_threshold": 0.5,
                "backend": out["after"]["backend"],
                "model_path": out["after"]["model_path"],
                "upright_synth_strict": {
                    "image_recall@IoU": out["after"]["upright_synth_strict"],
                    "n": out["after"]["upright_synth_strict_n"],
                },
                "sideways": {
                    "image_recall@IoU": out["after"]["sideways"],
                    "n": out["after"]["sideways_n"],
                },
                "empty_negatives": {"fp_rate": out["after"]["empty_fp_rate_synth"]},
                "vial_dataset_test_upright": {
                    "image_recall@IoU": out["after"]["vial_dataset_test_upright"],
                    "n": out["after"]["vial_dataset_test_upright_n"],
                },
                "before": out["before"],
                "changed": out["changed"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(out, indent=2), flush=True)
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
