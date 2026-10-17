"""Smoke tests that do not require ROS."""

from pathlib import Path

import numpy as np

from agent_10_ensemble.ensemble_pipeline import EnsembleConfig, EnsembleVialDetector
from agent_10_ensemble.nms import box_iou, nms
from agent_10_ensemble.types import Detection


def test_nms_keeps_best():
    dets = [
        Detection(0, 0, 10, 10, 0.9, source="a"),
        Detection(1, 1, 11, 11, 0.5, source="b"),
        Detection(50, 50, 60, 60, 0.8, source="c"),
    ]
    kept = nms(dets, iou_thresh=0.3)
    assert len(kept) == 2
    assert kept[0].score == 0.9


def test_iou_identity():
    b = (0.0, 0.0, 10.0, 10.0)
    assert box_iou(b, b) == 1.0


def test_detect_on_blank():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    det = EnsembleVialDetector(EnsembleConfig(use_yolo=False, verifier_thresh=0.9))
    assert det.detect(img) == []
