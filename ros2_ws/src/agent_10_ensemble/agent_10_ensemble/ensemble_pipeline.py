"""Multi-stage vial finder: proposals → shape filter → verifier → NMS."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .features import Verifier
from .nms import nms, soft_nms
from .proposals import YoloProposer, propose_cap, propose_edges, shape_filter
from .types import Detection

# Default YOLO weights relative to Camera root (optional recall branch).
_DEFAULT_WEIGHT_CANDIDATES = (
    "runs/vial-live/weights/best.pt",
    "runs/train-race/r3-winner/weights/best.pt",
    "runs/bon-winner/weights/best.pt",
)


@dataclass
class EnsembleConfig:
    """Runtime knobs for the cascade."""

    use_cap: bool = True
    use_edge: bool = True
    use_yolo: bool = True
    yolo_weights: str | None = None
    yolo_conf: float = 0.10
    verifier_path: str | None = None
    verifier_thresh: float = 0.42
    nms_iou: float = 0.45
    use_soft_nms: bool = False
    max_det: int = 5
    min_aspect: float = 0.38
    max_aspect: float = 1.45
    min_area_frac: float = 0.0015
    max_area_frac: float = 0.95
    camera_root: str | None = None
    device: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def resolve_yolo_weights(explicit: str | None, camera_root: str | Path | None) -> str | None:
    if explicit and Path(explicit).is_file():
        return str(Path(explicit))
    here = Path(__file__).resolve()
    default_root = Path(
        os.environ.get("ORBBEC_CAMERA_ROOT", here.parents[4] / "robotics")
    )
    root = Path(camera_root) if camera_root else default_root
    candidates_roots = []
    if camera_root:
        candidates_roots.append(Path(camera_root))
    candidates_roots.extend(
        [
            default_root,
            here.parents[3],  # ros2_ws
            Path.cwd(),
        ]
    )
    for base in candidates_roots:
        for rel in _DEFAULT_WEIGHT_CANDIDATES:
            p = base / rel
            if p.is_file():
                return str(p)
            p2 = base.parent / rel  # if base is ros2_ws
            if p2.is_file():
                return str(p2)
    return explicit if explicit else None


def default_verifier_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "verifier.pkl"


class EnsembleVialDetector:
    """Full cascade used by the ROS node and offline evaluator."""

    def __init__(self, config: EnsembleConfig | None = None):
        self.config = config or EnsembleConfig()
        vpath = self.config.verifier_path or str(default_verifier_path())
        self.verifier = Verifier(vpath if Path(vpath).is_file() else None)
        weights = None
        if self.config.use_yolo:
            weights = resolve_yolo_weights(self.config.yolo_weights, self.config.camera_root)
        self.yolo = YoloProposer(weights, conf=self.config.yolo_conf, device=self.config.device)
        self.yolo_weights = weights if self.yolo.available else None

    def propose(self, bgr: np.ndarray) -> list[Detection]:
        cfg = self.config
        proposals: list[Detection] = []
        if cfg.use_cap:
            proposals.extend(propose_cap(bgr))
        if cfg.use_edge:
            proposals.extend(propose_edges(bgr))
        if cfg.use_yolo and self.yolo.available:
            proposals.extend(self.yolo.propose(bgr))
        h, w = bgr.shape[:2]
        return shape_filter(
            proposals,
            w,
            h,
            min_aspect=cfg.min_aspect,
            max_aspect=cfg.max_aspect,
            min_area_frac=cfg.min_area_frac,
            max_area_frac=cfg.max_area_frac,
        )

    def verify(self, bgr: np.ndarray, proposals: list[Detection]) -> list[Detection]:
        """Score proposals; never drop a strong YOLO hit below a soft floor.

        Classical proposals must clear ``verifier_thresh``. YOLO proposals are
        kept if the verifier agrees *or* the detector confidence is already
        solid — this preserves in-domain YOLO recall while still letting
        cap/edge proposals recover YOLO misses.
        """
        out: list[Detection] = []
        thr = self.config.verifier_thresh
        for det in proposals:
            vscore = self.verifier.score(bgr, det)
            yolo_c = float(det.extras.get("yolo_conf", 0.0))
            if det.source == "yolo":
                ok = (vscore >= thr * 0.70) or (yolo_c >= 0.22)
                # Prefer YOLO geometry in NMS when it fires — classical
                # proposals often score high but localize the lid only.
                score = float(min(0.99, max(vscore, yolo_c) + 0.12))
            else:
                ok = vscore >= thr
                # Cap classical so a verified YOLO box wins overlaps.
                score = float(min(0.82, vscore))
            if not ok:
                continue
            fused = Detection(
                x1=det.x1,
                y1=det.y1,
                x2=det.x2,
                y2=det.y2,
                score=score,
                source=f"ens:{det.source}",
                extras={
                    **det.extras,
                    "proposal_score": det.score,
                    "verifier": vscore,
                },
            )
            out.append(fused)
        return out

    def fuse(self, verified: list[Detection]) -> list[Detection]:
        if self.config.use_soft_nms:
            kept = soft_nms(verified, iou_thresh=self.config.nms_iou)
        else:
            kept = nms(verified, iou_thresh=self.config.nms_iou)
        kept.sort(key=lambda d: d.score, reverse=True)
        return kept[: self.config.max_det]

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        """Run the full multi-stage pipeline on a BGR image."""
        if bgr is None or bgr.size == 0:
            return []
        proposals = self.propose(bgr)
        verified = self.verify(bgr, proposals)
        return self.fuse(verified)

    def annotate(self, bgr: np.ndarray, detections: list[Detection] | None = None) -> np.ndarray:
        if detections is None:
            detections = self.detect(bgr)
        vis = bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det.as_int()
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 80), 2)
            label = f"vial {det.score:.2f} {det.source}"
            cv2.putText(
                vis,
                label,
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 80),
                1,
                cv2.LINE_AA,
            )
        return vis
