"""Regression checks for launch-safe optional preview."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orbbec_dual_sync import preview_gui


def test_default_backend_does_not_probe_cv2_highgui(monkeypatch):
    def unsafe_probe():
        raise AssertionError("default preview must not probe native HighGUI")

    monkeypatch.delenv("ORBBEC_PREVIEW_CV2", raising=False)
    monkeypatch.setattr(preview_gui, "probe_cv2_gui", unsafe_probe)

    window = preview_gui.PreviewWindow("test")
    window._pick_backend()

    assert window.backend == "tk"
    assert window.enabled
