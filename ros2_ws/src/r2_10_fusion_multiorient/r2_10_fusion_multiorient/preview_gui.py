"""Launch-safe desktop preview for annotated BGR frames.

Windows RoboStack OpenCV builds may abort a ``ros2 launch`` child when
HighGUI is probed.  Prefer tkinter/Pillow and only use HighGUI when explicitly
enabled with ``R2_10_PREVIEW_CV2=1``.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np


def probe_cv2_gui() -> bool:
    try:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        cv2.namedWindow("__r2_10_gui_probe__", cv2.WINDOW_NORMAL)
        cv2.imshow("__r2_10_gui_probe__", image)
        cv2.waitKey(1)
        cv2.destroyWindow("__r2_10_gui_probe__")
        return True
    except Exception:
        try:
            cv2.destroyWindow("__r2_10_gui_probe__")
        except Exception:
            pass
        return False


class PreviewWindow:
    """Display BGR frames; Q, Escape, or window-close disables the preview."""

    def __init__(
        self,
        title: str,
        *,
        max_height: int = 480,
        allow_cv2: Optional[bool] = None,
    ) -> None:
        self.title = title
        self.max_height = int(max_height)
        if allow_cv2 is None:
            allow_cv2 = os.environ.get("R2_10_PREVIEW_CV2", "").strip() == "1"
        self.allow_cv2 = bool(allow_cv2)
        self.enabled = True
        self.user_closed = False
        self.last_error = ""
        self.backend: Optional[str] = None
        self._tk_root = None
        self._tk_label = None
        self._tk_photo = None
        self._cv2_ok: Optional[bool] = None

    def show(self, bgr: np.ndarray) -> bool:
        if not self.enabled:
            return False
        if bgr is None or not isinstance(bgr, np.ndarray) or bgr.size == 0:
            return True
        frame = self._fit(bgr)
        if self.backend is None:
            self._pick_backend()
        if not self.enabled or self.backend is None:
            return False
        try:
            if self.backend == "cv2":
                return self._show_cv2(frame)
            return self._show_tk(frame)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.enabled = False
            self.backend = None
            return False

    def close(self) -> None:
        self.enabled = False
        if self.backend == "cv2":
            try:
                cv2.destroyWindow(self.title)
            except Exception:
                pass
        root = self._tk_root
        self._tk_root = None
        self._tk_label = None
        self._tk_photo = None
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass

    def _fit(self, bgr: np.ndarray) -> np.ndarray:
        height = int(bgr.shape[0])
        if height <= self.max_height or self.max_height <= 0:
            return bgr
        scale = self.max_height / float(height)
        width = max(1, int(bgr.shape[1] * scale))
        return cv2.resize(bgr, (width, self.max_height))

    def _pick_backend(self) -> None:
        try:
            import tkinter  # noqa: F401
            from PIL import Image, ImageTk  # noqa: F401
        except Exception as exc:
            if self.allow_cv2:
                if self._cv2_ok is None:
                    self._cv2_ok = probe_cv2_gui()
                if self._cv2_ok:
                    self.backend = "cv2"
                    return
            self.last_error = (
                "tkinter/Pillow preview unavailable: "
                f"{type(exc).__name__}: {exc}"
            )
            self.enabled = False
            return
        self.backend = "tk"

    def _disable(self) -> None:
        self.user_closed = True
        self.close()
        self.user_closed = True

    def _show_cv2(self, frame: np.ndarray) -> bool:
        cv2.imshow(self.title, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            self._disable()
            return False
        return True

    def _show_tk(self, frame: np.ndarray) -> bool:
        import tkinter as tk
        from PIL import Image, ImageTk

        if self._tk_root is None:
            root = tk.Tk()
            root.title(self.title)
            root.protocol("WM_DELETE_WINDOW", self._disable)
            for sequence in ("<q>", "<Q>", "<Escape>"):
                root.bind(sequence, lambda _event: self._disable())
            label = tk.Label(root)
            label.pack()
            self._tk_root = root
            self._tk_label = label

        if not self.enabled or self._tk_root is None:
            return False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self._tk_photo = photo
        self._tk_label.configure(image=photo)
        self._tk_root.update_idletasks()
        self._tk_root.update()
        return self.enabled
