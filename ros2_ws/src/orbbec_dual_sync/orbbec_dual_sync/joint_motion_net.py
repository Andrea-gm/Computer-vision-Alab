"""Tiny online MLP: predict the next joint offset from the purple-tape origin.

Trains itself from HSV tracks while a joint is visible. When the joint leaves
both cameras, the same net rolls the position forward so the ID is kept.
No labeled dataset and no PyTorch — NumPy only, safe inside the ROS process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

HIST = 6  # past positions
SCALE_MM = 400.0  # normalize xyz
HIDDEN = 32
IN_DIM = HIST * 3
MAX_STEP_MM = 70.0


class JointMotionNet:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path
        rng = np.random.default_rng(0)
        self.W1 = rng.normal(0, 0.08, size=(HIDDEN, IN_DIM)).astype(np.float64)
        self.b1 = np.zeros(HIDDEN, dtype=np.float64)
        self.W2 = rng.normal(0, 0.08, size=(3, HIDDEN)).astype(np.float64)
        self.b2 = np.zeros(3, dtype=np.float64)
        self.steps = 0
        self.last_loss = 0.0
        if path is not None and Path(path).is_file():
            self._load(Path(path))

    @property
    def ready(self) -> bool:
        return self.steps >= 40

    def _load(self, path: Path) -> None:
        try:
            data = np.load(path)
            self.W1 = np.asarray(data["W1"], dtype=np.float64)
            self.b1 = np.asarray(data["b1"], dtype=np.float64)
            self.W2 = np.asarray(data["W2"], dtype=np.float64)
            self.b2 = np.asarray(data["b2"], dtype=np.float64)
            self.steps = int(data["steps"]) if "steps" in data else 0
        except Exception:
            return

    def save(self) -> None:
        if self.path is None:
            return
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            steps=np.array(self.steps),
        )

    @staticmethod
    def _feat(hist: np.ndarray) -> np.ndarray:
        """hist: (HIST, 3) mm relative to purple origin → flat normalized."""
        x = np.asarray(hist, dtype=np.float64).reshape(HIST, 3) / SCALE_MM
        return np.clip(x.reshape(-1), -3.0, 3.0)

    def _forward(self, feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = self.W1 @ feat + self.b1
        h = np.tanh(z)
        delta_n = self.W2 @ h + self.b2  # normalized mm / SCALE
        return delta_n, h

    def predict_next(self, hist: np.ndarray) -> np.ndarray:
        """Next xyz (mm) from a (HIST, 3) history. Clamped step."""
        feat = self._feat(hist)
        delta_n, _h = self._forward(feat)
        delta = np.clip(delta_n, -1.5, 1.5) * SCALE_MM
        step = float(np.linalg.norm(delta))
        if step > MAX_STEP_MM:
            delta = delta * (MAX_STEP_MM / step)
        last = np.asarray(hist, dtype=np.float64).reshape(HIST, 3)[-1]
        return last + delta

    def train(self, hist: np.ndarray, target_xyz: np.ndarray, lr: float = 0.05) -> float:
        """One SGD step in normalized units so the weights stay finite."""
        feat = self._feat(hist)
        delta_n, h = self._forward(feat)
        last = np.asarray(hist, dtype=np.float64).reshape(HIST, 3)[-1]
        target_n = (np.asarray(target_xyz, dtype=np.float64).reshape(3) - last) / SCALE_MM
        target_n = np.clip(target_n, -1.5, 1.5)
        err = delta_n - target_n
        loss = float(0.5 * np.dot(err, err))
        dW2 = np.outer(err, h)
        db2 = err
        dh = self.W2.T @ err
        dz = dh * (1.0 - h * h)
        dW1 = np.outer(dz, feat)
        db1 = dz
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        # Keep weights from running away.
        self.W1 = np.clip(self.W1, -2.0, 2.0)
        self.W2 = np.clip(self.W2, -2.0, 2.0)
        self.steps += 1
        self.last_loss = loss
        if self.steps % 80 == 0:
            self.save()
        return loss
