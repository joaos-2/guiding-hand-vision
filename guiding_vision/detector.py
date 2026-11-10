"""Wrapper do YOLO treinado: recebe um frame BGR e devolve detecções normalizadas."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from .config import ClassInfo, Classes


@dataclass
class Detection:
    cls: ClassInfo
    conf: float
    x0: float  # coordenadas normalizadas 0..1
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def height(self) -> float:
        return self.y1 - self.y0


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ObstacleDetector:
    def __init__(self, weights: str, classes: Classes, conf: float = 0.35, imgsz: int = 640,
                 device: str | None = None):
        self.model = YOLO(weights)
        self.classes = classes
        self.conf = conf
        self.imgsz = imgsz
        self.device = device or pick_device()
        self.last_ms = 0.0

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        t = time.perf_counter()
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        self.last_ms = (time.perf_counter() - t) * 1000
        h, w = frame.shape[:2]
        out = []
        for (x0, y0, x1, y1), c, k in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
            out.append(Detection(self.classes[int(k)], float(c), x0 / w, y0 / h, x1 / w, y1 / h))
        return out
