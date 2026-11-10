"""Anonimização (LGPD): pixeliza rostos e placas de veículos detectados pelo modelo."""
from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection


def blur_regions(frame: np.ndarray, dets: list[Detection], pad: float = 0.15, pixel: int = 12) -> np.ndarray:
    h, w = frame.shape[:2]
    for d in dets:
        if d.cls.role != "privacy":
            continue
        bw, bh = (d.x1 - d.x0) * w, (d.y1 - d.y0) * h
        x0 = max(0, int(d.x0 * w - pad * bw))
        y0 = max(0, int(d.y0 * h - pad * bh))
        x1 = min(w, int(d.x1 * w + pad * bw))
        y1 = min(h, int(d.y1 * h + pad * bh))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        small = cv2.resize(frame[y0:y1, x0:x1], (max(1, (x1 - x0) // pixel), max(1, (y1 - y0) // pixel)))
        frame[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    return frame
