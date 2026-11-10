"""Anonimiza fotos da equipe antes de entrarem no dataset (LGPD). Depois de conferir, apague as originais.

    python scripts/blur_faces_plates.py fotos_originais/ data/raw/own/images
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from guiding_vision.config import Classes  # noqa: E402
from guiding_vision.detector import ObstacleDetector  # noqa: E402
from guiding_vision.privacy import blur_regions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--weights", default=str(ROOT / "models" / "guiding_hand_yolo11.pt"))
    ap.add_argument("--conf", type=float, default=0.15, help="limiar baixo de propósito: melhor borrar a mais do que a menos")
    ap.add_argument("--max-side", type=int, default=1280)
    args = ap.parse_args()

    det = ObstacleDetector(args.weights, Classes.load(), conf=args.conf, imgsz=960)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(Path(args.src).iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".heic"}:
            continue
        img = cv2.imread(str(p))
        if img is None:
            print(f"pulando {p.name} (formato não suportado; converta HEIC para JPG)")
            continue
        h, w = img.shape[:2]
        s = args.max_side / max(h, w)
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        dets = det(img)
        blur_regions(img, dets)
        k = sum(d.cls.role == "privacy" for d in dets)
        cv2.imwrite(str(dst / f"sp_{p.stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        n += 1
        print(f"{p.name}: {k} região(ões) borrada(s)")
    print(f"{n} fotos anonimizadas em {dst}. Confira e apague as originais.")


if __name__ == "__main__":
    main()
