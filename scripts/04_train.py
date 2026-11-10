"""Treino customizado do YOLO11 a partir dos pesos pré-treinados no COCO.

    python scripts/04_train.py                                          # Mac (MPS)
    python scripts/04_train.py --device 0 --batch 32 --cache ram        # GPU CUDA (modelo entregue, Kaggle)
"""
from __future__ import annotations

import argparse
import shutil

from ultralytics import YOLO

from common import DATA, ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps", help="mps, 0 (CUDA), 0,1 (duas GPUs) ou cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default="disk", help="disk, ram ou false")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--name", default="gh_y11s")
    args = ap.parse_args()

    YOLO(args.model).train(
        data=str(DATA / "dataset" / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=False if args.cache == "false" else args.cache,
        fraction=args.fraction,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        patience=15,
        cos_lr=True,
        close_mosaic=10,
        # Aumentos pensados para uma câmera de óculos na rua: luz variável,
        # cabeça levemente inclinada e obstáculos a distâncias diferentes.
        fliplr=0.5,
        hsv_v=0.5,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        seed=42,
        plots=True,
    )
    dest = ROOT / "models" / "guiding_hand_yolo11.pt"
    dest.parent.mkdir(exist_ok=True)
    shutil.copy(ROOT / "runs" / args.name / "weights" / "best.pt", dest)
    print(f"Melhor modelo copiado para {dest}")


if __name__ == "__main__":
    main()
