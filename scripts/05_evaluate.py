"""Avalia o modelo no conjunto de teste (e no test_sp, se existir), mede a velocidade
de inferência e gera as figuras do relatório em docs/resultados/."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from common import DATA, ROOT

sys.path.insert(0, str(ROOT))
from guiding_vision.alerts import AlertEngine  # noqa: E402
from guiding_vision.config import Classes  # noqa: E402
from guiding_vision.detector import ObstacleDetector  # noqa: E402
from demo import strip_accents  # noqa: E402
from guiding_vision.privacy import blur_regions  # noqa: E402

OUT = ROOT / "docs" / "resultados"


def evaluate(model: YOLO, data_yaml: Path, split: str, device: str, tag: str) -> dict:
    m = model.val(data=str(data_yaml), split=split, device=device, plots=True, project=str(ROOT / "runs"),
                  name=f"eval_{tag}", exist_ok=True, verbose=False)
    names = model.names
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[names[int(c)]] = {"precision": round(p, 3), "recall": round(r, 3),
                                   "mAP50": round(ap50, 3), "mAP50_95": round(ap, 3)}
    res = {"mAP50": round(m.box.map50, 3), "mAP50_95": round(m.box.map, 3),
           "precision": round(m.box.mp, 3), "recall": round(m.box.mr, 3), "per_class": per_class}
    for f in ("confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png"):
        src = Path(m.save_dir) / f
        if src.exists():
            shutil.copy(src, OUT / f"{tag}_{f}")
    return res


def speed(weights: str, device: str, n: int = 50) -> float:
    model = YOLO(weights)
    img = np.random.randint(0, 255, (480, 640, 3), np.uint8)
    for _ in range(5):
        model.predict(img, device=device, verbose=False)
    t = time.perf_counter()
    for _ in range(n):
        model.predict(img, device=device, verbose=False)
    return (time.perf_counter() - t) / n * 1000


def examples(weights: str, n: int = 12) -> None:
    classes = Classes.load()
    det = ObstacleDetector(weights, classes)
    imgs = sorted((DATA / "dataset" / "images" / "test").glob("*.jpg"))
    random.Random(7).shuffle(imgs)
    tiles = []
    for p in imgs:
        frame = cv2.imread(str(p))
        dets = det(frame)
        if not any(d.cls.role != "privacy" for d in dets):
            continue
        blur_regions(frame, dets)
        alert = AlertEngine().step(dets, now=0)
        h, w = frame.shape[:2]
        for d in dets:
            if d.cls.role == "privacy":
                continue
            color = (0, 140, 255) if d.cls.role == "obstacle" else (255, 160, 0)
            cv2.rectangle(frame, (int(d.x0 * w), int(d.y0 * h)), (int(d.x1 * w), int(d.y1 * h)), color, 3)
            cv2.putText(frame, f"{d.cls.name} {d.conf:.2f}", (int(d.x0 * w) + 3, int(d.y0 * h) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        frame = cv2.resize(frame, (640, int(640 * h / w)))
        frame = cv2.copyMakeBorder(frame, 0, max(0, 480 - frame.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:480]
        text = strip_accents(alert.text) if alert else "(sem aviso)"
        cv2.rectangle(frame, (0, 440), (640, 480), (0, 0, 0), -1)
        cv2.putText(frame, text, (8, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(frame)
        if len(tiles) == n:
            break
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles) - len(tiles) % 3, 3)]
    cv2.imwrite(str(OUT / "exemplos_teste.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "guiding_hand_yolo11.pt"))
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    results = {"teste": evaluate(model, DATA / "dataset" / "data.yaml", "test", args.device, "teste")}
    sp = DATA / "dataset" / "data_sp.yaml"
    if sp.exists():
        results["teste_sp"] = evaluate(model, sp, "test", args.device, "teste_sp")
    results["ms_por_imagem"] = {args.device: round(speed(args.weights, args.device), 1),
                                "cpu": round(speed(args.weights, "cpu"), 1)}
    (OUT / "metricas.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    examples(args.weights)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
