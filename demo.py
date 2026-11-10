"""Demonstração em tempo real do componente de Visão do Guiding Hand.

    python demo.py                                         # webcam
    python demo.py --source calcada.mp4 --save out/demo.mp4
    python demo.py --api http://localhost                  # envia barreiras ao mapa

Teclas: q = sair · v = liga/desliga voz · espaço = pausa
"""
from __future__ import annotations

import argparse
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np

from guiding_vision.alerts import CORRIDOR, HORIZON, AlertEngine, FixedObstacleTracker
from guiding_vision.api_client import BarrierReporter
from guiding_vision.config import ROOT, Classes
from guiding_vision.detector import ObstacleDetector
from guiding_vision.privacy import blur_regions
from guiding_vision.speech import Speaker

COLORS = {"obstacle": (0, 140, 255), "moving": (255, 160, 0), "privacy": (160, 160, 160)}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def frames(source: str):
    p = Path(source)
    if p.is_dir() or p.suffix.lower() in IMG_EXT:
        files = sorted(f for f in (p.iterdir() if p.is_dir() else [p]) if f.suffix.lower() in IMG_EXT)
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                yield img, True
        return
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Não consegui abrir a fonte de vídeo: {source}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame, False
    cap.release()


def draw(frame: np.ndarray, dets, alert_text: str, fps: float, infer_ms: float, voice: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (int(CORRIDOR[0] * w), 0), (int(CORRIDOR[1] * w), h), (0, 255, 0), -1)
    frame = cv2.addWeighted(overlay, 0.06, frame, 0.94, 0)
    cv2.line(frame, (0, int(HORIZON * h)), (w, int(HORIZON * h)), (0, 200, 255), 1)
    cv2.putText(frame, "linha da cabeca", (8, int(HORIZON * h) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    for d in dets:
        if d.cls.role == "privacy":
            continue
        c = COLORS[d.cls.role]
        p0, p1 = (int(d.x0 * w), int(d.y0 * h)), (int(d.x1 * w), int(d.y1 * h))
        cv2.rectangle(frame, p0, p1, c, 2)
        label = f"{d.cls.name} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (p0[0], p0[1] - th - 6), (p0[0] + tw + 4, p0[1]), c, -1)
        cv2.putText(frame, label, (p0[0] + 2, p0[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.rectangle(frame, (0, 0), (w, 34), (0, 0, 0), -1)
    hud = f"FPS {fps:4.1f} | inferencia {infer_ms:5.1f} ms | voz {'on' if voice else 'off'}"
    cv2.putText(frame, hud, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if alert_text:
        cv2.rectangle(frame, (0, h - 44), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, alert_text, (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame


def strip_accents(s: str) -> str:
    """O OpenCV não desenha acentos; o HUD usa o texto sem acento e a voz usa o original."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="índice da webcam, arquivo de vídeo, imagem ou pasta")
    ap.add_argument("--weights", default=str(ROOT / "models" / "guiding_hand_yolo11.pt"))
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-speak", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--save", help="salva o vídeo (ou imagens, se a fonte for pasta) com as anotações")
    ap.add_argument("--api", help="URL do Guiding Hand Cloud, ex.: http://localhost")
    ap.add_argument("--lat", type=float, default=-23.5614)
    ap.add_argument("--lng", type=float, default=-46.6559)
    ap.add_argument("--image-delay", type=int, default=1500, help="ms por imagem quando a fonte é pasta")
    args = ap.parse_args()

    classes = Classes.load()
    detector = ObstacleDetector(args.weights, classes, conf=args.conf, imgsz=args.imgsz)
    engine = AlertEngine()
    tracker = FixedObstacleTracker()
    speaker = Speaker(enabled=not args.no_speak)
    reporter = BarrierReporter(args.api, args.lat, args.lng)
    print(f"Modelo: {args.weights} | dispositivo: {detector.device}")

    writer = None
    fps, last_t, alert_text, alert_until = 0.0, time.perf_counter(), "", 0.0
    latencies = []
    paused = False
    for i, (frame, is_image) in enumerate(frames(args.source)):
        t0 = time.perf_counter()
        dets = detector(frame)
        blur_regions(frame, dets)  # antes de qualquer exibição ou gravação

        alert = engine.step(dets, now=time.monotonic() if not is_image else i * 10.0)
        if alert:
            latencies.append((time.perf_counter() - t0) * 1000)
            alert_text, alert_until = alert.text, time.monotonic() + 2.5
            speaker.say(alert.text)
        for d in tracker.update(dets) if not is_image else []:
            reporter.report(d.cls.api_type)
        if time.monotonic() > alert_until:
            alert_text = ""

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1 / max(now - last_t, 1e-6)) if fps else 1 / max(now - last_t, 1e-6)
        last_t = now
        vis = draw(frame, dets, strip_accents(alert_text), fps, detector.last_ms, speaker.enabled)

        if args.save:
            if is_image:
                Path(args.save).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(args.save) / f"{i:04d}.jpg"), vis)
            else:
                if writer is None:
                    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20, (vis.shape[1], vis.shape[0]))
                writer.write(vis)
        if not args.no_show:
            cv2.imshow("Guiding Hand - Visao", vis)
            key = cv2.waitKey(args.image_delay if is_image else 1) & 0xFF
            while paused and key != ord(" "):
                key = cv2.waitKey(50) & 0xFF
            if key == ord("q"):
                break
            if key == ord("v"):
                speaker.enabled = not speaker.enabled
            if key == ord(" "):
                paused = not paused

    if writer:
        writer.release()
    cv2.destroyAllWindows()
    if latencies:
        print(f"Latência detecção→aviso: média {np.mean(latencies):.1f} ms, p95 {np.percentile(latencies, 95):.1f} ms "
              f"({len(latencies)} avisos)")
    if args.api:
        time.sleep(1)
        print(f"Barreiras enviadas ao mapa: {reporter.sent} (falhas: {reporter.failed})")


if __name__ == "__main__":
    main()
