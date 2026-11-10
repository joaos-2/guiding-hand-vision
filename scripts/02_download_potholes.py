"""Baixa datasets abertos de buracos (formato YOLO) do Hugging Face e remapeia para a classe `buraco`.

- rupesh002/pothole-detection-dataset (MIT)
- Ryukijano/Pothole-detection-Yolov8 (CC BY 4.0)
"""
from __future__ import annotations

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

from common import DATA, class_index

HF = "https://huggingface.co"
OUT = DATA / "raw" / "potholes"
BURACO = class_index("buraco")


def save_pair(stem: str, img_bytes: bytes, label_text: str) -> None:
    (OUT / "images" / f"{stem}.jpg").write_bytes(img_bytes)
    lines = [" ".join([str(BURACO)] + p[1:]) for p in (l.split() for l in label_text.strip().splitlines()) if len(p) == 5]
    (OUT / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")


def rupesh() -> int:
    url = f"{HF}/datasets/rupesh002/pothole-detection-dataset/resolve/main/pothole_dataset.zip"
    data = requests.get(url, timeout=300).content
    n = 0
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = set(z.namelist())
        for name in names:
            if "/images/" not in name or not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            label = name.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            if label in names:
                save_pair("ph_a_" + Path(name).stem, z.read(name), z.read(label).decode())
                n += 1
    return n


def ryukijano() -> int:
    repo = "Ryukijano/Pothole-detection-Yolov8"
    files = {s["rfilename"] for s in requests.get(f"{HF}/api/datasets/{repo}", timeout=60).json()["siblings"]}
    imgs = [f for f in files if "/images/" in f]
    base = f"{HF}/datasets/{repo}/resolve/main/"

    def job(f: str) -> int:
        label = f.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if label not in files:
            return 0
        save_pair("ph_b_" + Path(f).stem, requests.get(base + f, timeout=60).content,
                  requests.get(base + label, timeout=60).text)
        return 1

    with ThreadPoolExecutor(16) as ex:
        return sum(tqdm(ex.map(job, imgs), total=len(imgs), desc="Ryukijano"))


def main() -> None:
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    print("rupesh002:", rupesh(), "imagens")
    print("Ryukijano:", ryukijano(), "imagens")


if __name__ == "__main__":
    main()
