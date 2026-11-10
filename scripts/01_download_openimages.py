"""Baixa um recorte balanceado do Open Images V7 com as classes do Guiding Hand, em formato YOLO.

Saída: data/raw/openimages/{images,labels}/
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from common import DATA, load_classes

CSV_BASE = "https://storage.googleapis.com/openimages"
CSVS = {
    "classes": f"{CSV_BASE}/v7/oidv7-class-descriptions-boxable.csv",
    "validation": f"{CSV_BASE}/v5/validation-annotations-bbox.csv",
    "test": f"{CSV_BASE}/v5/test-annotations-bbox.csv",
    "train": f"{CSV_BASE}/v6/oidv6-train-annotations-bbox.csv",
}
IMG_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
COLS = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsDepiction"]
MAX_SIDE = 1024


def fetch(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.rename(dest)
    return dest


def fetch_train_filtered(dest: Path, mids: set[str]) -> Path:
    """O CSV de treino tem 2,3 GB: lê em streaming e guarda só as linhas das nossas classes."""
    if dest.exists():
        return dest
    tmp = dest.with_suffix(".part")
    with requests.get(CSVS["train"], stream=True, timeout=60) as r, open(tmp, "w") as f:
        r.raise_for_status()
        lines = r.iter_lines(decode_unicode=True)
        f.write(next(lines) + "\n")
        for line in tqdm(lines, desc="Filtrando CSV de treino", unit=" linhas", mininterval=5):
            if line.split(",", 3)[2] in mids:
                f.write(line + "\n")
    tmp.rename(dest)
    return dest


def load_boxes(cache: Path, mids: set[str], use_train: bool) -> pd.DataFrame:
    paths = {s: fetch(CSVS[s], cache / f"{s}-annotations-bbox.csv") for s in ("validation", "test")}
    if use_train:
        paths["train"] = fetch_train_filtered(cache / "train-annotations-bbox.filtered.csv", mids)
    frames = []
    for split, path in paths.items():
        df = pd.read_csv(path, usecols=COLS)
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def dedupe(boxes: list[tuple], iou_thr: float = 0.85) -> list[tuple]:
    """Remove caixas repetidas da mesma classe (ex.: 'Person' e 'Man' marcando a mesma pessoa)."""
    kept: list[tuple] = []
    for b in sorted(boxes, key=lambda b: -(b[3] - b[1]) * (b[4] - b[2])):
        dup = False
        for k in kept:
            if k[0] != b[0]:
                continue
            ix = max(0, min(b[3], k[3]) - max(b[1], k[1]))
            iy = max(0, min(b[4], k[4]) - max(b[2], k[2]))
            inter = ix * iy
            union = (b[3] - b[1]) * (b[4] - b[2]) + (k[3] - k[1]) * (k[4] - k[2]) - inter
            if union > 0 and inter / union > iou_thr:
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


def download_image(split: str, image_id: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        r = requests.get(IMG_URL.format(split=split, image_id=image_id), timeout=60)
        r.raise_for_status()
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return False
        h, w = img.shape[:2]
        scale = MAX_SIDE / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dest), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-class", type=int, default=1000)
    ap.add_argument("--no-train-split", action="store_true", help="usa só validation/test (download menor)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    cache = DATA / "cache"
    out = DATA / "raw" / "openimages"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    classes = load_classes()
    names = pd.read_csv(fetch(CSVS["classes"], cache / "class-descriptions-boxable.csv"),
                        header=None, names=["mid", "name"])
    name_to_mid = dict(zip(names["name"], names["mid"]))
    mid_to_idx = {name_to_mid[n]: idx for idx, c in enumerate(classes) for n in c["openimages"]}

    print("Lendo anotações do Open Images...")
    df = load_boxes(cache, set(mid_to_idx), not args.no_train_split)
    df = df[df["LabelName"].isin(mid_to_idx)].copy()
    df["cls"] = df["LabelName"].map(mid_to_idx)

    depiction_ids = set(df.loc[df["IsDepiction"] == 1, "ImageID"])
    df = df[~df["ImageID"].isin(depiction_ids)]

    per_image: dict[str, dict] = {}
    for row in df.itertuples(index=False):
        d = per_image.setdefault(row.ImageID, {"split": row.split, "boxes": []})
        d["boxes"].append((row.cls, row.XMin, row.YMin, row.XMax, row.YMax))

    images_by_class: dict[int, list[str]] = defaultdict(list)
    for img_id, d in per_image.items():
        for c in {b[0] for b in d["boxes"]}:
            images_by_class[c].append(img_id)

    # Preenche primeiro as classes raras; entre as candidatas, prefere imagens com
    # várias classes nossas, que tendem a ser cenas de rua.
    selected: set[str] = set()
    count = defaultdict(int)
    for c in sorted(images_by_class, key=lambda c: len(images_by_class[c])):
        cands = images_by_class[c]
        random.shuffle(cands)
        cands.sort(key=lambda i: -len({b[0] for b in per_image[i]["boxes"]}))
        for img_id in cands:
            if count[c] >= args.max_per_class:
                break
            if img_id in selected:
                continue
            selected.add(img_id)
            for cc in {b[0] for b in per_image[img_id]["boxes"]}:
                count[cc] += 1

    print("Imagens por classe:")
    for idx, c in enumerate(classes):
        print(f"  {c['name']:<14} {count[idx]:>5}")
    print(f"Total de imagens selecionadas: {len(selected)}")

    with ThreadPoolExecutor(args.workers) as ex:
        jobs = [ex.submit(download_image, per_image[i]["split"], i, out / "images" / f"{i}.jpg") for i in selected]
        ok = sum(f.result() for f in tqdm(as_completed(jobs), total=len(jobs), desc="Baixando imagens"))
    print(f"Imagens baixadas: {ok}/{len(jobs)}")

    for old in (out / "labels").glob("*.txt"):
        old.unlink()
    for img_id in selected:
        if not (out / "images" / f"{img_id}.jpg").exists():
            continue
        lines = [f"{c} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"
                 for c, x0, y0, x1, y1 in dedupe(per_image[img_id]["boxes"])]
        (out / "labels" / f"{img_id}.txt").write_text("\n".join(lines) + "\n")
    print(f"Pronto: {out}")


if __name__ == "__main__":
    main()
