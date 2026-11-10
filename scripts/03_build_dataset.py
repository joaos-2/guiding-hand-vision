"""Junta as fontes de data/raw/ em um dataset YOLO e separa treino/validação/teste.

As fotos da equipe (data/raw/own) vão metade para o treino e metade para um teste
separado de mundo real (test_sp, avaliado com data_sp.yaml).
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter

import yaml

from common import DATA, class_names

SPLITS = {"openimages": (0.8, 0.1, 0.1), "potholes": (0.8, 0.1, 0.1), "own": (0.5, 0.0, 0.0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    out = DATA / "dataset"
    if out.exists():
        shutil.rmtree(out)
    names = class_names()
    stats: dict[str, Counter] = {}

    for source, (p_train, p_val, _) in SPLITS.items():
        src = DATA / "raw" / source
        if not (src / "labels").exists():
            print(f"[{source}] não encontrado, pulando")
            continue
        stems = sorted(p.stem for p in (src / "labels").glob("*.txt")
                       if (src / "images" / f"{p.stem}.jpg").exists())
        rng.shuffle(stems)
        n = len(stems)
        a, b = int(n * p_train), int(n * (p_train + p_val))
        parts = {"train": stems[:a], "val": stems[a:b], "test_sp" if source == "own" else "test": stems[b:]}
        for split, items in parts.items():
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            c = stats.setdefault(split, Counter())
            for s in items:
                (out / "images" / split / f"{s}.jpg").hardlink_to(src / "images" / f"{s}.jpg")
                lab = (src / "labels" / f"{s}.txt").read_text()
                (out / "labels" / split / f"{s}.txt").write_text(lab)
                c["_imagens"] += 1
                for line in lab.splitlines():
                    if line.strip():
                        c[names[int(line.split()[0])]] += 1
        print(f"[{source}] {n} imagens -> " + ", ".join(f"{k}={len(v)}" for k, v in parts.items()))

    data_yaml = {"path": str(out), "train": "images/train", "val": "images/val", "test": "images/test",
                 "names": dict(enumerate(names))}
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False))
    if (out / "images" / "test_sp").exists():
        sp = dict(data_yaml, test="images/test_sp")
        (out / "data_sp.yaml").write_text(yaml.safe_dump(sp, allow_unicode=True, sort_keys=False))

    print("\nInstâncias por classe:")
    splits = list(stats)
    print(f"{'classe':<14}" + "".join(f"{s:>9}" for s in splits))
    for name in ["_imagens"] + names:
        print(f"{name:<14}" + "".join(f"{stats[s][name]:>9}" for s in splits))


if __name__ == "__main__":
    main()
