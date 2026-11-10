"""Utilidades compartilhadas pelos scripts de dataset e treino."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLASSES_FILE = ROOT / "configs" / "classes.yaml"


def load_classes() -> list[dict]:
    with open(CLASSES_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)["classes"]


def class_names() -> list[str]:
    return [c["name"] for c in load_classes()]


def class_index(name: str) -> int:
    return class_names().index(name)
