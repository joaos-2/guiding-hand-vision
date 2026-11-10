from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ClassInfo:
    idx: int
    name: str
    label_pt: str
    role: str  # obstacle | moving | privacy
    fixed: bool
    api_type: str | None = None


@dataclass
class Classes:
    items: list[ClassInfo] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> "Classes":
        path = path or ROOT / "configs" / "classes.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))["classes"]
        return cls([ClassInfo(i, c["name"], c["label_pt"], c["role"], bool(c["fixed"]), c.get("api_type"))
                    for i, c in enumerate(raw)])

    def by_name(self, name: str) -> ClassInfo:
        return next(c for c in self.items if c.name == name)

    def __getitem__(self, idx: int) -> ClassInfo:
        return self.items[idx]
