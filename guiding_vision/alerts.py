"""Transforma detecções em avisos curtos para o usuário.

A câmera fica nos óculos, na altura dos olhos, então a geometria da caixa basta para
estimar direção (terço do quadro), se está no caminho (faixa central), se está
suspensa na altura da cabeça (base acima do horizonte) e a proximidade (altura da
caixa; no produto final vem do sensor ToF).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from .detector import Detection

HORIZON = 0.55
CORRIDOR = (0.30, 0.70)
NEAR_H = 0.40
MID_H = 0.18

# Um carro ou poste distante também aparece acima do horizonte, mas está no chão.
SUSPENDED = {"placa", "arvore"}


@dataclass
class Alert:
    text: str
    priority: float
    detection: Detection
    near: bool
    head_height: bool
    direction: str


def direction_of(d: Detection) -> str:
    if d.cx < 1 / 3:
        return "à esquerda"
    if d.cx > 2 / 3:
        return "à direita"
    return "à frente"


def in_corridor(d: Detection) -> bool:
    return d.x1 > CORRIDOR[0] and d.x0 < CORRIDOR[1]


def proximity(d: Detection, distance_m: float | None = None) -> str:
    if distance_m is not None:
        return "perto" if distance_m < 1.5 else "médio" if distance_m < 3 else "longe"
    if d.height >= NEAR_H:
        return "perto"
    if d.height >= MID_H:
        return "médio"
    return "longe"


def is_head_height(d: Detection) -> bool:
    return d.cls.name in SUSPENDED and d.y1 < HORIZON


class AlertEngine:
    """Escolhe no máximo um aviso por quadro, evitando fadiga de alarme:
    pessoas só avisam quando estão perto e no caminho, e cada (classe, direção)
    espera `cooldown_s` antes de ser repetida."""

    def __init__(self, cooldown_s: float = 4.0, min_conf: float = 0.4):
        self.cooldown_s = cooldown_s
        self.min_conf = min_conf
        self._last: dict[tuple[str, str], float] = {}

    def candidates(self, dets: list[Detection], distance_m: float | None = None) -> list[Alert]:
        out = []
        for d in dets:
            if d.cls.role == "privacy" or d.conf < self.min_conf:
                continue
            corridor = in_corridor(d)
            prox = proximity(d, distance_m if corridor else None)
            head = is_head_height(d)
            if d.cls.role == "moving" and not (prox == "perto" and corridor):
                continue
            if prox == "longe" and not head:
                continue
            direction = direction_of(d)
            text = d.cls.label_pt + (" na altura da cabeça" if head else "") + ", " + direction
            if prox == "perto":
                text += ", perto"
            priority = {"perto": 3, "médio": 2, "longe": 1}[prox] + (2 if corridor else 0) + (1.5 if head else 0) + d.conf
            out.append(Alert(text, priority, d, prox == "perto", head, direction))
        return sorted(out, key=lambda a: -a.priority)

    def step(self, dets: list[Detection], distance_m: float | None = None, now: float | None = None) -> Alert | None:
        now = now if now is not None else time.monotonic()
        for a in self.candidates(dets, distance_m):
            key = (a.detection.cls.name, a.direction)
            if now - self._last.get(key, -1e9) >= self.cooldown_s:
                self._last[key] = now
                return a
        return None


class FixedObstacleTracker:
    """Só reporta um obstáculo fixo ao mapa de barreiras quando ele aparece em
    `min_hits` dos últimos `window` quadros, filtrando detecções falsas isoladas."""

    def __init__(self, window: int = 15, min_hits: int = 8, report_cooldown_s: float = 60.0, min_conf: float = 0.5):
        self.min_hits = min_hits
        self.cooldown = report_cooldown_s
        self.min_conf = min_conf
        self.history: deque[set[str]] = deque(maxlen=window)
        self._reported: dict[str, float] = {}

    def update(self, dets: list[Detection], now: float | None = None) -> list[Detection]:
        now = now if now is not None else time.monotonic()
        seen = {d.cls.name: d for d in dets if d.cls.fixed and d.cls.api_type and d.conf >= self.min_conf}
        self.history.append(set(seen))
        out = []
        for name, d in seen.items():
            hits = sum(name in h for h in self.history)
            if hits >= self.min_hits and now - self._reported.get(name, -1e9) >= self.cooldown:
                self._reported[name] = now
                out.append(d)
        return out
