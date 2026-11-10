"""Envia obstáculos fixos confirmados ao Guiding Hand Cloud (POST /api/events).

Só saem o tipo do obstáculo e uma posição aproximada, nunca imagens. O envio roda
em segundo plano para que uma falha de rede não atrase o aviso ao usuário.
"""
from __future__ import annotations

import queue
import threading

import requests


class BarrierReporter:
    def __init__(self, base_url: str | None, lat: float, lng: float, timeout: float = 3.0):
        self.base_url = base_url.rstrip("/") if base_url else None
        self.lat, self.lng = lat, lng
        self.timeout = timeout
        self.q: queue.Queue[str] = queue.Queue(maxsize=100)
        self.sent = 0
        self.failed = 0
        if self.base_url:
            threading.Thread(target=self._loop, daemon=True).start()

    def report(self, api_type: str) -> None:
        if not self.base_url:
            print(f"[mapa] (offline) barreira detectada: {api_type}")
            return
        try:
            self.q.put_nowait(api_type)
        except queue.Full:
            self.failed += 1

    def _loop(self) -> None:
        while True:
            body = {"obstacle": self.q.get(), "lat": round(self.lat, 4), "lng": round(self.lng, 4)}
            try:
                r = requests.post(f"{self.base_url}/api/events", json=body, timeout=self.timeout)
                r.raise_for_status()
                self.sent += 1
                print(f"[mapa] enviado {body} -> {r.json().get('status')}")
            except requests.RequestException as e:
                self.failed += 1
                print(f"[mapa] falha ao enviar {body}: {e}")
