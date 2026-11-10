"""Voz em pt-BR sem bloquear o laço de vídeo (`say` no macOS, `espeak-ng` no Linux)."""
from __future__ import annotations

import platform
import queue
import shutil
import subprocess
import threading


class Speaker:
    def __init__(self, enabled: bool = True, voice: str = "Luciana", rate: int = 210):
        self.enabled = enabled
        self.q: queue.Queue[str] = queue.Queue(maxsize=1)
        if platform.system() == "Darwin" and shutil.which("say"):
            self.cmd = ["say", "-v", voice, "-r", str(rate)]
        elif shutil.which("espeak-ng"):
            self.cmd = ["espeak-ng", "-v", "pt-br"]
        else:
            self.cmd = None
        threading.Thread(target=self._loop, daemon=True).start()

    def say(self, text: str) -> None:
        print(f"[aviso] {text}")
        if not self.enabled or not self.cmd:
            return
        # Aviso antigo perde a validade: fica só o mais recente na fila.
        try:
            self.q.get_nowait()
        except queue.Empty:
            pass
        self.q.put_nowait(text)

    def _loop(self) -> None:
        while True:
            subprocess.run([*self.cmd, self.q.get()], check=False)
