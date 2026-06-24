from __future__ import annotations
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Iterable, List

class LogStreamer:
    """Thread-safe log buffer backed by a local log file."""
    def __init__(self, log_dir: str | Path = "backend/logs", max_lines: int = 2000) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training.log"
        self._buffer: deque[str] = deque(maxlen=max_lines)
        self._lock = Lock()

    def reset(self) -> None:
        with self._lock:
            self._buffer.clear()
            self.log_file.write_text("", encoding="utf-8")

    def write(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        with self._lock:
            self._buffer.append(line)
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def extend(self, messages: Iterable[str]) -> None:
        for message in messages:
            self.write(message)

    def read(self, limit: int = 400) -> List[str]:
        with self._lock:
            return list(self._buffer) if limit <= 0 else list(self._buffer)[-limit:]
