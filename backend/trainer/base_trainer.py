from __future__ import annotations
from pathlib import Path
from threading import Event
from typing import Any, Callable, Dict

class TrainerError(RuntimeError):
    pass

class BaseTrainer:
    """Shared behavior for real training backends."""
    def __init__(self, config: Dict[str, Any], model_info: Dict[str, Any], logger: Callable[[str], None], stop_event: Event) -> None:
        self.config = config
        self.model_info = model_info
        self.logger = logger
        self.stop_event = stop_event
        self.metrics: Dict[str, Any] = {}

    def log(self, message: str) -> None:
        self.logger(message)

    @property
    def dataset_path(self) -> Path:
        return Path(self.config.get("dataset", {}).get("path", "")).expanduser()

    @property
    def output_dir(self) -> Path:
        path = Path(self.config.get("output", {}).get("output_dir") or "backend/runs")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def experiment_name(self) -> str:
        return self.config.get("output", {}).get("experiment_name") or "exp"

    def resolve_torch_device(self) -> str:
        try:
            import torch
        except Exception as exc:
            raise TrainerError(f"PyTorch is required for this trainer but failed to import: {exc}") from exc
        mode = self.config.get("device", {}).get("mode", "cpu")
        gpu_id = int(self.config.get("device", {}).get("gpu_id", 0))
        if mode == "gpu":
            if not torch.cuda.is_available():
                raise TrainerError("GPU mode was selected, but torch.cuda.is_available() is False.")
            return f"cuda:{gpu_id}"
        return "cpu"

    def should_stop(self) -> bool:
        return self.stop_event.is_set()

    def run(self) -> Dict[str, Any]:
        raise NotImplementedError
