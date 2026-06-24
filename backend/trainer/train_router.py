from __future__ import annotations
import argparse, traceback
from datetime import datetime
from threading import Event, Lock, Thread
from typing import Any, Dict

from .anomaly_trainer import AnomalyTrainer
from .base_trainer import TrainerError
from .classification_trainer import ClassificationTrainer
from .config_manager import build_training_command, get_model_by_id, load_models, normalize_config
from .dataset_validator import validate_dataset
from .detection_torchvision_trainer import TorchVisionDetectionTrainer
from .experiment_store import record_experiment
from .log_streamer import LogStreamer
from .ocr_trainer import OCRTrainer
from .pose_trainer import PoseTrainer
from .segmentation_trainer import SegmentationTrainer
from .yolo_trainer import YoloTrainer

TRAINER_CLASSES = {
    "yolo_trainer": YoloTrainer,
    "pose_trainer": PoseTrainer,
    "classification_trainer": ClassificationTrainer,
    "segmentation_trainer": SegmentationTrainer,
    "detection_torchvision_trainer": TorchVisionDetectionTrainer,
    "anomaly_trainer": AnomalyTrainer,
    "ocr_trainer": OCRTrainer,
}

class TrainingRouter:
    def __init__(self, logger: LogStreamer | None = None) -> None:
        self.logger = logger or LogStreamer()
        self._thread: Thread | None = None
        self._stop_event = Event()
        self._lock = Lock()
        self.status = "idle"
        self.error: str | None = None
        self.metrics: Dict[str, Any] = {}
        self.active_config: Dict[str, Any] | None = None
        self.active_model: Dict[str, Any] | None = None
        self.started_at: str | None = None
        self.finished_at: str | None = None

    def _log(self, message: str) -> None:
        self.logger.write(message)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _validate_start(self, config: Dict[str, Any], model_info: Dict[str, Any]) -> None:
        if model_info.get("status") != "supported":
            raise TrainerError(f"Model {model_info.get('name')} is marked as {model_info.get('status')} and cannot be started.")
        trainer_name = model_info.get("trainer")
        if trainer_name not in TRAINER_CLASSES:
            raise TrainerError(f"No real trainer is registered for trainer key: {trainer_name}")
        if config.get("device", {}).get("mode") == "gpu":
            try:
                import torch
            except Exception as exc:
                raise TrainerError(f"GPU mode requires PyTorch, but import failed: {exc}") from exc
            if not torch.cuda.is_available():
                raise TrainerError("GPU mode selected but CUDA is not available. Start is blocked.")
        validation = validate_dataset(config)
        self._log(f"Dataset validation result: valid={validation.get('valid')} format={validation.get('dataset_format')} images={validation.get('total_images')}")
        for warning in validation.get("warnings", []):
            self._log(f"Dataset warning: {warning}")
        if not validation.get("valid"):
            errors = "; ".join(validation.get("errors", []))
            fixes = "; ".join(validation.get("recommended_fixes", []))
            raise TrainerError(f"Dataset validation failed: {errors}. Recommended fixes: {fixes}")
        if validation.get("dataset_format") != model_info.get("dataset_format"):
            raise TrainerError(f"Dataset format {validation.get('dataset_format')} does not match model requirement {model_info.get('dataset_format')}.")

    def start(self, config: Dict[str, Any]) -> Dict[str, Any]:
        config = normalize_config(config)
        model_id = config.get("model", {}).get("id")
        model_info = get_model_by_id(model_id)
        if not model_info:
            raise TrainerError(f"Unknown model id: {model_id}")
        with self._lock:
            if self.is_running():
                raise TrainerError("A training job is already running. Stop it before starting another job.")
            self.logger.reset()
            self.status = "validating"
            self.error = None
            self.metrics = {}
            self.active_config = config
            self.active_model = model_info
            self.started_at = datetime.now().isoformat(timespec="seconds")
            self.finished_at = None
            self._stop_event.clear()

        self._log("VisionTrain training request received.")
        self._log(f"Training command: {build_training_command(config)}")
        self._validate_start(config, model_info)

        self._thread = Thread(target=self._run_worker, args=(config, model_info), daemon=True)
        self._thread.start()
        return {"status": "running", "message": "Training started with real trainer backend.", "command": build_training_command(config)}

    def _run_worker(self, config: Dict[str, Any], model_info: Dict[str, Any]) -> None:
        trainer_name = model_info.get("trainer")
        trainer_cls = TRAINER_CLASSES[trainer_name]
        with self._lock:
            self.status = "running"
        self._log(f"Selected model: {model_info.get('name')} | trainer={trainer_name} | task={model_info.get('task')}")
        self._log(f"Selected device mode: {config.get('device', {}).get('mode')}")
        try:
            trainer = trainer_cls(config, model_info, self._log, self._stop_event)
            metrics = trainer.run()
            with self._lock:
                self.metrics = metrics or {}
                self.status = "stopped" if self._stop_event.is_set() or self.metrics.get("status") == "stopped" else "completed"
                self.finished_at = datetime.now().isoformat(timespec="seconds")
            record_experiment(config, self.status, self.metrics)
            self._log(f"Training status: {self.status}")
        except Exception as exc:
            trace = traceback.format_exc()
            with self._lock:
                self.status = "failed"
                self.error = str(exc)
                self.finished_at = datetime.now().isoformat(timespec="seconds")
                self.metrics = {"error": str(exc)}
            self._log(f"Training failed: {exc}")
            self._log(trace)
            record_experiment(config, "failed", self.metrics, error=str(exc))

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if not self.is_running():
                self.status = "idle" if self.status in {"completed", "failed", "stopped"} else self.status
                return {"status": self.status, "message": "No active training process."}
            self._stop_event.set()
            self.status = "stopped"
        self._log("Stop requested by user. Trainers that support interruption will stop at the next safe boundary.")
        return {"status": "stopped", "message": "Stop signal sent."}

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "running": self.is_running(),
                "error": self.error,
                "metrics": self.metrics,
                "model": self.active_model,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "device_mode": (self.active_config or {}).get("device", {}).get("mode"),
            }

def verify_model_trainer_mapping() -> Dict[str, Any]:
    models = load_models().get("models", [])
    issues = []
    for model in models:
        if model.get("status") == "supported" and model.get("trainer") not in TRAINER_CLASSES:
            issues.append({"model_id": model.get("id"), "trainer": model.get("trainer")})
    return {"ok": not issues, "issues": issues, "supported_model_count": sum(1 for m in models if m.get("status") == "supported")}

def main() -> None:
    parser = argparse.ArgumentParser(description="VisionTrain Studio training router helper")
    parser.add_argument("--check-mapping", action="store_true")
    parser.add_argument("--config", help="Path to a VisionTrain JSON config")
    args, _unknown = parser.parse_known_args()
    if args.check_mapping:
        print(verify_model_trainer_mapping())
        return
    if args.config:
        from .config_manager import load_json
        cfg = load_json(args.config)
        cli_logger = LogStreamer()
        r = TrainingRouter(logger=cli_logger)
        print(r.start(cfg))
        # Wait for direct CLI training to finish. FastAPI uses a background thread.
        while r.is_running():
            import time
            time.sleep(1)
        print(r.get_status())
        return
    parser.print_help()

if __name__ == "__main__":
    main()
