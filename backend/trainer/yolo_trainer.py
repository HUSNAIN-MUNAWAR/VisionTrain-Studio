from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from .base_trainer import BaseTrainer, TrainerError

class YoloTrainer(BaseTrainer):
    """Real Ultralytics training backend for YOLO/RT-DETR models."""
    def _data_arg(self) -> str:
        dataset = self.config.get("dataset", {})
        if self.config.get("task") == "image_classification" or dataset.get("format") == "classification_folder":
            return str(self.dataset_path)
        return str(Path(dataset.get("yaml") or self.dataset_path / "dataset.yaml"))

    def run(self) -> Dict[str, Any]:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise TrainerError(f"Ultralytics is not installed or failed to import: {exc}") from exc
        train, device_cfg = self.config.get("training", {}), self.config.get("device", {})
        weights = self.model_info.get("weights") or self.config.get("model", {}).get("weights")
        if not weights:
            raise TrainerError("No Ultralytics weights were provided for this model.")
        data_arg = self._data_arg()
        if not Path(data_arg).exists():
            raise TrainerError(f"Dataset argument does not exist: {data_arg}")
        device = "cpu" if device_cfg.get("mode") == "cpu" else int(device_cfg.get("gpu_id", 0))
        precision = train.get("precision", "fp32")
        half = bool(device != "cpu" and precision == "fp16")
        project, name = str(self.output_dir), self.experiment_name
        self.log(f"Ultralytics trainer selected: {self.model_info.get('name')} ({weights})")
        self.log(f"Dataset: {data_arg}")
        self.log(f"Device: {device} | precision: {precision}")
        self.log("Loading model through ultralytics.YOLO...")
        model = YOLO(weights)

        def _epoch_log(trainer):
            epoch = getattr(trainer, "epoch", None)
            metrics = getattr(trainer, "metrics", {}) or {}
            loss_items = getattr(trainer, "loss_items", None)
            self.log(f"YOLO epoch_end epoch={epoch} metrics={metrics} loss_items={loss_items}")
            if self.should_stop():
                self.log("Stop requested. Raising KeyboardInterrupt at YOLO epoch boundary.")
                raise KeyboardInterrupt("VisionTrain stop requested")
        try:
            model.add_callback("on_fit_epoch_end", _epoch_log)
            model.add_callback("on_train_epoch_end", _epoch_log)
        except Exception as exc:
            self.log(f"Callback registration warning: {exc}")
        try:
            result = model.train(
                data=data_arg, epochs=int(train.get("epochs", 50)), imgsz=int(train.get("image_size", 640)),
                batch=int(train.get("batch_size", 8)), device=device, lr0=float(train.get("learning_rate", 0.001)),
                optimizer=str(train.get("optimizer", "auto")), workers=int(train.get("workers", 4)),
                seed=int(train.get("seed", 42)), patience=int(train.get("patience", 20)), project=project, name=name,
                exist_ok=True, save_period=int(train.get("checkpoint_interval", 1)), half=half,
            )
        except KeyboardInterrupt:
            self.metrics = {"status": "stopped"}
            self.log("YOLO training stopped by user.")
            return self.metrics
        except Exception as exc:
            raise TrainerError(f"Ultralytics training failed: {exc}") from exc
        save_dir = getattr(result, "save_dir", None) or str(Path(project) / name)
        self.metrics = {"status": "completed", "checkpoint": str(save_dir), "score": "see Ultralytics results.csv"}
        self.log(f"YOLO training completed. Results saved at: {save_dir}")
        return self.metrics
