from __future__ import annotations
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .exporter.export_router import (
    convert_precision,
    get_export_capabilities,
    get_export_logs,
    get_export_status,
    run_export,
)
from .trainer.config_manager import build_export_command, build_training_command, load_default_config, load_models, normalize_config, save_config
from .trainer.cpu_monitor import get_cpu_monitor
from .trainer.dataset_validator import validate_dataset
from .trainer.experiment_store import list_experiments
from .trainer.gpu_monitor import get_gpu_monitor
from .trainer.hardware_profiler import get_hardware_profile
from .trainer.log_streamer import LogStreamer
from .trainer.model_compatibility import evaluate_model_compatibility
from .trainer.system_info import get_system_info
from .trainer.train_router import TrainingRouter, verify_model_trainer_mapping

app = FastAPI(title="VisionTrain Studio API", version="3.0.0-final")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

logger = LogStreamer()
router = TrainingRouter(logger=logger)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    mapping = verify_model_trainer_mapping()
    return {
        "status": "ok",
        "service": "VisionTrain Studio",
        "version": "3.0.0-final",
        "trainer_mapping_ok": mapping["ok"],
        "supported_model_count": mapping["supported_model_count"],
    }


@app.get("/api/system-info")
def system_info() -> Dict[str, Any]:
    data = get_system_info()
    data["training_status"] = router.get_status()
    return data


@app.get("/api/hardware-profile")
def hardware_profile() -> Dict[str, Any]:
    profile = get_hardware_profile()
    profile["training_status"] = router.get_status()
    return profile


@app.get("/api/cpu-monitor")
def cpu_monitor() -> Dict[str, Any]:
    status = router.get_status()
    cfg = router.active_config or load_default_config()
    train = cfg.get("training", {})
    return get_cpu_monitor(
        active_training=status["running"],
        device_mode=cfg.get("device", {}).get("mode", "cpu"),
        batch_size=int(train.get("batch_size", 8)),
        image_size=int(train.get("image_size", 640)),
    )


@app.get("/api/gpu-monitor")
def gpu_monitor() -> Dict[str, Any]:
    cfg = router.active_config or load_default_config()
    return get_gpu_monitor(
        selected_gpu_id=int(cfg.get("device", {}).get("gpu_id", 0)),
        active_training=router.get_status()["running"],
        device_mode=cfg.get("device", {}).get("mode", "cpu"),
    )


@app.get("/api/models")
def models() -> Dict[str, Any]:
    return load_models()


@app.get("/api/model-compatibility")
def model_compatibility(device_mode: str = Query("cpu"), dataset_format: str | None = Query(None)) -> Dict[str, Any]:
    return evaluate_model_compatibility(device_mode=device_mode, dataset_format=dataset_format)


@app.get("/api/default-config")
def default_config() -> Dict[str, Any]:
    return load_default_config()


@app.post("/api/validate-dataset")
def validate_dataset_endpoint(config: Dict[str, Any]) -> Dict[str, Any]:
    return validate_dataset(normalize_config(config))


@app.post("/api/start-training")
def start_training(config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        cfg = normalize_config(config)
        validation = validate_dataset(cfg)
        if not validation.get("valid"):
            raise ValueError("Dataset validation failed: " + "; ".join(validation.get("errors", [])))
        compat = evaluate_model_compatibility(
            device_mode=cfg.get("device", {}).get("mode", "cpu"),
            dataset_format=validation.get("dataset_format"),
        )
        model_id = cfg.get("model", {}).get("id")
        model_status = next((m for m in compat.get("models", []) if m.get("id") == model_id), None)
        if not model_status or not model_status.get("enabled"):
            reason = model_status.get("reason") if model_status else "Model not found in compatibility registry"
            raise ValueError(f"Model is disabled: {reason}")
        save_config(cfg)
        return router.start(cfg)
    except Exception as exc:
        logger.write(f"Start training rejected: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/stop-training")
def stop_training(_payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return router.stop()


@app.get("/api/training-status")
def training_status() -> Dict[str, Any]:
    return router.get_status()


@app.get("/api/training-logs")
def training_logs(limit: int = 400) -> Dict[str, Any]:
    logs = logger.read(limit=limit)
    if not logs and not router.get_status().get("running"):
        logs = ["No active training process."]
    return {"logs": logs}


@app.get("/api/experiments")
def experiments() -> Dict[str, Any]:
    return {"experiments": list_experiments()}


@app.post("/api/save-config")
def save_config_endpoint(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_config(config)
    info = save_config(normalized)
    return {"status": "saved", **info, "command": build_training_command(normalized)}


@app.post("/api/import-config")
def import_config_endpoint(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_config(config)
    return {"status": "imported", "config": normalized, "command": build_training_command(normalized)}


@app.get("/api/export/capabilities")
def export_capabilities() -> Dict[str, Any]:
    return get_export_capabilities()


@app.post("/api/export/onnx")
def export_onnx(payload: Dict[str, Any]) -> Dict[str, Any]:
    return run_export({**payload, "format": "onnx"})


@app.post("/api/export/tensorrt")
def export_tensorrt(payload: Dict[str, Any]) -> Dict[str, Any]:
    return run_export({**payload, "format": "tensorrt"})


@app.post("/api/export/torchscript")
def export_torchscript(payload: Dict[str, Any]) -> Dict[str, Any]:
    return run_export({**payload, "format": "torchscript"})


@app.post("/api/export/openvino")
def export_openvino(payload: Dict[str, Any]) -> Dict[str, Any]:
    return run_export({**payload, "format": "openvino"})


@app.post("/api/export/precision")
def export_precision(payload: Dict[str, Any]) -> Dict[str, Any]:
    return convert_precision(payload)


@app.get("/api/export/status")
def export_status() -> Dict[str, Any]:
    return get_export_status()


@app.get("/api/export/logs")
def export_logs(limit: int = 300) -> Dict[str, Any]:
    return get_export_logs(limit=limit)


@app.post("/api/export/command-preview")
def export_command_preview(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"command": build_export_command(payload)}
