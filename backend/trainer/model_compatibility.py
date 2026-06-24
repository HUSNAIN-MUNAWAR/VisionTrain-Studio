from __future__ import annotations

from typing import Any, Dict, List

from .config_manager import load_models
from .hardware_profiler import get_hardware_profile
from .train_router import TRAINER_CLASSES


def evaluate_model_compatibility(device_mode: str = "cpu", dataset_format: str | None = None) -> Dict[str, Any]:
    profile = get_hardware_profile()
    cpu = profile["cpu"]
    gpu = profile["gpu"]
    packages = profile["packages"]
    available_ram = float(cpu.get("ram_total_gb") or 0)
    selected_gpu = gpu.get("selected_device") or {}
    vram = float(selected_gpu.get("vram_total_gb") or 0)
    cuda_ok = bool(gpu.get("cuda_available"))
    models = []
    for model in load_models().get("models", []):
        enabled = True
        reason = "Ready"
        warnings: List[str] = []
        missing_dependency = None

        status = model.get("status", "future")
        if status != "supported":
            enabled = False
            reason = "Future integration"
        trainer = model.get("trainer")
        if enabled and trainer not in TRAINER_CLASSES:
            enabled = False
            reason = f"No real trainer registered for {trainer}"
        family = model.get("family")
        if enabled and family in {"torch", "torchvision", "anomaly", "ultralytics"} and not packages.get("torch"):
            enabled = False
            reason = "PyTorch is not installed"
            missing_dependency = "torch"
        if enabled and family == "ultralytics" and not packages.get("ultralytics"):
            enabled = False
            reason = "Ultralytics is not installed"
            missing_dependency = "ultralytics"
        if enabled and trainer in {"classification_trainer", "segmentation_trainer", "detection_torchvision_trainer", "anomaly_trainer"} and not packages.get("torchvision"):
            enabled = False
            reason = "TorchVision is not installed"
            missing_dependency = "torchvision"
        if enabled and device_mode == "gpu":
            if not model.get("supports_gpu", False):
                enabled = False
                reason = "This model does not support GPU mode in this registry"
            elif not cuda_ok:
                enabled = False
                reason = "GPU mode selected but CUDA is not available"
        if enabled and device_mode == "cpu" and not model.get("supports_cpu", False):
            enabled = False
            reason = "Requires CUDA GPU"
        min_ram = float(model.get("min_ram_gb", 0) or 0)
        rec_ram = float(model.get("recommended_ram_gb", min_ram) or min_ram)
        min_vram = float(model.get("min_vram_gb", 0) or 0)
        rec_vram = float(model.get("recommended_vram_gb", min_vram) or min_vram)
        if enabled and available_ram < min_ram:
            enabled = False
            reason = f"Requires at least {min_ram:g} GB RAM, detected {available_ram:g} GB"
        elif available_ram < rec_ram:
            warnings.append(f"Recommended RAM {rec_ram:g} GB, detected {available_ram:g} GB")
        if enabled and device_mode == "gpu":
            if vram < min_vram:
                enabled = False
                reason = f"Requires at least {min_vram:g} GB VRAM, detected {vram:g} GB"
            elif vram < rec_vram:
                warnings.append(f"Recommended VRAM {rec_vram:g} GB, detected {vram:g} GB")
        if enabled and dataset_format and model.get("dataset_format") != dataset_format:
            enabled = False
            reason = f"Dataset format mismatch: requires {model.get('dataset_format')}, current {dataset_format}"
        recommended_batch = model.get("recommended_batch_gpu") if device_mode == "gpu" else model.get("recommended_batch_cpu")
        recommended_precision = "fp16" if device_mode == "gpu" and model.get("supports_fp16") and gpu.get("fp16_supported") else "fp32"
        models.append({
            **model,
            "enabled": bool(enabled),
            "reason": reason,
            "warnings": warnings,
            "recommended_batch_size": recommended_batch,
            "recommended_image_size": model.get("recommended_image_size"),
            "recommended_precision": recommended_precision,
            "expected_device": "cuda:0" if device_mode == "gpu" else "cpu",
            "missing_dependency": missing_dependency,
        })
    return {"device_mode": device_mode, "dataset_format": dataset_format, "hardware_profile": profile, "models": models}
