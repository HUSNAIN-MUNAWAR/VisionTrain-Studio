from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from backend.trainer.hardware_profiler import get_hardware_profile


def validate_export_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint = Path(payload.get("checkpoint", "")).expanduser()
    fmt = str(payload.get("format", "onnx")).lower()
    family = str(payload.get("model_family", payload.get("family", "torch"))).lower()
    precision = str(payload.get("precision", "fp32")).lower()
    errors, warnings = [], []
    if not checkpoint.exists():
        errors.append(f"Checkpoint not found: {checkpoint}")
    if fmt not in {"onnx", "torchscript", "openvino", "tensorrt", "engine"}:
        errors.append(f"Unsupported export format: {fmt}")
    if precision not in {"fp32", "fp16"}:
        errors.append("Only fp32 and fp16 are supported. INT8 is future integration until calibration is implemented.")
    hw = get_hardware_profile()
    if fmt in {"tensorrt", "engine"}:
        if not hw["gpu"].get("tensorrt_available"):
            errors.append("TensorRT not detected. Install NVIDIA TensorRT and CUDA-compatible PyTorch to enable engine export.")
        if not hw["gpu"].get("cuda_available"):
            errors.append("TensorRT engine export requires CUDA GPU availability.")
    if precision == "fp16" and not hw["gpu"].get("fp16_supported") and fmt in {"tensorrt", "engine"}:
        errors.append("FP16 TensorRT export requires a CUDA GPU with FP16 support.")
    if fmt == "onnx" and family != "ultralytics" and not hw["packages"].get("onnx"):
        warnings.append("onnx package is not installed; export can still write a file, but validation will be skipped.")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "hardware_profile": hw}
