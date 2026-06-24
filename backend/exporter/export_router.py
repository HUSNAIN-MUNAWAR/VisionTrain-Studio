from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List

from backend.trainer.config_manager import build_export_command
from backend.trainer.log_streamer import LogStreamer
from backend.trainer.hardware_profiler import get_hardware_profile

from .export_validator import validate_export_request
from .onnx_exporter import export_pytorch_to_onnx, export_ultralytics
from .precision_converter import convert_pytorch_precision
from .tensorrt_exporter import export_to_tensorrt

logger = LogStreamer(log_dir="backend/logs", max_lines=2000)
_status_lock = Lock()
_status: Dict[str, Any] = {"status": "idle", "error": None, "output": None, "command": None}


def _set_status(**kwargs: Any) -> None:
    with _status_lock:
        _status.update(kwargs)


def get_export_status() -> Dict[str, Any]:
    with _status_lock:
        return dict(_status)


def get_export_logs(limit: int = 300) -> Dict[str, List[str]]:
    logs = logger.read(limit=limit)
    if not logs:
        logs = ["No active export process."]
    return {"logs": logs}


def get_export_capabilities() -> Dict[str, Any]:
    hw = get_hardware_profile()
    return {
        "onnx": {"enabled": True, "reason": "PyTorch ONNX export is available when torch is installed; validation requires onnx package."},
        "ultralytics_onnx": {"enabled": bool(hw["packages"].get("ultralytics")), "reason": "Requires ultralytics package."},
        "torchscript": {"enabled": True, "reason": "PyTorch TorchScript export is available for compatible PyTorch models."},
        "openvino": {"enabled": bool(hw["packages"].get("ultralytics")), "reason": "Available through Ultralytics export when local package supports it."},
        "tensorrt": {"enabled": bool(hw["gpu"].get("tensorrt_available") and hw["gpu"].get("cuda_available")), "reason": "Requires TensorRT, CUDA, and compatible NVIDIA GPU."},
        "onnxruntime_gpu": {"enabled": bool(hw["gpu"].get("onnxruntime_gpu_provider_available")), "reason": "Requires onnxruntime-gpu CUDAExecutionProvider."},
        "fp16": {"enabled": bool(hw["gpu"].get("fp16_supported")), "reason": "FP16 runtime acceleration requires CUDA GPU with FP16 support."},
        "int8": {"enabled": False, "reason": "Future integration. INT8 needs calibration dataset logic."},
        "hardware_profile": hw,
    }


def run_export(payload: Dict[str, Any]) -> Dict[str, Any]:
    logger.reset()
    command = build_export_command(payload)
    _set_status(status="running", error=None, output=None, command=command)
    logger.write(f"Export command: {command}")
    try:
        validation = validate_export_request(payload)
        for warning in validation.get("warnings", []):
            logger.write(f"Warning: {warning}")
        if not validation.get("valid"):
            raise RuntimeError("; ".join(validation.get("errors", [])))
        fmt = str(payload.get("format", "onnx")).lower()
        family = str(payload.get("model_family", payload.get("family", "torch"))).lower()
        if fmt == "onnx":
            result = export_ultralytics(payload, logger.write) if family == "ultralytics" else export_pytorch_to_onnx(payload, logger.write)
        elif fmt == "torchscript":
            if family == "ultralytics":
                result = export_ultralytics({**payload, "format": "torchscript"}, logger.write)
            else:
                result = _export_torchscript(payload, logger.write)
        elif fmt == "openvino":
            if family != "ultralytics":
                raise RuntimeError("OpenVINO export is only exposed through Ultralytics in this local build.")
            result = export_ultralytics({**payload, "format": "openvino"}, logger.write)
        elif fmt in {"tensorrt", "engine"}:
            result = export_to_tensorrt(payload, logger.write)
        else:
            raise RuntimeError(f"Unsupported export format: {fmt}")
        _set_status(status="completed", error=None, output=result.get("output"), command=command)
        logger.write(f"Export completed: {result.get('output')}")
        return {**result, "command": command}
    except Exception as exc:
        trace = traceback.format_exc()
        log_path = Path("backend/logs/export_error_traceback.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(trace, encoding="utf-8")
        _set_status(status="failed", error=str(exc), output=None, command=command, traceback_path=str(log_path))
        logger.write(f"Export failed: {exc}")
        logger.write(f"Traceback saved: {log_path}")
        return {"status": "failed", "error": str(exc), "traceback_path": str(log_path), "suggested_fix": suggest_export_fix(str(exc)), "command": command}


def _export_torchscript(payload: Dict[str, Any], log: Callable[[str], None]) -> Dict[str, Any]:
    raise RuntimeError("Generic TorchScript export for arbitrary PyTorch checkpoints is future work unless a model rebuild config is available. Use ONNX for classification/anomaly or Ultralytics TorchScript export for YOLO.")


def convert_precision(payload: Dict[str, Any]) -> Dict[str, Any]:
    logger.reset()
    try:
        result = convert_pytorch_precision(payload, logger.write)
        _set_status(status="completed", error=None, output=result.get("output"), command=build_export_command(payload))
        return result
    except Exception as exc:
        trace = traceback.format_exc()
        Path("backend/logs/export_error_traceback.log").write_text(trace, encoding="utf-8")
        _set_status(status="failed", error=str(exc))
        logger.write(f"Precision conversion failed: {exc}")
        return {"status": "failed", "error": str(exc), "suggested_fix": suggest_export_fix(str(exc))}


def suggest_export_fix(error: str) -> str:
    e = error.lower()
    if "tensorrt" in e:
        return "Install NVIDIA TensorRT and a CUDA-compatible PyTorch build, then run scripts/check_gpu.bat."
    if "ultralytics" in e:
        return "Run scripts/setup_windows.bat and ensure ultralytics is installed in the active environment."
    if "checkpoint not found" in e:
        return "Select an existing .pt/.pth checkpoint from backend/runs or enter an absolute checkpoint path."
    if "onnx" in e:
        return "Install onnx for validation: pip install onnx."
    return "Check backend/logs/export_error_traceback.log for the full traceback."


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionTrain Studio export router")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--format", default="onnx")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--model-family", default="torch")
    args = parser.parse_args()
    print(run_export({"checkpoint": args.checkpoint, "format": args.format, "precision": args.precision, "image_size": args.imgsz, "opset": args.opset, "model_family": args.model_family}))


if __name__ == "__main__":
    main()
