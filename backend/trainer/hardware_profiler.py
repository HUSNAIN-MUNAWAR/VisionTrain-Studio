from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import psutil


def _gb(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return round(float(value) / (1024 ** 3), 2)


def _cpu_name() -> str:
    name = platform.processor() or platform.machine() or "Unknown CPU"
    if sys.platform.startswith("win"):
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name", "/value"], stderr=subprocess.DEVNULL, text=True, timeout=4
            )
            for line in out.splitlines():
                if line.startswith("Name=") and line.split("=", 1)[1].strip():
                    return line.split("=", 1)[1].strip()
        except Exception:
            pass
    return name


def _package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def get_cpu_profile() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    physical = psutil.cpu_count(logical=False) or 1
    logical = psutil.cpu_count(logical=True) or physical
    ram_gb = _gb(vm.total)
    # Conservative defaults for local laptop/desktop training.
    if ram_gb >= 32:
        batch_cpu = {"classification": 32, "yolo": 4, "segmentation": 2, "detection": 1, "anomaly": 32}
    elif ram_gb >= 16:
        batch_cpu = {"classification": 16, "yolo": 2, "segmentation": 1, "detection": 1, "anomaly": 16}
    else:
        batch_cpu = {"classification": 8, "yolo": 1, "segmentation": 1, "detection": 1, "anomaly": 8}
    workers = max(0, min(8, logical - 1))
    return {
        "name": _cpu_name(),
        "physical_cores": physical,
        "logical_cores": logical,
        "usage_percent": psutil.cpu_percent(interval=0.2),
        "ram_total_gb": ram_gb,
        "ram_available_gb": _gb(vm.available),
        "ram_used_gb": _gb(vm.used),
        "ram_usage_percent": vm.percent,
        "recommended_batch_sizes": batch_cpu,
        "recommended_workers": workers,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def get_gpu_profile() -> Dict[str, Any]:
    gpu: Dict[str, Any] = {
        "cuda_available": False,
        "pytorch_cuda_available": False,
        "gpu_count": 0,
        "devices": [],
        "selected_device": None,
        "cuda_version": None,
        "cudnn_version": None,
        "fp16_supported": False,
        "bf16_supported": False,
        "tensorrt_available": _package_available("tensorrt"),
        "onnx_available": _package_available("onnx"),
        "onnxruntime_available": _package_available("onnxruntime"),
        "onnxruntime_gpu_provider_available": False,
        "torch_version": None,
    }
    try:
        import torch

        gpu["torch_version"] = torch.__version__
        gpu["cuda_version"] = getattr(torch.version, "cuda", None)
        try:
            gpu["cudnn_version"] = torch.backends.cudnn.version()
        except Exception:
            gpu["cudnn_version"] = None
        available = bool(torch.cuda.is_available())
        gpu["cuda_available"] = available
        gpu["pytorch_cuda_available"] = available
        if available:
            count = torch.cuda.device_count()
            gpu["gpu_count"] = count
            devices: List[Dict[str, Any]] = []
            for idx in range(count):
                props = torch.cuda.get_device_properties(idx)
                allocated = torch.cuda.memory_allocated(idx)
                reserved = torch.cuda.memory_reserved(idx)
                cap = f"{props.major}.{props.minor}"
                devices.append(
                    {
                        "id": idx,
                        "name": props.name,
                        "vram_total_gb": _gb(props.total_memory),
                        "vram_allocated_gb": _gb(allocated),
                        "vram_reserved_gb": _gb(reserved),
                        "compute_capability": cap,
                        "multiprocessors": getattr(props, "multi_processor_count", None),
                    }
                )
            gpu["devices"] = devices
            gpu["selected_device"] = devices[0] if devices else None
            major = int(devices[0]["compute_capability"].split(".")[0]) if devices else 0
            gpu["fp16_supported"] = major >= 6
            try:
                gpu["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            except Exception:
                gpu["bf16_supported"] = False
    except Exception as exc:
        gpu["torch_error"] = str(exc)

    if gpu["onnxruntime_available"]:
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            gpu["onnxruntime_providers"] = providers
            gpu["onnxruntime_gpu_provider_available"] = any("CUDA" in p or "Tensorrt" in p for p in providers)
        except Exception as exc:
            gpu["onnxruntime_error"] = str(exc)

    return gpu


def get_hardware_profile() -> Dict[str, Any]:
    cpu = get_cpu_profile()
    gpu = get_gpu_profile()
    readiness = []
    readiness.append({"name": "Python", "ok": True, "detail": cpu.get("python_version")})
    readiness.append({"name": "PyTorch", "ok": bool(gpu.get("torch_version")), "detail": gpu.get("torch_version") or "Not importable"})
    readiness.append({"name": "CUDA", "ok": bool(gpu.get("cuda_available")), "detail": gpu.get("cuda_version") or "CPU-only"})
    readiness.append({"name": "Ultralytics", "ok": _package_available("ultralytics"), "detail": "Required for YOLO / RT-DETR"})
    readiness.append({"name": "ONNX", "ok": bool(gpu.get("onnx_available")), "detail": "Required for ONNX validation"})
    readiness.append({"name": "TensorRT", "ok": bool(gpu.get("tensorrt_available")), "detail": "Required for .engine export"})
    score = round(100 * sum(1 for item in readiness if item["ok"]) / len(readiness))
    return {"cpu": cpu, "gpu": gpu, "packages": {"torch": _package_available("torch"), "ultralytics": _package_available("ultralytics"), "timm": _package_available("timm"), "torchvision": _package_available("torchvision"), "onnx": gpu.get("onnx_available"), "onnxruntime": gpu.get("onnxruntime_available"), "tensorrt": gpu.get("tensorrt_available")}, "readiness": readiness, "readiness_score": score}
