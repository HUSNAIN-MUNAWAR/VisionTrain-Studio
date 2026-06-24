from __future__ import annotations
from typing import Any, Dict

def _gb(value: int | float) -> float:
    return round(float(value) / (1024 ** 3), 2)

def get_gpu_monitor(selected_gpu_id: int = 0, active_training: bool = False, device_mode: str = "cpu") -> Dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"cuda_available": False, "gpu_count": 0, "gpu_name": "torch not installed", "selected_gpu_id": selected_gpu_id, "vram_total_gb": 0, "vram_allocated_gb": 0, "vram_reserved_gb": 0, "cuda_version": None, "pytorch_version": None, "fp16_supported": False, "training_device": device_mode, "warning": f"PyTorch import failed: {exc}"}
    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    valid_id = selected_gpu_id if available and selected_gpu_id < count else 0
    name = torch.cuda.get_device_name(valid_id) if available else "No CUDA GPU detected"
    total = torch.cuda.get_device_properties(valid_id).total_memory if available else 0
    allocated = torch.cuda.memory_allocated(valid_id) if available else 0
    reserved = torch.cuda.memory_reserved(valid_id) if available else 0
    fp16 = False
    if available:
        major, _minor = torch.cuda.get_device_capability(valid_id)
        fp16 = major >= 7
    warning = "GPU mode selected but CUDA is not available. Use CPU mode or reinstall CUDA PyTorch." if device_mode == "gpu" and not available else "ok"
    return {
        "cuda_available": available, "gpu_count": count, "gpu_name": name, "selected_gpu_id": selected_gpu_id,
        "vram_total_gb": _gb(total), "vram_allocated_gb": _gb(allocated), "vram_reserved_gb": _gb(reserved),
        "cuda_version": torch.version.cuda, "pytorch_version": torch.__version__, "fp16_supported": fp16,
        "active_training_process": active_training, "training_device": "cuda" if device_mode == "gpu" and available else "cpu",
        "warning": warning,
    }
