from __future__ import annotations
import platform, sys
from typing import Any, Dict
import psutil

def _gb(value: int | float) -> float:
    return round(float(value) / (1024 ** 3), 2)

def safe_batch_recommendation(ram_total_gb: float, image_size: int = 640) -> int:
    if ram_total_gb < 8: return 1
    if image_size >= 1024: return 1 if ram_total_gb < 16 else 2
    if ram_total_gb < 16: return 2
    if ram_total_gb < 32: return 4
    return 8

def estimated_workers(logical_cores: int | None) -> int:
    return 0 if not logical_cores else max(0, min(8, logical_cores // 2))

def get_cpu_monitor(active_training: bool = False, device_mode: str = "cpu", batch_size: int = 8, image_size: int = 640) -> Dict[str, Any]:
    mem = psutil.virtual_memory()
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    ram_gb = _gb(mem.total)
    safe_batch = safe_batch_recommendation(ram_gb, image_size)
    return {
        "cpu_name": platform.processor() or platform.machine(),
        "physical_cores": physical,
        "logical_cores": logical,
        "cpu_usage_percent": psutil.cpu_percent(interval=0.15),
        "ram_total_gb": ram_gb,
        "ram_used_gb": _gb(mem.used),
        "ram_percent": mem.percent,
        "python_version": sys.version.split()[0],
        "active_training_process": active_training,
        "current_device_mode": device_mode,
        "safe_batch_recommendation": safe_batch,
        "estimated_workers": estimated_workers(logical),
        "batch_warning": batch_size > safe_batch,
        "safe_mode_status": "safe" if batch_size <= safe_batch else "reduce batch size",
    }
