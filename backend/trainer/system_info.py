from __future__ import annotations
import platform, sys
from typing import Any, Dict
import psutil

def _bytes_to_gb(value: int | float) -> float:
    return round(float(value) / (1024 ** 3), 2)

def get_system_info() -> Dict[str, Any]:
    mem = psutil.virtual_memory()
    return {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu_name": platform.processor() or platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_usage_percent": psutil.cpu_percent(interval=0.1),
        "ram_total_gb": _bytes_to_gb(mem.total),
        "ram_used_gb": _bytes_to_gb(mem.used),
        "ram_percent": mem.percent,
    }
