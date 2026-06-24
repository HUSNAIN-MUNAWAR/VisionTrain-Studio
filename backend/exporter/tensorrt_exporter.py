from __future__ import annotations

from typing import Any, Callable, Dict

from .export_validator import validate_export_request
from .onnx_exporter import export_ultralytics


def export_to_tensorrt(payload: Dict[str, Any], log: Callable[[str], None]) -> Dict[str, Any]:
    validation = validate_export_request({**payload, "format": "tensorrt"})
    if not validation["valid"]:
        raise RuntimeError("; ".join(validation["errors"]))
    family = str(payload.get("model_family", payload.get("family", "ultralytics"))).lower()
    if family == "ultralytics":
        return export_ultralytics({**payload, "format": "engine"}, log)
    raise RuntimeError("Generic ONNX-to-TensorRT engine building is not bundled. Use Ultralytics TensorRT export or install TensorRT tooling and extend tensorrt_exporter.py.")
