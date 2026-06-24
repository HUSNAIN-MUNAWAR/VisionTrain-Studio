from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict


def convert_pytorch_precision(payload: Dict[str, Any], log: Callable[[str], None]) -> Dict[str, Any]:
    import torch

    checkpoint = Path(payload["checkpoint"]).expanduser()
    precision = str(payload.get("precision", "fp16")).lower()
    if precision != "fp16":
        raise RuntimeError("Only FP16 PyTorch checkpoint conversion is implemented. FP32 is already the default storage format.")
    data = torch.load(checkpoint, map_location="cpu")
    if not isinstance(data, dict):
        raise RuntimeError("Checkpoint must be a dictionary containing model_state/state_dict for safe precision conversion.")
    key = "model_state" if "model_state" in data else "state_dict" if "state_dict" in data else None
    if not key:
        raise RuntimeError("Could not find model_state or state_dict inside checkpoint.")
    converted = {}
    for k, v in data[key].items():
        converted[k] = v.half() if hasattr(v, "half") and getattr(v, "is_floating_point", lambda: False)() else v
    data[key] = converted
    out_dir = Path(payload.get("output_dir") or "backend/runs/exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{checkpoint.stem}_fp16.pt"
    torch.save(data, output)
    log(f"FP16 checkpoint written: {output}")
    return {"status": "completed", "output": str(output), "warning": "Use FP16 checkpoints only with compatible model/device code."}
