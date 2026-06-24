from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.json"
MODELS_PATH = CONFIG_DIR / "models.json"
SAVED_CONFIG_DIR = CONFIG_DIR / "saved"

def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_default_config() -> Dict[str, Any]:
    return load_json(DEFAULT_CONFIG_PATH)

def load_models() -> Dict[str, Any]:
    return load_json(MODELS_PATH)

def get_model_by_id(model_id: str) -> Dict[str, Any] | None:
    for model in load_models().get("models", []):
        if model.get("id") == model_id:
            return model
    return None

def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    default = load_default_config()
    def merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(base)
        for key, value in (incoming or {}).items():
            out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
        return out
    return merge(default, config or {})

def save_config(config: Dict[str, Any]) -> Dict[str, str]:
    SAVED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    name = config.get("output", {}).get("experiment_name") or "visiontrain_config"
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    path = SAVED_CONFIG_DIR / f"{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    current_path = SAVED_CONFIG_DIR / "current_config.json"
    save_json(path, config)
    save_json(current_path, config)
    return {"path": str(path), "current_path": str(current_path)}

def build_training_command(config: Dict[str, Any]) -> str:
    """Return the canonical launcher command generated from JSON config.

    The API starts training in-process via FastAPI, but the preview is a real
    equivalent command for reproducible CLI runs.
    """
    return "python -m backend.trainer.train_router --config backend/configs/saved/current_config.json"


def build_export_command(payload: Dict[str, Any]) -> str:
    checkpoint = payload.get("checkpoint") or "backend/runs/exp001/best.pt"
    fmt = payload.get("format", "onnx")
    precision = payload.get("precision", "fp32")
    imgsz = payload.get("image_size") or payload.get("imgsz") or 640
    opset = payload.get("opset", 17)
    return (
        f'python -m backend.exporter.export_router --checkpoint "{checkpoint}" '
        f"--format {fmt} --precision {precision} --imgsz {imgsz} --opset {opset}"
    )
