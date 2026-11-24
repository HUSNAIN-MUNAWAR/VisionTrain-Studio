from __future__ import annotations
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
INDEX_PATH = RUNS_DIR / "experiments.json"

def _read_index() -> List[Dict[str, Any]]:
    if not INDEX_PATH.exists():
        return []
    try:
        data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Experiment index cannot be read safely: {INDEX_PATH}") from exc
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise RuntimeError(f"Experiment index must contain a JSON list of records: {INDEX_PATH}")
    return data


def _write_index(data: List[Dict[str, Any]]) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = INDEX_PATH.with_name(f".{INDEX_PATH.name}.{uuid.uuid4().hex}.tmp")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(temporary, flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(data, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, INDEX_PATH)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

def list_experiments(limit: int = 50) -> List[Dict[str, Any]]:
    return list(reversed(_read_index()))[:limit]

def record_experiment(config: Dict[str, Any], status: str, metrics: Dict[str, Any] | None = None, error: str | None = None) -> Dict[str, Any]:
    model, output = config.get("model", {}), config.get("output", {})
    entry = {
        "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "experiment_name": output.get("experiment_name"),
        "project_name": output.get("project_name"),
        "model_id": model.get("id"),
        "model_name": model.get("name"),
        "task": config.get("task"),
        "dataset_format": config.get("dataset", {}).get("format"),
        "device": config.get("device", {}).get("mode"),
        "status": status,
        "metrics": metrics or {},
        "error": error,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    data = _read_index()
    data.append(entry)
    _write_index(data)
    return entry
