from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
INDEX_PATH = RUNS_DIR / "experiments.json"

def _read_index() -> List[Dict[str, Any]]:
    if not INDEX_PATH.exists():
        return []
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

def list_experiments(limit: int = 50) -> List[Dict[str, Any]]:
    return list(reversed(_read_index()))[:limit]

def record_experiment(config: Dict[str, Any], status: str, metrics: Dict[str, Any] | None = None, error: str | None = None) -> Dict[str, Any]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
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
    INDEX_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return entry
