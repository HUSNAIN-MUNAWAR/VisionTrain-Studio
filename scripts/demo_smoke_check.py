from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEMO_DATASET = ROOT / "demo" / "classification_dataset"
DEMO_CONFIG = ROOT / "backend" / "configs" / "saved" / "demo_classification_config.json"


def build_demo_config() -> Dict[str, Any]:
    from backend.trainer.config_manager import normalize_config

    return normalize_config(
        {
            "task": "image_classification",
            "model": {
                "id": "resnet18",
                "name": "ResNet18",
                "family": "torch",
                "weights": "torchvision",
                "trainer": "classification_trainer",
                "dataset_format": "classification_folder",
            },
            "dataset": {
                "path": str(DEMO_DATASET),
                "format": "classification_folder",
                "yaml": "",
            },
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "image_size": 64,
                "learning_rate": 0.001,
                "workers": 0,
                "precision": "fp32",
                "checkpoint_interval": 1,
            },
            "output": {
                "experiment_name": "demo_classification_smoke",
                "output_dir": "backend/runs",
            },
        }
    )


def write_demo_config(config: Dict[str, Any]) -> None:
    DEMO_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    DEMO_CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    from backend.trainer.train_router import TrainingRouter

    router = TrainingRouter()
    result = router.start(config)
    while router.is_running():
        time.sleep(0.25)
    status = router.get_status()
    return {"start": result, "status": status}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight VisionTrain demo smoke check.")
    parser.add_argument("--train", action="store_true", help="Attempt a one-epoch demo training run when dependencies are installed.")
    args = parser.parse_args()

    from backend.trainer.dataset_validator import validate_dataset
    from backend.trainer.model_compatibility import evaluate_model_compatibility

    config = build_demo_config()
    validation = validate_dataset(config)
    write_demo_config(config)

    print(f"Demo dataset: {DEMO_DATASET}")
    print(f"Demo config:  {DEMO_CONFIG}")
    print(f"Dataset valid: {validation.get('valid')}")
    print(f"Images: train={validation.get('train_image_count')} val={validation.get('val_image_count')} classes={validation.get('class_count')}")

    if not validation.get("valid"):
        print("Dataset validation errors:")
        for error in validation.get("errors", []):
            print(f"- {error}")
        return 1

    compatibility = evaluate_model_compatibility(device_mode="cpu", dataset_format="classification_folder")
    resnet = next((model for model in compatibility.get("models", []) if model.get("id") == "resnet18"), None)
    if not resnet:
        print("ResNet18 was not found in the model registry.")
        return 1

    print(f"ResNet18 enabled: {resnet.get('enabled')} ({resnet.get('reason')})")
    if resnet.get("missing_dependency"):
        print(f"Missing dependency for demo training: {resnet.get('missing_dependency')}")

    if not args.train:
        print("Smoke check passed. Add --train after setup to run the one-epoch training proof.")
        return 0

    if not resnet.get("enabled"):
        print("Training skipped because the current environment is not train-ready.")
        return 2

    result = run_training(config)
    status = result["status"]
    print(json.dumps(status, indent=2, default=str))
    return 0 if status.get("status") in {"completed", "stopped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
