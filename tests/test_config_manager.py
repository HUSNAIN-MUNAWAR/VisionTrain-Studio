from pathlib import Path

from backend.trainer.config_manager import build_training_command, normalize_config


def test_normalize_config_merges_defaults() -> None:
    dataset_path = str(Path("demo") / "classification_dataset")
    config = normalize_config(
        {
            "task": "image_classification",
            "dataset": {"path": dataset_path, "format": "classification_folder"},
            "training": {"epochs": 1, "workers": 0},
        }
    )

    assert config["dataset"]["path"] == dataset_path
    assert config["dataset"]["format"] == "classification_folder"
    assert config["training"]["epochs"] == 1
    assert config["training"]["workers"] == 0
    assert "batch_size" in config["training"]
    assert "device" in config


def test_training_command_points_to_saved_config() -> None:
    assert build_training_command({}) == "python -m backend.trainer.train_router --config backend/configs/saved/current_config.json"
