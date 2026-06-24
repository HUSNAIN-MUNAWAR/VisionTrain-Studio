from pathlib import Path

from backend.trainer.dataset_validator import validate_dataset


def test_demo_classification_dataset_validates() -> None:
    root = Path(__file__).resolve().parents[1]
    result = validate_dataset(
        {
            "task": "image_classification",
            "dataset": {
                "path": str(root / "demo" / "classification_dataset"),
                "format": "classification_folder",
            },
        }
    )

    assert result["valid"] is True
    assert result["dataset_format"] == "classification_folder"
    assert result["train_image_count"] == 6
    assert result["val_image_count"] == 4
    assert result["class_count"] == 2


def test_empty_dataset_path_reports_actionable_error() -> None:
    result = validate_dataset({"dataset": {"path": "", "format": "classification_folder"}})

    assert result["valid"] is False
    assert "Dataset path is empty." in result["errors"]
    assert result["recommended_fixes"]
