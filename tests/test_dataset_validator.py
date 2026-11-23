import hashlib
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


def test_demo_train_and_validation_files_are_byte_disjoint() -> None:
    root = Path(__file__).resolve().parents[1] / "demo" / "classification_dataset"
    train_hashes = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root / "train").rglob("*.png")
    }
    overlap = [
        path.relative_to(root).as_posix()
        for path in (root / "val").rglob("*.png")
        if hashlib.sha256(path.read_bytes()).hexdigest() in train_hashes
    ]
    assert overlap == []
