from pathlib import Path
from threading import Event

from backend.trainer.yolo_trainer import YoloTrainer


def test_yolo_relative_yaml_resolves_inside_dataset_path() -> None:
    config = {
        "task": "object_detection",
        "dataset": {
            "path": str(Path("C:/datasets/visiontrain_demo")),
            "format": "yolo",
            "yaml": "dataset.yaml",
        },
        "output": {},
    }
    trainer = YoloTrainer(config, {"weights": "yolov8n.pt"}, lambda _message: None, Event())

    assert trainer._data_arg().replace("\\", "/").endswith("C:/datasets/visiontrain_demo/dataset.yaml")


def test_ultralytics_classification_uses_dataset_folder() -> None:
    config = {
        "task": "image_classification",
        "dataset": {
            "path": str(Path("C:/datasets/classification")),
            "format": "classification_folder",
            "yaml": "dataset.yaml",
        },
        "output": {},
    }
    trainer = YoloTrainer(config, {"weights": "yolov8n-cls.pt"}, lambda _message: None, Event())

    assert trainer._data_arg().replace("\\", "/").endswith("C:/datasets/classification")
