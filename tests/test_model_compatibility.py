from typing import Any, Dict

from backend.trainer import model_compatibility


def fake_profile() -> Dict[str, Any]:
    return {
        "cpu": {"ram_total_gb": 32},
        "gpu": {
            "cuda_available": False,
            "fp16_supported": False,
            "selected_device": None,
        },
        "packages": {
            "torch": True,
            "torchvision": False,
            "ultralytics": False,
            "onnx": True,
            "onnxruntime": False,
            "tensorrt": False,
        },
        "readiness": [],
        "readiness_score": 0,
    }


def test_missing_optional_training_packages_disable_matching_models(monkeypatch) -> None:
    monkeypatch.setattr(model_compatibility, "get_hardware_profile", fake_profile)

    data = model_compatibility.evaluate_model_compatibility(device_mode="cpu", dataset_format="classification_folder")
    by_id = {model["id"]: model for model in data["models"]}

    assert by_id["resnet18"]["enabled"] is False
    assert by_id["resnet18"]["reason"] == "TorchVision is not installed"
    assert by_id["resnet18"]["missing_dependency"] == "torchvision"

    assert by_id["yolov8n-det"]["enabled"] is False
    assert by_id["yolov8n-det"]["reason"] == "Ultralytics is not installed"
    assert by_id["yolov8n-det"]["missing_dependency"] == "ultralytics"
