"""Run VisionTrain's real classifier and render its measured evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "demo" / "classification_dataset"
OUTPUT_DIR = ROOT / "docs" / "animation" / "runtime-evidence"
RUN_DIR = ROOT / "backend" / "runs" / "real_runtime"
VIDEO_PATH = OUTPUT_DIR / "visiontrain-runtime.mp4"
POSTER_PATH = OUTPUT_DIR / "visiontrain-runtime-poster.png"
SUMMARY_PATH = OUTPUT_DIR / "visiontrain-runtime-summary.json"
PREDICTIONS_PATH = OUTPUT_DIR / "visiontrain-predictions.csv"
DATASET_FILES_PATH = OUTPUT_DIR / "visiontrain-dataset-files.csv"
LOG_PATH = OUTPUT_DIR / "visiontrain-training.log"

sys.path.insert(0, str(ROOT))

from backend.trainer.config_manager import normalize_config  # noqa: E402
from backend.trainer.dataset_validator import validate_dataset  # noqa: E402
from backend.trainer.log_streamer import LogStreamer  # noqa: E402
from backend.trainer.model_compatibility import evaluate_model_compatibility  # noqa: E402
from backend.trainer.train_router import TrainingRouter  # noqa: E402


def sha256_file(path: Path) -> str:
    """Hash one local artifact in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    """Make the CPU demo run repeatable where the local libraries allow it."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def demo_config() -> dict[str, Any]:
    """Build the same normalized config used by the repository smoke check."""
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
            "dataset": {"path": str(DATASET_ROOT), "format": "classification_folder"},
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "image_size": 64,
                "learning_rate": 0.001,
                "workers": 0,
                "precision": "fp32",
                "checkpoint_interval": 1,
                "seed": 42,
            },
            "output": {"experiment_name": "visiontrain_real_runtime", "output_dir": str(RUN_DIR)},
        }
    )


def dataset_inventory() -> list[dict[str, Any]]:
    """Record every committed demo image and its digest."""
    rows = []
    for split in ("train", "val"):
        for image_path in sorted((DATASET_ROOT / split).rglob("*.png")):
            rows.append(
                {
                    "split": split,
                    "class_name": image_path.parent.name,
                    "path": image_path.relative_to(ROOT).as_posix(),
                    "bytes": image_path.stat().st_size,
                    "sha256": sha256_file(image_path),
                }
            )
    return rows


def train_and_infer() -> dict[str, Any]:
    """Run the app router, load its checkpoint, and measure validation predictions."""
    set_seed(42)
    config = demo_config()
    validation = validate_dataset(config)
    compatibility = evaluate_model_compatibility(device_mode="cpu", dataset_format="classification_folder")
    selected = next(model for model in compatibility["models"] if model["id"] == "resnet18")
    logger = LogStreamer(log_dir=OUTPUT_DIR / "_router_log")
    router = TrainingRouter(logger=logger)
    start_response = router.start(config)
    while router.is_running():
        time.sleep(0.1)
    status = router.get_status()
    if status.get("status") != "completed":
        raise RuntimeError(f"Real VisionTrain training did not complete: {status}")

    checkpoint_path = RUN_DIR / "visiontrain_real_runtime" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Training reported completion but checkpoint is missing: {checkpoint_path}")

    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = list(checkpoint["classes"])
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    image_size = int(config["training"]["image_size"])
    val_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    val_dataset = datasets.ImageFolder(DATASET_ROOT / "val", transform=val_transform)
    predictions: list[dict[str, Any]] = []
    previews: list[np.ndarray] = []
    with torch.no_grad():
        for index in range(len(val_dataset)):
            image_tensor, target = val_dataset[index]
            logits = model(image_tensor.unsqueeze(0).to(device))
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted = int(probabilities.argmax().item())
            image_path = Path(val_dataset.samples[index][0])
            predictions.append(
                {
                    "path": image_path.relative_to(ROOT).as_posix(),
                    "ground_truth": classes[int(target)],
                    "predicted": classes[predicted],
                    "confidence": float(probabilities[predicted].item()),
                    "correct": bool(predicted == int(target)),
                    "logits": [float(value) for value in logits[0].cpu().tolist()],
                }
            )
            raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if raw is None:
                raise RuntimeError(f"Could not load validation preview: {image_path}")
            previews.append(cv2.resize(raw, (180, 140), interpolation=cv2.INTER_NEAREST))

    correct = sum(1 for row in predictions if row["correct"])
    return {
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "python_version": sys.version.split()[0],
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "config": config,
        "dataset_validation": validation,
        "compatibility": {key: selected.get(key) for key in ("id", "name", "enabled", "reason", "missing_dependency", "expected_device", "recommended_precision")},
        "start_response": start_response,
        "training_status": status,
        "checkpoint": {
            "path": checkpoint_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "epoch": checkpoint.get("epoch"),
            "classes": classes,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "runtime_transform": {
            "train": [
                "Resize(64,64)",
                "RandomHorizontalFlip()",
                "ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)",
                "ToTensor()",
                "Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])",
            ],
            "validation": [
                "Resize(64,64)",
                "ToTensor()",
                "Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])",
            ],
        },
        "validation_inference": {
            "count": len(predictions),
            "correct": correct,
            "accuracy": correct / max(1, len(predictions)),
            "predictions": predictions,
        },
        "_previews": previews,
        "_log_file": logger.log_file,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write a reviewable CSV without discarding any measured field."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def put_lines(canvas: np.ndarray, lines: list[str], origin: tuple[int, int], color: tuple[int, int, int], scale: float = 0.58) -> None:
    """Draw ASCII-only lines for reliable video text rendering."""
    x, y = origin
    for line in lines:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += 27


def render_frame(evidence: dict[str, Any], stage: str, index: int, total: int) -> np.ndarray:
    """Render a stage from measured evidence, including real validation samples."""
    canvas = np.full((720, 1280, 3), (10, 19, 28), dtype=np.uint8)
    white, muted = (238, 245, 242), (157, 178, 176)
    accent, green, amber = (113, 224, 189), (127, 222, 153), (80, 190, 255)
    cv2.putText(canvas, "VISIONTRAIN STUDIO | REAL LOCAL CLASSIFIER", (52, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.95, accent, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Measured ResNet18 checkpoint | committed bright/dark images | CPU only", (53, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.56, muted, 1, cv2.LINE_AA)
    cv2.line(canvas, (52, 112), (1228, 112), (45, 72, 73), 1)
    cv2.rectangle(canvas, (52, 140), (824, 652), (17, 31, 39), -1)
    cv2.rectangle(canvas, (850, 140), (1228, 652), (17, 31, 39), -1)
    cv2.putText(canvas, f"{index + 1:02d}/{total:02d}  {stage.upper()}", (82, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.76, green, 2, cv2.LINE_AA)

    validation = evidence["dataset_validation"]
    status = evidence["training_status"]
    checkpoint = evidence["checkpoint"]
    inference = evidence["validation_inference"]
    if stage == "dataset":
        put_lines(canvas, ["actual dataset validator", f"valid: {validation['valid']}", f"train images: {validation['train_image_count']}", f"val images: {validation['val_image_count']}", f"classes: {validation['class_count']}", f"errors: {len(validation['errors'])}"], (86, 270), white)
    elif stage == "model gate":
        put_lines(canvas, ["registry + compatibility service", f"model: {evidence['compatibility']['name']}", f"enabled: {evidence['compatibility']['enabled']}", f"reason: {evidence['compatibility']['reason']}", f"device: {evidence['compatibility']['expected_device']}", f"precision: {evidence['compatibility']['recommended_precision']}"], (86, 270), white)
    elif stage == "training":
        put_lines(canvas, ["TrainingRouter.start()", "ClassificationTrainer.run()", f"status: {status['status']}", f"epoch: {status['metrics'].get('epoch')}", "one epoch / batch size 2", "seed: 42"], (86, 270), white)
    elif stage == "metrics":
        metrics = status["metrics"]
        put_lines(canvas, ["actual epoch result", f"val loss: {metrics.get('loss')}", f"val accuracy: {metrics.get('accuracy')}", f"best score: {metrics.get('score')}", f"checkpoint epoch: {checkpoint['epoch']}", "no fabricated metric"], (86, 270), white)
    elif stage == "checkpoint":
        put_lines(canvas, ["checkpoint loaded and verified", f"classes: {checkpoint['classes']}", f"parameters: {checkpoint['parameter_count']:,}", f"bytes: {checkpoint['bytes']:,}", f"sha256: {checkpoint['sha256'][:24]}...", "state_dict load: passed"], (86, 270), white)
    elif stage == "validation inference":
        for image_index, (preview, prediction) in enumerate(zip(evidence["_previews"], inference["predictions"])):
            x = 86 + (image_index % 3) * 234
            y = 240 + (image_index // 3) * 190
            canvas[y : y + 140, x : x + 180] = preview
            border = green if prediction["correct"] else (70, 75, 225)
            cv2.rectangle(canvas, (x, y), (x + 180, y + 140), border, 2)
            cv2.putText(canvas, f"gt {prediction['ground_truth']} -> {prediction['predicted']}", (x, y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.43, white, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"conf {prediction['confidence']:.3f}", (x, y + 178), cv2.FONT_HERSHEY_SIMPLEX, 0.43, muted, 1, cv2.LINE_AA)
    else:
        put_lines(canvas, ["reviewable evidence written", "MP4 + poster + predictions CSV", "dataset inventory CSV + log + JSON", f"val samples: {inference['count']}", f"correct: {inference['correct']}", "production boundary remains explicit"], (86, 270), white)

    cv2.putText(canvas, "RUNTIME FACTS", (878, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.7, amber, 2, cv2.LINE_AA)
    put_lines(canvas, [f"torch {evidence['torch_version']}", f"torchvision {evidence['torchvision_version']}", f"device {evidence['device']}", f"CUDA {evidence['cuda_available']}", "model weights: trained locally", "data: committed demo only"], (878, 238), muted, 0.56)
    cv2.line(canvas, (878, 418), (1198, 418), (45, 72, 73), 1)
    put_lines(canvas, ["HONEST BOUNDARY", "One epoch on 10", "bright/dark demo images.", "Validation score is not", "real-world accuracy."], (878, 464), green, 0.56)
    cv2.putText(canvas, "No fabricated detections or metrics", (53, 695), cv2.FONT_HERSHEY_SIMPLEX, 0.54, muted, 1, cv2.LINE_AA)
    return canvas


def write_outputs(evidence: dict[str, Any]) -> None:
    """Persist actual trace files and render the measured animation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = dataset_inventory()
    write_csv(DATASET_FILES_PATH, files, ["split", "class_name", "path", "bytes", "sha256"])
    prediction_rows = [{key: value for key, value in row.items() if key != "logits"} for row in evidence["validation_inference"]["predictions"]]
    write_csv(PREDICTIONS_PATH, prediction_rows, ["path", "ground_truth", "predicted", "confidence", "correct"])

    log_file = Path(evidence.pop("_log_file"))
    if log_file.exists():
        LOG_PATH.write_text(log_file.read_text(encoding="utf-8"), encoding="utf-8")
        log_file.unlink()
        log_file.parent.rmdir()
    evidence["_log_file"] = None

    stages = ["environment", "dataset", "model gate", "training", "metrics", "checkpoint", "validation inference", "artifacts"]
    writer = cv2.VideoWriter(str(VIDEO_PATH), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {VIDEO_PATH}")
    try:
        for index, stage in enumerate(stages):
            frame = render_frame(evidence, stage, index, len(stages))
            if stage == "validation inference":
                cv2.imwrite(str(POSTER_PATH), frame)
            for _ in range(32):
                writer.write(frame)
    finally:
        writer.release()

    evidence.pop("_previews", None)
    evidence["artifacts"] = {
        "video": {"path": VIDEO_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(VIDEO_PATH), "frames": len(stages) * 32, "fps": 20},
        "poster": {"path": POSTER_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(POSTER_PATH)},
        "predictions_csv": {"path": PREDICTIONS_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(PREDICTIONS_PATH)},
        "dataset_files_csv": {"path": DATASET_FILES_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(DATASET_FILES_PATH)},
        "training_log": {"path": LOG_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(LOG_PATH) if LOG_PATH.exists() else None},
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def main() -> None:
    """Execute the real model path and generate evidence."""
    evidence = train_and_infer()
    write_outputs(evidence)
    print(json.dumps({"summary": SUMMARY_PATH.as_posix(), "video": VIDEO_PATH.as_posix(), "checkpoint": evidence["checkpoint"]["sha256"], "validation": evidence["validation_inference"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
