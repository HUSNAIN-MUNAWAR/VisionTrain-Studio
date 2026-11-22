# VisionTrain preflight animation enhancement

## Purpose

Make the product’s safety boundary visible: local image data is validated, hardware and packages are profiled, the selected model is checked against the dataset, and only then can a real trainer be launched.

## Real service boundary

The generator executes:

- `backend/trainer/dataset_validator.py::validate_dataset`
- `backend/trainer/model_compatibility.py::evaluate_model_compatibility`
- `backend/trainer/config_manager.py::normalize_config` and the real model registry
- Pillow image inspection over the committed synthetic classification fixture

It never fabricates training loss, accuracy, a checkpoint, or export output. If PyTorch/TorchVision is missing, the animation shows the exact model gate reason.

## Reproduce

```bash
python scripts/generate_training_animation.py
```

Outputs:

- `docs/animation/visiontrain_training_sequence.json` — source checksum, image validation, image statistics, model registry evidence, and scene events.
- `frontend/animation/visiontrain-training-reel.html` — dependency-free animated reference reel with pause, replay, and stage selection.
