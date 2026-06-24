# VisionTrain Studio Architecture

VisionTrain Studio is a local-first computer vision training control center. The frontend is static HTML/CSS/Vanilla JS, and the backend is FastAPI. Training and export operations run locally in Python.

```text
Browser UI
  | fetch /api/*
FastAPI backend
  |-- hardware profiler
  |-- dataset validator
  |-- model compatibility service
  |-- training router
  |     |-- Ultralytics trainer
  |     |-- PyTorch classification trainer
  |     |-- TorchVision semantic segmentation trainer
  |     |-- TorchVision COCO detection starter
  |     |-- Autoencoder anomaly trainer
  |-- exporter router
        |-- ONNX exporter
        |-- Ultralytics export wrapper
        |-- TensorRT gate / wrapper
        |-- Precision converter
```

## Design principles

- Supported means runnable: every supported model maps to a real trainer.
- Unsupported integrations stay visible but disabled.
- Hardware determines model and export availability.
- JSON config is the source of truth.
- No cloud services, API keys, or paid dependencies are required.

## Important runtime folders

```text
backend/runs/              training outputs and checkpoints
backend/runs/exports/      exported ONNX / engine / TorchScript outputs
backend/logs/              training, export, backend, frontend runtime logs
backend/configs/saved/     saved JSON configs
backend/datasets/          optional local datasets folder
```
