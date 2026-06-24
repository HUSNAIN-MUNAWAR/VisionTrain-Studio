# Export and Conversion Guide

VisionTrain Studio includes a local export center. It does not fake TensorRT, ONNX Runtime GPU, or OpenVINO support.

## PyTorch checkpoint to ONNX

Supported for PyTorch classification and anomaly checkpoints when the checkpoint includes enough config metadata to rebuild the model.

```powershell
python -m backend.exporter.export_router --checkpoint backend/runs/exp001/best.pt --format onnx --precision fp32 --imgsz 224 --opset 17
```

## Ultralytics export

YOLO and RT-DETR exports use the real Ultralytics API:

```python
from ultralytics import YOLO
model = YOLO(weights_or_checkpoint)
model.export(format="onnx", imgsz=640, half=False, dynamic=True)
```

Supported formats depend on your local Ultralytics install.

## TensorRT

TensorRT export is disabled unless all are true:

- CUDA is available through PyTorch.
- NVIDIA TensorRT Python package is importable.
- The selected export workflow supports engine export.

INT8 is intentionally marked as future work because real calibration dataset handling is required.
