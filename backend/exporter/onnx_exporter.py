from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict


def _build_classification_model(config: Dict[str, Any], num_classes: int):
    from backend.trainer.classification_trainer import ClassificationTrainer
    from threading import Event

    dummy_model = config.get("model", {})
    trainer = ClassificationTrainer(config, dummy_model, lambda _m: None, Event())
    return trainer._create_model(dummy_model.get("id", "resnet18"), num_classes)


def _load_checkpoint(checkpoint: Path) -> Dict[str, Any]:
    import torch

    data = torch.load(checkpoint, map_location="cpu")
    if isinstance(data, dict):
        return data
    return {"model_state": data}


def export_pytorch_to_onnx(payload: Dict[str, Any], log: Callable[[str], None]) -> Dict[str, Any]:
    import torch

    checkpoint = Path(payload["checkpoint"]).expanduser()
    out_dir = Path(payload.get("output_dir") or "backend/runs/exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    image_size = int(payload.get("image_size", payload.get("imgsz", 224)))
    opset = int(payload.get("opset", 17))
    dynamic = bool(payload.get("dynamic_batch", True))
    precision = str(payload.get("precision", "fp32")).lower()
    cfg = payload.get("config") or {}
    ckpt = _load_checkpoint(checkpoint)
    cfg = ckpt.get("config") or cfg
    task = cfg.get("task") or payload.get("task", "image_classification")
    log(f"Loading PyTorch checkpoint: {checkpoint}")
    if task == "image_classification":
        classes = ckpt.get("classes") or list(range(int(payload.get("num_classes", 2))))
        cfg.setdefault("model", {}).setdefault("id", payload.get("model_id", "resnet18"))
        model = _build_classification_model(cfg, len(classes))
        state = ckpt.get("model_state") or ckpt.get("state_dict")
        model.load_state_dict(state, strict=False)
        dummy_shape = (1, 3, image_size, image_size)
    elif task == "anomaly_detection":
        from backend.trainer.anomaly_trainer import SmallAutoencoder

        model = SmallAutoencoder()
        state = ckpt.get("model_state") or ckpt.get("state_dict")
        model.load_state_dict(state, strict=False)
        dummy_shape = (1, 3, image_size, image_size)
    else:
        raise RuntimeError(f"PyTorch ONNX export for task '{task}' is not implemented in this local exporter yet.")
    model.eval()
    if precision == "fp16":
        model.half()
        dummy = torch.randn(*dummy_shape).half()
    else:
        dummy = torch.randn(*dummy_shape)
    output = out_dir / f"{checkpoint.stem}_{task}_{precision}.onnx"
    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic else None
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )
    log(f"ONNX export written: {output}")
    validation = {"onnx_checked": False}
    try:
        import onnx

        model_proto = onnx.load(str(output))
        onnx.checker.check_model(model_proto)
        validation = {"onnx_checked": True, "valid": True}
        log("ONNX checker validation passed.")
    except Exception as exc:
        validation = {"onnx_checked": False, "warning": str(exc)}
        log(f"ONNX validation skipped/failed: {exc}")
    return {"status": "completed", "output": str(output), "validation": validation}


def export_ultralytics(payload: Dict[str, Any], log: Callable[[str], None]) -> Dict[str, Any]:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(f"Ultralytics is not installed or failed to import: {exc}") from exc
    checkpoint = payload["checkpoint"]
    fmt = str(payload.get("format", "onnx")).lower()
    if fmt == "tensorrt":
        fmt = "engine"
    image_size = int(payload.get("image_size", payload.get("imgsz", 640)))
    half = str(payload.get("precision", "fp32")).lower() == "fp16"
    dynamic = bool(payload.get("dynamic_batch", True))
    log(f"Starting Ultralytics export: checkpoint={checkpoint} format={fmt} imgsz={image_size} half={half} dynamic={dynamic}")
    model = YOLO(checkpoint)
    exported = model.export(format=fmt, imgsz=image_size, half=half, dynamic=dynamic)
    log(f"Ultralytics export completed: {exported}")
    return {"status": "completed", "output": str(exported)}
