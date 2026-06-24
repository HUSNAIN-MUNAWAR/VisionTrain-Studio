from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except Exception:
    yaml = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _images(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

def _result(task: str, fmt: str) -> Dict[str, Any]:
    return {"valid": False, "task": task, "dataset_format": fmt, "total_images": 0, "train_image_count": 0, "val_image_count": 0, "label_count": 0, "class_count": None, "missing_files": [], "warnings": [], "errors": [], "recommended_fixes": []}

def validate_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    dataset = config.get("dataset", {})
    task = config.get("task", "unknown")
    fmt = dataset.get("format") or config.get("model", {}).get("dataset_format") or "unknown"
    root = Path(dataset.get("path", "")).expanduser()
    result = _result(task, fmt)
    if not str(root):
        result["errors"].append("Dataset path is empty.")
        result["recommended_fixes"].append("Set an absolute local dataset path before validating.")
        return result
    if not root.exists():
        result["errors"].append(f"Dataset path does not exist: {root}")
        result["recommended_fixes"].append("Check spelling, drive letter, and folder permissions.")
        return result
    validators = {"yolo": _validate_yolo, "classification_folder": _validate_classification, "semantic_masks": _validate_semantic_masks, "coco_detection": _validate_coco_detection, "anomaly_folder": _validate_anomaly, "ocr_dataset": _validate_ocr}
    func = validators.get(fmt)
    if not func:
        result["errors"].append(f"Unsupported dataset format: {fmt}")
        return result
    return func(root, result, dataset)

def _validate_yolo(root: Path, result: Dict[str, Any], dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    yaml_path = Path(dataset_cfg.get("yaml") or root / "dataset.yaml")
    if not yaml_path.is_absolute():
        yaml_path = root / yaml_path
    train_img_dir, val_img_dir = root / "images" / "train", root / "images" / "val"
    train_lbl_dir, val_lbl_dir = root / "labels" / "train", root / "labels" / "val"
    for required in [yaml_path, train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        if not required.exists():
            result["missing_files"].append(str(required))
    train_images, val_images = _images(train_img_dir), _images(val_img_dir)
    labels = list(train_lbl_dir.glob("*.txt")) + list(val_lbl_dir.glob("*.txt")) if train_lbl_dir.exists() and val_lbl_dir.exists() else []
    result.update({"total_images": len(train_images)+len(val_images), "train_image_count": len(train_images), "val_image_count": len(val_images), "label_count": len(labels)})
    for img in train_images + val_images:
        split = "train" if train_img_dir in img.parents else "val"
        lbl_dir = train_lbl_dir if split == "train" else val_lbl_dir
        label_file = lbl_dir / f"{img.stem}.txt"
        if not label_file.exists():
            result["missing_files"].append(str(label_file))
            if len(result["missing_files"]) > 100:
                result["warnings"].append("More than 100 missing labels detected; showing first 100 only.")
                break
    if yaml_path.exists() and yaml is not None:
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            names = data.get("names") or []
            result["class_count"] = len(names) if isinstance(names, (list, dict)) else None
        except Exception as exc:
            result["warnings"].append(f"dataset.yaml exists but could not be parsed: {exc}")
    elif yaml_path.exists() and yaml is None:
        result["warnings"].append("PyYAML is not installed, so dataset.yaml class parsing was skipped.")
    if result["total_images"] == 0:
        result["errors"].append("No supported images found in images/train or images/val.")
    if result["label_count"] == 0:
        result["errors"].append("No YOLO .txt labels found.")
    if result["missing_files"]:
        result["errors"].append("Required YOLO files/folders or image labels are missing.")
        result["recommended_fixes"].append("Use dataset/images/train, dataset/images/val, dataset/labels/train, dataset/labels/val, and dataset.yaml.")
    result["valid"] = not result["errors"]
    return result

def _validate_classification(root: Path, result: Dict[str, Any], _dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    train_dir, val_dir = root / "train", root / "val"
    for required in [train_dir, val_dir]:
        if not required.exists():
            result["missing_files"].append(str(required))
    train_classes = [p for p in train_dir.iterdir() if p.is_dir()] if train_dir.exists() else []
    val_classes = [p for p in val_dir.iterdir() if p.is_dir()] if val_dir.exists() else []
    train_images, val_images = _images(train_dir), _images(val_dir)
    result.update({"total_images": len(train_images)+len(val_images), "train_image_count": len(train_images), "val_image_count": len(val_images), "class_count": len(train_classes)})
    if not train_classes:
        result["errors"].append("No class folders found inside train/.")
    if set(p.name for p in train_classes) != set(p.name for p in val_classes):
        result["warnings"].append("Train and val class folder names do not match exactly.")
    if result["total_images"] == 0:
        result["errors"].append("No classification images found.")
    if result["missing_files"]:
        result["errors"].append("Classification train/val folders are missing.")
        result["recommended_fixes"].append("Use dataset/train/class_name/*.jpg and dataset/val/class_name/*.jpg.")
    result["valid"] = not result["errors"]
    return result

def _validate_semantic_masks(root: Path, result: Dict[str, Any], _dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    train_img, val_img = root / "images" / "train", root / "images" / "val"
    train_mask, val_mask = root / "masks" / "train", root / "masks" / "val"
    for required in [train_img, val_img, train_mask, val_mask]:
        if not required.exists():
            result["missing_files"].append(str(required))
    train_images, val_images = _images(train_img), _images(val_img)
    masks = _images(train_mask) + _images(val_mask)
    result.update({"total_images": len(train_images)+len(val_images), "train_image_count": len(train_images), "val_image_count": len(val_images), "label_count": len(masks)})
    mask_stems = {p.stem for p in masks}
    for img in train_images + val_images:
        if img.stem not in mask_stems:
            result["missing_files"].append(f"mask for {img.name}")
            if len(result["missing_files"]) > 100:
                result["warnings"].append("More than 100 missing masks detected; showing first 100 only.")
                break
    if not masks:
        result["errors"].append("No semantic mask images found.")
    if result["missing_files"]:
        result["errors"].append("Required semantic segmentation files/folders or masks are missing.")
        result["recommended_fixes"].append("Use matching image and mask filenames under images/train, images/val, masks/train, masks/val.")
    if result["total_images"] == 0:
        result["errors"].append("No training/validation images found.")
    result["valid"] = not result["errors"]
    return result

def _validate_coco_detection(root: Path, result: Dict[str, Any], _dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    train_img, val_img = root / "images" / "train", root / "images" / "val"
    train_json, val_json = root / "annotations" / "train.json", root / "annotations" / "val.json"
    for required in [train_img, val_img, train_json, val_json]:
        if not required.exists():
            result["missing_files"].append(str(required))
    train_images, val_images = _images(train_img), _images(val_img)
    result.update({"total_images": len(train_images)+len(val_images), "train_image_count": len(train_images), "val_image_count": len(val_images)})
    categories, ann_count = set(), 0
    for file in [train_json, val_json]:
        if file.exists():
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                categories.update(c.get("id") for c in data.get("categories", []) if c.get("id") is not None)
                ann_count += len(data.get("annotations", []))
            except Exception as exc:
                result["errors"].append(f"Could not parse {file.name}: {exc}")
    result["class_count"], result["label_count"] = len(categories) if categories else None, ann_count
    if not ann_count:
        result["errors"].append("No COCO annotations found.")
    if result["missing_files"]:
        result["errors"].append("COCO images or annotation JSON files are missing.")
        result["recommended_fixes"].append("Use images/train, images/val, annotations/train.json, annotations/val.json.")
    result["valid"] = not result["errors"]
    return result

def _validate_anomaly(root: Path, result: Dict[str, Any], _dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    train_normal, val_normal, val_anomaly = root / "train" / "normal", root / "val" / "normal", root / "val" / "anomaly"
    for required in [train_normal, val_normal, val_anomaly]:
        if not required.exists():
            result["missing_files"].append(str(required))
    train_images, val_images = _images(train_normal), _images(val_normal) + _images(val_anomaly)
    result.update({"total_images": len(train_images)+len(val_images), "train_image_count": len(train_images), "val_image_count": len(val_images), "class_count": 2})
    if not train_images:
        result["errors"].append("No normal training images found.")
    if not val_images:
        result["warnings"].append("No validation images found; threshold quality cannot be estimated.")
    if result["missing_files"]:
        result["errors"].append("Anomaly dataset folders are missing.")
        result["recommended_fixes"].append("Use train/normal, val/normal, and val/anomaly folders.")
    result["valid"] = not result["errors"]
    return result

def _validate_ocr(root: Path, result: Dict[str, Any], _dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    images = _images(root)
    result["total_images"] = len(images)
    result["warnings"].append("OCR training is a future integration hook. Validation only checks that images exist.")
    if not images:
        result["errors"].append("No OCR images found.")
    result["valid"] = not result["errors"]
    return result
