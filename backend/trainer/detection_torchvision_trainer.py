from __future__ import annotations
import json
from typing import Any, Dict
from .base_trainer import BaseTrainer, TrainerError

class TorchVisionDetectionTrainer(BaseTrainer):
    """Real TorchVision object detection trainer for COCO-style annotations.

    Training is real. Full COCO mAP evaluation is a future improvement because
    production mAP needs pycocotools/evaluator integration.
    """
    def _create_model(self, model_id: str, num_classes: int):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn, ssd300_vgg16
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        if model_id == "fasterrcnn-resnet50-fpn":
            model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            return model
        if model_id == "retinanet-resnet50-fpn":
            try:
                return retinanet_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
            except TypeError as exc:
                raise TrainerError("This torchvision version does not support RetinaNet num_classes constructor cleanly. Upgrade torchvision.") from exc
        if model_id == "ssd300-vgg16":
            try:
                return ssd300_vgg16(weights=None, weights_backbone=None, num_classes=num_classes)
            except TypeError as exc:
                raise TrainerError("This torchvision version does not support SSD300 num_classes constructor cleanly. Upgrade torchvision.") from exc
        raise TrainerError(f"Unsupported torchvision detection model: {model_id}")

    def run(self) -> Dict[str, Any]:
        try:
            import torch
            from PIL import Image
            from torch.utils.data import DataLoader, Dataset
            from torchvision.transforms import functional as TF
        except Exception as exc:
            raise TrainerError(f"TorchVision detection dependencies failed to import: {exc}") from exc

        root = self.dataset_path
        train_cfg = self.config.get("training", {})
        batch, workers, epochs = int(train_cfg.get("batch_size", 2)), int(train_cfg.get("workers", 2)), int(train_cfg.get("epochs", 20))
        device = torch.device(self.resolve_torch_device())
        model_id = self.model_info.get("model_key") or self.model_info.get("id")

        class CocoDataset(Dataset):
            def __init__(self, split: str):
                self.img_dir = root / "images" / split
                ann_file = root / "annotations" / f"{split}.json"
                data = json.loads(ann_file.read_text(encoding="utf-8"))
                self.images = data.get("images", [])
                self.categories = data.get("categories", [])
                self.anns_by_image = {}
                for ann in data.get("annotations", []):
                    self.anns_by_image.setdefault(ann["image_id"], []).append(ann)
            def __len__(self): return len(self.images)
            def __getitem__(self, idx):
                item = self.images[idx]
                image = Image.open(self.img_dir / item["file_name"]).convert("RGB")
                image_t = TF.to_tensor(image)
                anns = self.anns_by_image.get(item["id"], [])
                boxes, labels, area, iscrowd = [], [], [], []
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    boxes.append([x, y, x + w, y + h])
                    labels.append(int(ann.get("category_id", 1)))
                    area.append(float(ann.get("area", w * h)))
                    iscrowd.append(int(ann.get("iscrowd", 0)))
                target = {
                    "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                    "labels": torch.as_tensor(labels, dtype=torch.int64),
                    "image_id": torch.tensor([int(item["id"])]),
                    "area": torch.as_tensor(area, dtype=torch.float32),
                    "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
                }
                return image_t, target

        def collate(batch_items):
            return tuple(zip(*batch_items))

        train_ds, val_ds = CocoDataset("train"), CocoDataset("val")
        max_cat = max([int(c.get("id", 0)) for c in (train_ds.categories or val_ds.categories)] or [1])
        num_classes = max_cat + 1
        if len(train_ds) == 0: raise TrainerError("No COCO training images found.")
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers, collate_fn=collate)

        self.log(f"TorchVision detection trainer selected: {self.model_info.get('name')}")
        self.log(f"COCO classes including background: {num_classes} | train={len(train_ds)} val={len(val_ds)} device={device}")
        model = self._create_model(model_id, num_classes).to(device)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=float(train_cfg.get("learning_rate", 0.005)), momentum=0.9, weight_decay=0.0005)
        run_dir = self.output_dir / self.experiment_name
        run_dir.mkdir(parents=True, exist_ok=True)
        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            if self.should_stop(): break
            model.train(); epoch_loss = 0.0; steps = 0
            for images, targets in train_loader:
                if self.should_stop(): break
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad(set_to_none=True)
                losses.backward(); optimizer.step()
                epoch_loss += float(losses.item()); steps += 1
            avg_loss = epoch_loss / max(1, steps)
            self.metrics = {"epoch": epoch, "loss": round(avg_loss, 5), "score": "train_loss"}
            self.log(f"Epoch {epoch}/{epochs} detection_train_loss={avg_loss:.5f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                ckpt = run_dir / "best.pt"
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "loss": avg_loss, "num_classes": num_classes, "config": self.config}, ckpt)
                self.metrics["checkpoint"] = str(ckpt)
                self.log(f"Best detection checkpoint saved: {ckpt}")
            if len(val_ds):
                self.log("Validation mAP is a future improvement; install pycocotools and add COCO evaluator for production-grade mAP.")
        status = "stopped" if self.should_stop() else "completed"
        self.metrics.update({"status": status, "loss": round(best_loss, 5)})
        self.log(f"TorchVision detection training {status}.")
        return self.metrics
