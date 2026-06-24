from __future__ import annotations
from typing import Any, Dict
from .base_trainer import BaseTrainer, TrainerError

class SegmentationTrainer(BaseTrainer):
    """Real TorchVision semantic segmentation trainer for integer-mask datasets."""
    def _create_model(self, model_id: str, num_classes: int):
        from torchvision import models
        if model_id == "deeplabv3-resnet50":
            return models.segmentation.deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=num_classes)
        if model_id == "fcn-resnet50":
            return models.segmentation.fcn_resnet50(weights=None, weights_backbone=None, num_classes=num_classes)
        if model_id == "lraspp-mobilenetv3":
            if hasattr(models.segmentation, "lraspp_mobilenet_v3_large"):
                return models.segmentation.lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=num_classes)
            raise TrainerError("LR-ASPP MobileNetV3 is not available in this torchvision version.")
        raise TrainerError(f"Unsupported segmentation model: {model_id}")

    def run(self) -> Dict[str, Any]:
        try:
            import torch
            import torch.nn as nn
            from PIL import Image
            from torch.utils.data import DataLoader, Dataset
            from torchvision.transforms import functional as TF
        except Exception as exc:
            raise TrainerError(f"Segmentation dependencies failed to import: {exc}") from exc

        root = self.dataset_path
        train_cfg = self.config.get("training", {})
        img_size, batch, workers = int(train_cfg.get("image_size", 512)), int(train_cfg.get("batch_size", 2)), int(train_cfg.get("workers", 2))
        epochs = int(train_cfg.get("epochs", 30))
        num_classes = int(self.config.get("dataset", {}).get("num_classes") or self.model_info.get("num_classes") or 2)
        device = torch.device(self.resolve_torch_device())
        model_id = self.model_info.get("model_key") or self.model_info.get("id")

        class MaskDataset(Dataset):
            def __init__(self, split: str):
                self.img_dir = root / "images" / split
                self.mask_dir = root / "masks" / split
                exts = {".jpg",".jpeg",".png",".bmp",".webp"}
                self.images = sorted([p for p in self.img_dir.glob("*") if p.suffix.lower() in exts])
                self.masks_by_stem = {p.stem: p for p in self.mask_dir.glob("*") if p.suffix.lower() in exts}
            def __len__(self): return len(self.images)
            def __getitem__(self, idx):
                img_path = self.images[idx]
                mask_path = self.masks_by_stem.get(img_path.stem)
                if mask_path is None:
                    raise FileNotFoundError(f"Missing mask for {img_path.name}")
                image = Image.open(img_path).convert("RGB").resize((img_size, img_size))
                mask = Image.open(mask_path).convert("L").resize((img_size, img_size), Image.Resampling.NEAREST)
                image_t = TF.normalize(TF.to_tensor(image), [0.485,0.456,0.406], [0.229,0.224,0.225])
                mask_t = torch.as_tensor(list(mask.getdata()), dtype=torch.long).view(img_size, img_size)
                return image_t, mask_t

        train_ds, val_ds = MaskDataset("train"), MaskDataset("val")
        if len(train_ds) == 0: raise TrainerError("No segmentation training images found.")
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=device.type == "cuda")
        val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")

        self.log(f"Semantic segmentation trainer selected: {self.model_info.get('name')}")
        self.log(f"Images: train={len(train_ds)} val={len(val_ds)} classes={num_classes} device={device}")
        self.log("Masks must be single-channel integer class-id images: background=0, class1=1, etc.")
        model = self._create_model(model_id, num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("learning_rate", 0.0005)), weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        run_dir = self.output_dir / self.experiment_name
        run_dir.mkdir(parents=True, exist_ok=True)
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            if self.should_stop(): break
            model.train(); total_loss = total = 0
            for images, masks in train_loader:
                if self.should_stop(): break
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images)["out"], masks)
                loss.backward(); optimizer.step()
                total_loss += loss.item() * images.size(0); total += images.size(0)
            model.eval(); val_loss = val_total = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    loss = criterion(model(images)["out"], masks)
                    val_loss += loss.item() * images.size(0); val_total += images.size(0)
            epoch_train = total_loss / max(1, total)
            epoch_val = val_loss / max(1, val_total)
            self.metrics = {"epoch": epoch, "loss": round(epoch_val, 5), "score": "val_loss"}
            self.log(f"Epoch {epoch}/{epochs} train_loss={epoch_train:.5f} val_loss={epoch_val:.5f}")
            if epoch_val < best_val_loss:
                best_val_loss = epoch_val
                ckpt = run_dir / "best.pt"
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": epoch_val, "num_classes": num_classes, "config": self.config}, ckpt)
                self.metrics["checkpoint"] = str(ckpt)
                self.log(f"Best segmentation checkpoint saved: {ckpt}")

        status = "stopped" if self.should_stop() else "completed"
        self.metrics.update({"status": status, "loss": round(best_val_loss, 5)})
        self.log(f"Semantic segmentation training {status}.")
        return self.metrics
