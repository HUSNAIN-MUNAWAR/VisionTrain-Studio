from __future__ import annotations
from typing import Any, Dict
from .base_trainer import BaseTrainer, TrainerError

class ClassificationTrainer(BaseTrainer):
    """Real PyTorch/TorchVision/TIMM image classification trainer."""
    def _create_model(self, model_id: str, num_classes: int):
        import torch.nn as nn
        from torchvision import models
        if model_id == "resnet18":
            model = models.resnet18(weights=None); model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_id == "resnet50":
            model = models.resnet50(weights=None); model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_id == "mobilenetv3-small":
            model = models.mobilenet_v3_small(weights=None); model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        elif model_id == "mobilenetv3-large":
            model = models.mobilenet_v3_large(weights=None); model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        elif model_id == "efficientnet-b0":
            model = models.efficientnet_b0(weights=None); model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        elif model_id == "convnext-tiny":
            model = models.convnext_tiny(weights=None); model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        elif model_id == "vit-b-16":
            if hasattr(models, "vit_b_16"):
                model = models.vit_b_16(weights=None); model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            else:
                try:
                    import timm
                    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
                except Exception as exc:
                    raise TrainerError(f"ViT-B/16 is unavailable in torchvision and timm import failed: {exc}") from exc
        else:
            raise TrainerError(f"Unsupported classification model id: {model_id}")
        return model

    def _optimizer(self, params):
        import torch
        train = self.config.get("training", {})
        opt, lr = str(train.get("optimizer", "adamw")).lower(), float(train.get("learning_rate", 0.001))
        if opt == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    def run(self) -> Dict[str, Any]:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader
            from torchvision import datasets, transforms
        except Exception as exc:
            raise TrainerError(f"PyTorch/TorchVision import failed: {exc}") from exc

        root = self.dataset_path
        if not (root / "train").exists() or not (root / "val").exists():
            raise TrainerError("Classification dataset must contain train/ and val/ folders.")

        train_cfg = self.config.get("training", {})
        img_size, batch, workers = int(train_cfg.get("image_size", 224)), int(train_cfg.get("batch_size", 8)), int(train_cfg.get("workers", 4))
        epochs, checkpoint_interval = int(train_cfg.get("epochs", 50)), int(train_cfg.get("checkpoint_interval", 1))
        device = torch.device(self.resolve_torch_device())
        model_id = self.model_info.get("model_key") or self.model_info.get("id")

        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        val_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_ds, val_ds = datasets.ImageFolder(root / "train", transform=train_tf), datasets.ImageFolder(root / "val", transform=val_tf)
        if len(train_ds.classes) < 2:
            raise TrainerError("At least two class folders are recommended for classification training.")
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=device.type == "cuda")
        val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")

        self.log(f"Classification trainer selected: {self.model_info.get('name')}")
        self.log(f"Classes: {train_ds.classes}")
        self.log(f"Train images: {len(train_ds)} | Val images: {len(val_ds)} | Device: {device}")

        model = self._create_model(model_id, len(train_ds.classes)).to(device)
        criterion, optimizer = nn.CrossEntropyLoss(), self._optimizer(model.parameters())
        scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()
        scheduler = None
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)

        best_acc = 0.0
        run_dir = self.output_dir / self.experiment_name
        run_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            if self.should_stop():
                self.log("Stop requested before epoch start.")
                break
            model.train()
            train_loss = train_correct = train_total = 0
            for images, labels in train_loader:
                if self.should_stop(): break
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward(); optimizer.step()
                train_loss += loss.item() * images.size(0)
                train_correct += (outputs.argmax(dim=1) == labels).sum().item()
                train_total += labels.size(0)
            if scheduler: scheduler.step()

            model.eval()
            val_loss = val_correct = val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                    val_total += labels.size(0)
            train_acc = train_correct / max(1, train_total)
            val_acc = val_correct / max(1, val_total)
            epoch_train_loss = train_loss / max(1, train_total)
            epoch_val_loss = val_loss / max(1, val_total)
            self.metrics = {"loss": round(epoch_val_loss, 5), "accuracy": round(val_acc, 5), "epoch": epoch}
            self.log(f"Epoch {epoch}/{epochs} train_loss={epoch_train_loss:.5f} train_acc={train_acc:.4f} val_loss={epoch_val_loss:.5f} val_acc={val_acc:.4f}")
            if val_acc >= best_acc:
                best_acc = val_acc
                ckpt = run_dir / "best.pt"
                torch.save({"model_state": model.state_dict(), "classes": train_ds.classes, "epoch": epoch, "val_acc": val_acc, "config": self.config}, ckpt)
                self.metrics["checkpoint"] = str(ckpt)
                self.log(f"Best checkpoint saved: {ckpt}")
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                torch.save({"model_state": model.state_dict(), "classes": train_ds.classes, "epoch": epoch, "val_acc": val_acc}, run_dir / f"epoch_{epoch}.pt")

        status = "stopped" if self.should_stop() else "completed"
        self.metrics.update({"status": status, "score": round(best_acc, 5)})
        self.log(f"Classification training {status}. Best val accuracy={best_acc:.4f}")
        return self.metrics
