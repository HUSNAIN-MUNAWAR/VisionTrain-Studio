from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from .base_trainer import BaseTrainer, TrainerError


class SmallAutoencoder:
    """Lazy factory wrapper so exporter can rebuild the anomaly model without importing torch at module import time."""
    def __new__(cls):
        import torch.nn as nn
        class _AutoEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(nn.Conv2d(3,32,4,2,1), nn.ReLU(True), nn.Conv2d(32,64,4,2,1), nn.ReLU(True), nn.Conv2d(64,128,4,2,1), nn.ReLU(True))
                self.decoder = nn.Sequential(nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(True), nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(True), nn.ConvTranspose2d(32,3,4,2,1), nn.Sigmoid())
            def forward(self, x):
                return self.decoder(self.encoder(x))
        return _AutoEncoder()

class AnomalyTrainer(BaseTrainer):
    """Practical PyTorch convolutional autoencoder trainer for anomaly detection."""
    def run(self) -> Dict[str, Any]:
        try:
            import torch
            import torch.nn as nn
            from PIL import Image
            from torch.utils.data import DataLoader, Dataset
            from torchvision.transforms import Compose, Resize, ToTensor
        except Exception as exc:
            raise TrainerError(f"Anomaly trainer dependencies failed to import: {exc}") from exc

        root = self.dataset_path
        train_cfg = self.config.get("training", {})
        img_size, batch, workers = int(train_cfg.get("image_size", 224)), int(train_cfg.get("batch_size", 8)), int(train_cfg.get("workers", 2))
        epochs = int(train_cfg.get("epochs", 30))
        device = torch.device(self.resolve_torch_device())
        transform = Compose([Resize((img_size, img_size)), ToTensor()])
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        class ImageSet(Dataset):
            def __init__(self, folder: Path, label: int = 0):
                self.files = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
                self.label = label
            def __len__(self): return len(self.files)
            def __getitem__(self, idx):
                img = Image.open(self.files[idx]).convert("RGB")
                return transform(img), self.label

        train_ds = ImageSet(root / "train" / "normal", 0)
        val_normal, val_anomaly = ImageSet(root / "val" / "normal", 0), ImageSet(root / "val" / "anomaly", 1)
        if len(train_ds) == 0: raise TrainerError("Anomaly training requires train/normal images.")
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers)
        val_loader = DataLoader(torch.utils.data.ConcatDataset([val_normal, val_anomaly]), batch_size=batch, shuffle=False, num_workers=workers)

        class AutoEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(nn.Conv2d(3,32,4,2,1), nn.ReLU(True), nn.Conv2d(32,64,4,2,1), nn.ReLU(True), nn.Conv2d(64,128,4,2,1), nn.ReLU(True))
                self.decoder = nn.Sequential(nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(True), nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(True), nn.ConvTranspose2d(32,3,4,2,1), nn.Sigmoid())
            def forward(self, x): return self.decoder(self.encoder(x))

        model = AutoEncoder().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("learning_rate", 0.001)), weight_decay=1e-5)
        criterion = nn.MSELoss(reduction="none")
        run_dir = self.output_dir / self.experiment_name
        run_dir.mkdir(parents=True, exist_ok=True)
        best_loss, train_scores = float("inf"), []

        self.log(f"Anomaly autoencoder trainer selected. train_normal={len(train_ds)} val={len(val_normal)+len(val_anomaly)} device={device}")
        for epoch in range(1, epochs + 1):
            if self.should_stop(): break
            model.train(); total_loss = total = 0
            for images, _ in train_loader:
                if self.should_stop(): break
                images = images.to(device)
                recon = model(images)
                loss_map = criterion(recon, images)
                loss = loss_map.mean()
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                total_loss += loss.item() * images.size(0); total += images.size(0)
                train_scores.extend(loss_map.mean(dim=(1,2,3)).detach().cpu().tolist())
            epoch_loss = total_loss / max(1, total)
            threshold = float(torch.tensor(train_scores).mean() + 3 * torch.tensor(train_scores).std()) if train_scores else epoch_loss
            model.eval(); correct = count = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    scores = criterion(model(images), images).mean(dim=(1,2,3)).cpu()
                    preds = (scores > threshold).long()
                    correct += (preds == labels).sum().item(); count += labels.numel()
            val_acc = correct / max(1, count)
            self.metrics = {"epoch": epoch, "loss": round(epoch_loss, 6), "accuracy": round(val_acc, 5), "threshold": round(threshold, 6)}
            self.log(f"Epoch {epoch}/{epochs} recon_loss={epoch_loss:.6f} threshold={threshold:.6f} val_acc={val_acc:.4f}")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                ckpt = run_dir / "best_autoencoder.pt"
                torch.save({"model_state": model.state_dict(), "threshold": threshold, "epoch": epoch, "config": self.config}, ckpt)
                self.metrics["checkpoint"] = str(ckpt)
                self.log(f"Best anomaly checkpoint saved: {ckpt}")
        status = "stopped" if self.should_stop() else "completed"
        self.metrics.update({"status": status, "loss": round(best_loss, 6)})
        self.log(f"Anomaly training {status}.")
        return self.metrics
