from __future__ import annotations
from typing import Any, Dict
from .base_trainer import BaseTrainer, TrainerError

class OCRTrainer(BaseTrainer):
    """OCR integration hook. This does not pretend to train OCR."""
    def build_external_command(self) -> str:
        dataset = self.config.get("dataset", {}).get("path", "")
        return f"# Future PaddleOCR example: python tools/train.py -c config.yml -o Global.train_data_dir={dataset}"

    def run(self) -> Dict[str, Any]:
        self.log("OCR trainer requested, but OCR training is marked as Future Integration.")
        self.log(self.build_external_command())
        raise TrainerError("OCR training is not enabled in this repo. Integrate PaddleOCR/EasyOCR separately before enabling it.")
