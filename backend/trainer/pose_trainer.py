from __future__ import annotations
from typing import Any, Dict
from .yolo_trainer import YoloTrainer

class PoseTrainer(YoloTrainer):
    """YOLO pose uses the real Ultralytics trainer with pose weights and YOLO pose dataset YAML."""
    def run(self) -> Dict[str, Any]:
        self.log("Pose trainer routed to Ultralytics YOLO pose backend.")
        return super().run()
