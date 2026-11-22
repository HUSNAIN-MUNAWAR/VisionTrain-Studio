from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "animation" / "visiontrain_training_sequence.json"
REEL = ROOT / "frontend" / "animation" / "visiontrain-training-reel.html"


def test_training_animation_uses_real_preflight_services() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/generate_training_animation.py"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dataset valid:" in result.stdout

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    reel = REEL.read_text(encoding="utf-8")
    assert manifest["source"]["synthetic"] is True
    assert manifest["dataset"]["validation"]["valid"] is True
    assert manifest["dataset"]["validation"]["class_count"] == 2
    assert manifest["model"]["selected"]["id"] == "resnet18"
    assert len(manifest["events"]) == 6
    assert "no fabricated loss" in reel.lower()
