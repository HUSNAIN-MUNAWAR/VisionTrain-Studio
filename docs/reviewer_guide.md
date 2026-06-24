# Reviewer Guide

This guide is for recruiters, hiring managers, and engineers who want to inspect VisionTrain Studio quickly.

## 90-Second Review

1. Read the first screen of the README.
2. Open the screenshots and workflow diagram.
3. Check the backend entrypoint: `backend/main.py`.
4. Check the training router: `backend/trainer/train_router.py`.
5. Check compatibility gating: `backend/trainer/model_compatibility.py`.
6. Run the smoke check:

```bash
python scripts/demo_smoke_check.py
```

## What The Smoke Check Proves

- The included demo dataset is present.
- Dataset validation can inspect a real ImageFolder dataset.
- Config normalization produces a runnable training config.
- Model compatibility reports whether the local environment can train the demo model.
- The script writes a ready-to-run demo config to `backend/configs/saved/demo_classification_config.json`.

## Optional Training Proof

After running setup, the same demo can attempt a one-epoch classification run:

```bash
python scripts/demo_smoke_check.py --train
```

This requires `torch` and `torchvision`. If they are missing, the script exits with a clear dependency message instead of pretending training is available.

## Useful Test Commands

```bash
python -m compileall backend
python -m backend.trainer.train_router --check-mapping
pytest
```
