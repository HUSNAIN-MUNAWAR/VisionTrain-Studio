# Real VisionTrain runtime evidence

This is the accepted VisionTrain animation evidence. It executes the application's real `TrainingRouter` and `ClassificationTrainer`, loads the emitted checkpoint, and runs checkpoint-backed validation inference. It is not a hand-authored training reel.

## Reproduce

From the repository root:

```powershell
py -3.12 scripts/demo_smoke_check.py --train
py -3.12 scripts/generate_real_runtime_video.py
```

The generator uses the committed `demo/classification_dataset`, CPU `ResNet18`, TorchVision transforms, one epoch, batch size `2`, image size `64`, and seed `42`. It writes the checkpoint to the ignored `backend/runs/real_runtime/` directory and preserves the observable run outputs under `docs/animation/runtime-evidence/`.

## Measured run

- PyTorch `2.7.1+cpu`, TorchVision `0.22.1+cpu`, device `cpu`, CUDA unavailable.
- Dataset validator: valid, `6` train images, `4` validation images, `2` classes, no validation errors.
- Trainer: real `TrainingRouter` -> `ClassificationTrainer` -> TorchVision `resnet18(weights=None)`.
- One epoch: train loss `0.25788`, train accuracy `0.8333`, validation loss `0.22180`, validation accuracy `1.0000`.
- Checkpoint reload: state dictionary loaded into a fresh ResNet18 with `11,177,538` parameters.
- Checkpoint-backed validation inference: `4/4` correct; each sample's ground truth, prediction, confidence, and logits are preserved in the CSV/JSON.

The validation result is only a smoke-run observation over ten tiny committed bright/dark images. It is not real-world accuracy, a production benchmark, or evidence of generalization.

## Preserved outputs

| Artifact | Evidence |
| --- | --- |
| [`visiontrain-runtime.mp4`](runtime-evidence/visiontrain-runtime.mp4) | Eight-stage 256-frame animation rendered from the actual validator, model gate, trainer status, epoch metrics, checkpoint load, and validation predictions. |
| [`visiontrain-runtime-poster.png`](runtime-evidence/visiontrain-runtime-poster.png) | Poster showing the actual checkpoint-backed predictions on all four validation images. Green borders mean the recorded prediction was correct; any incorrect result would be rendered as an error state. |
| [`visiontrain-predictions.csv`](runtime-evidence/visiontrain-predictions.csv) | One row per validation image with ground truth, predicted class, confidence, and correctness. |
| [`visiontrain-dataset-files.csv`](runtime-evidence/visiontrain-dataset-files.csv) | Every committed image, split/class assignment, byte size, and SHA-256. |
| [`visiontrain-training.log`](runtime-evidence/visiontrain-training.log) | Actual router/trainer log, including dataset validation, epoch metrics, checkpoint save, and completion. |
| [`visiontrain-runtime-summary.json`](runtime-evidence/visiontrain-runtime-summary.json) | Full configuration, model gate, status, checkpoint digest, per-sample logits, and artifact hashes. |

The binary checkpoint is intentionally ignored by repository policy. Its path, size, class list, parameter count, and SHA-256 are recorded in the JSON report; the committed prediction trace proves what was loaded and evaluated during this run.

The earlier HTML asset at [`frontend/animation/visiontrain-training-reel.html`](../../frontend/animation/visiontrain-training-reel.html) remains a preflight/reference UI contract. Use the runtime-evidence files above for model evidence.
