# Animation assets

`visiontrain-training-reel.html` is generated from the committed classification demo and the backend preflight services.

Regenerate after changing dataset validation, hardware profiling, model compatibility, or training routing:

```bash
python scripts/generate_training_animation.py
```

For actual model evidence after local dependencies are installed, use `scripts/generate_real_runtime_video.py` and review [`REAL_RUNTIME.md`](REAL_RUNTIME.md). That path executes the trainer, reloads the checkpoint, and records per-image validation predictions.
