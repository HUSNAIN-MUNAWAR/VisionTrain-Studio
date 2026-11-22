"""Generate a deterministic VisionTrain preflight/training animation manifest."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.trainer.config_manager import get_model_by_id, normalize_config  # noqa: E402
from backend.trainer.dataset_validator import validate_dataset  # noqa: E402
from backend.trainer.model_compatibility import evaluate_model_compatibility  # noqa: E402


FIXTURE = ROOT / "demo" / "animation_fixture.synthetic.json"
MANIFEST = ROOT / "docs" / "animation" / "visiontrain_training_sequence.json"
REEL = ROOT / "frontend" / "animation" / "visiontrain-training-reel.html"


def image_stats(dataset_root: Path) -> dict[str, object]:
    try:
        from PIL import Image, ImageStat
    except ImportError:
        return {"mode": "metadata_only", "reason": "Pillow is not installed"}
    classes: dict[str, dict[str, object]] = {}
    for split in ("train", "val"):
        for image_path in sorted((dataset_root / split).rglob("*.png")):
            class_name = image_path.parent.name
            with Image.open(image_path) as image:
                stats = ImageStat.Stat(image.convert("RGB"))
                entry = classes.setdefault(class_name, {"files": 0, "splits": {}, "mean_rgb": []})
                entry["files"] = int(entry["files"]) + 1
                entry["splits"][split] = int(entry["splits"].get(split, 0)) + 1
                entry["mean_rgb"].append([round(value, 2) for value in stats.mean])
    for entry in classes.values():
        values = entry.pop("mean_rgb")
        entry["mean_rgb"] = [round(sum(row[index] for row in values) / len(values), 2) for index in range(3)]
    return {"mode": "pillow_inspection", "classes": classes}


def build_manifest() -> dict[str, object]:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    dataset_root = ROOT / fixture["dataset_path"]
    config = normalize_config({
        "task": fixture["task"],
        "model": {"id": fixture["model_id"], "dataset_format": fixture["dataset_format"]},
        "dataset": {"path": str(dataset_root), "format": fixture["dataset_format"]},
        "training": {"epochs": fixture["epochs"], "batch_size": fixture["batch_size"], "image_size": fixture["image_size"], "workers": 0},
    })
    validation = validate_dataset(config)
    compatibility = evaluate_model_compatibility(device_mode="cpu", dataset_format=fixture["dataset_format"])
    selected = next(model for model in compatibility["models"] if model["id"] == fixture["model_id"])
    model_registry = get_model_by_id(fixture["model_id"])
    source_bytes = FIXTURE.read_bytes()
    return {
        "schema_version": "1.0",
        "fixture": fixture["fixture_id"],
        "source": {
            "path": FIXTURE.relative_to(ROOT).as_posix(),
            "bytes": len(source_bytes),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "synthetic": True,
            "dataset_path": dataset_root.relative_to(ROOT).as_posix(),
        },
        "model": {
            "registry": "backend/configs/models.json",
            "compatibility_service": "backend/trainer/model_compatibility.py::evaluate_model_compatibility",
            "selected": model_registry,
            "compatibility": {key: selected.get(key) for key in ["enabled", "reason", "missing_dependency", "expected_device", "recommended_batch_size", "recommended_precision"]},
            "claim_boundary": "Synthetic image-folder preflight and model registry evidence; no accuracy claim without a completed training run.",
        },
        "dataset": {
            "validation": validation,
            "image_stats": image_stats(dataset_root),
        },
        "events": [
            {"id": "dataset", "short": "Dataset", "title": "Demo images are discovered", "detail": "The real ImageFolder validator scans the committed bright/dark train and validation folders.", "metric": f"{validation['total_images']} images · {validation['class_count']} classes", "status": "discovered"},
            {"id": "validate", "short": "Validate", "title": "Dataset gate passes", "detail": "Train/val folders and class names are checked before a training request can start.", "metric": f"valid {validation['valid']} · train {validation['train_image_count']} · val {validation['val_image_count']}", "status": "passed"},
            {"id": "profile", "short": "Profile", "title": "Hardware and packages are profiled", "detail": "The compatibility service evaluates the selected device, RAM, CUDA, PyTorch, and TorchVision requirements.", "metric": f"device {selected['expected_device']} · precision {selected['recommended_precision']}", "status": "checked"},
            {"id": "model", "short": "Model", "title": "ResNet18 enters the registry gate", "detail": "The registry points to the real classification trainer and exposes the exact reason launch is enabled or blocked.", "metric": f"{selected['model_id'] if 'model_id' in selected else selected['id']} · {selected['reason']}", "status": "gated"},
            {"id": "training", "short": "Train", "title": "Training request stays honest", "detail": "The UI can launch a real trainer only when its declared dependencies are available; this manifest never fabricates loss or accuracy.", "metric": f"enabled {selected['enabled']} · missing {selected['missing_dependency'] or 'none'}", "status": "ready" if selected['enabled'] else "blocked"},
            {"id": "export", "short": "Export", "title": "Export remains downstream", "detail": "ONNX/TorchScript export is presented as a post-checkpoint stage, not as evidence when no checkpoint exists.", "metric": "checkpoint required · no synthetic benchmark", "status": "deferred"},
        ],
    }


def reel_html(manifest: dict[str, object]) -> str:
    sequence = json.dumps(manifest, indent=2, sort_keys=True, default=str).replace("</", "<\\/")
    return dedent(
        f"""\
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
          <title>VisionTrain Studio — Training preflight reference reel</title>
          <style>
            :root {{ color-scheme: dark; --ink: #eff6f4; --muted: #9bb3b0; --line: #2b4b4b; --accent: #71e0bd; }}
            * {{ box-sizing: border-box; }} body {{ margin: 0; min-width: 320px; background: radial-gradient(circle at 84% 0%, #173e3e 0, #091617 55%, #050b0b 100%); color: var(--ink); font: 15px/1.5 Inter, ui-sans-serif, system-ui, sans-serif; }}
            .reel {{ max-width: 1180px; margin: 0 auto; padding: 42px 28px 34px; }} .eyebrow {{ color: var(--accent); font-size: 12px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }}
            h1 {{ max-width: 820px; margin: 8px 0 10px; font-size: clamp(28px, 5vw, 52px); line-height: 1.02; letter-spacing: -.04em; }} .lede {{ max-width: 790px; margin: 0; color: var(--muted); font-size: 17px; }}
            .model-line {{ display: flex; flex-wrap: wrap; gap: 8px 18px; margin-top: 22px; color: var(--muted); font-size: 12px; }} .model-line span::before {{ content: "•"; color: var(--accent); margin-right: 8px; }}
            .stage {{ margin-top: 36px; overflow: hidden; border: 1px solid var(--line); border-radius: 22px; background: linear-gradient(145deg, rgba(18,54,52,.94), rgba(6,17,17,.97)); box-shadow: 0 22px 60px rgba(0,0,0,.28); }} svg {{ display: block; width: 100%; height: auto; min-height: 300px; }}
            .route {{ fill: none; stroke: #345c59; stroke-width: 4; stroke-linecap: round; stroke-dasharray: 7 12; }} .station circle {{ fill: #081717; stroke: #4c7974; stroke-width: 3; transition: fill .25s ease, stroke .25s ease, transform .25s ease; transform-origin: center; }}
            .station text {{ fill: #a1bbb5; font-size: 14px; text-anchor: middle; }} .station .index {{ fill: #d5ebe5; font-size: 13px; font-weight: 700; }} .station.active circle {{ fill: #155548; stroke: var(--accent); transform: scale(1.1); }} .station.active text {{ fill: var(--ink); }} .station.done circle {{ fill: #19433e; stroke: #69aa9c; }}
            .token {{ fill: var(--accent); stroke: #d9fff2; stroke-width: 3; filter: drop-shadow(0 0 12px rgba(113,224,189,.65)); }} .token-label {{ fill: #06231c; font-size: 10px; font-weight: 800; text-anchor: middle; }}
            .panel {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: end; gap: 18px; padding: 22px 24px 24px; border-top: 1px solid var(--line); }} .panel h2 {{ margin: 0 0 3px; font-size: 24px; letter-spacing: -.02em; }} .panel p {{ margin: 0; color: var(--muted); }} .metric {{ color: var(--accent); font-size: 13px; font-weight: 700; text-align: right; }}
            .controls {{ display: flex; gap: 8px; margin-top: 16px; }} button {{ border: 1px solid #46736d; border-radius: 999px; padding: 9px 15px; background: #123b38; color: var(--ink); font: inherit; cursor: pointer; }} button:hover, button:focus-visible {{ border-color: var(--accent); outline: 2px solid rgba(113,224,189,.35); outline-offset: 2px; }}
            .timeline {{ display: grid; grid-template-columns: repeat(3, minmax(120px, 1fr)); gap: 7px; margin-top: 18px; }} .timeline button {{ min-height: 60px; border-radius: 12px; padding: 8px; color: var(--muted); text-align: left; font-size: 12px; }} .timeline button[aria-current="step"] {{ border-color: var(--accent); background: #175149; color: var(--ink); }} .timeline strong {{ display: block; color: inherit; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }} .disclaimer {{ margin-top: 20px; color: var(--muted); font-size: 12px; }}
            @media (max-width: 760px) {{ .reel {{ padding: 28px 16px; }} .panel {{ grid-template-columns: 1fr; }} .metric {{ text-align: left; }} .timeline {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} svg {{ min-height: 420px; }}}} @media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ transition-duration: .01ms !important; animation-duration: .01ms !important; animation-iteration-count: 1 !important; }}}}
          </style>
        </head>
        <body><main class="reel" aria-labelledby="title"><div class="eyebrow">VisionTrain Studio · production reference reel</div><h1 id="title">Dataset gate first. Training evidence second.</h1><p class="lede">A real local image-folder preflight checks synthetic images, hardware, model compatibility, and the exact boundary before a trainer can run.</p><div class="model-line" aria-label="Model and data boundary"><span id="model-name"></span><span id="source-name"></span><span>synthetic dataset only</span></div>
          <section class="stage" aria-label="Animated training preflight sequence"><svg viewBox="0 0 1210 390" role="img" aria-labelledby="diagram-title diagram-description"><title id="diagram-title">Animated VisionTrain training stages</title><desc id="diagram-description">A token travels through dataset discovery, validation, hardware profiling, model compatibility, training gate, and export readiness.</desc><path class="route" d="M 80 190 C 220 82, 360 82, 510 190 S 800 298, 1130 100" /><g id="stations"></g><g aria-hidden="true"><circle id="token" class="token" cx="80" cy="190" r="22"></circle><text id="token-label" class="token-label" x="80" y="194">RUN</text></g></svg><div class="panel" aria-live="polite"><div><h2 id="event-title"></h2><p id="event-detail"></p></div><div id="event-metric" class="metric"></div></div></section>
          <div class="controls"><button id="toggle" type="button">Pause</button><button id="replay" type="button">Replay</button></div><div id="timeline" class="timeline" role="list" aria-label="Animation stages"></div><p class="disclaimer">Synthetic fixture · real preflight services · no fabricated loss, accuracy, or checkpoint evidence.</p></main>
          <script>
            const sequence = {sequence}; const events = sequence.events; const points = [[80,190],[270,96],[460,96],[650,190],[840,284],[1040,284]]; const stations = document.getElementById('stations'), timeline = document.getElementById('timeline'); const token = document.getElementById('token'), tokenLabel = document.getElementById('token-label'), title = document.getElementById('event-title'), detail = document.getElementById('event-detail'), metric = document.getElementById('event-metric'), toggle = document.getElementById('toggle'); let current = 0, started = performance.now(), playing = !window.matchMedia('(prefers-reduced-motion: reduce)').matches; const duration = 1850;
            points.forEach((point, index) => {{ const group = document.createElementNS('http://www.w3.org/2000/svg', 'g'); group.classList.add('station'); group.innerHTML = `<circle cx="${{point[0]}}" cy="${{point[1]}}" r="30"></circle><text class="index" x="${{point[0]}}" y="${{point[1] + 5}}">${{index + 1}}</text><text x="${{point[0]}}" y="${{point[1] + 57}}">${{events[index].short}}</text>`; stations.appendChild(group); }}); events.forEach((event, index) => {{ const item = document.createElement('button'); item.type = 'button'; item.setAttribute('role', 'listitem'); item.innerHTML = `<strong>${{String(index + 1).padStart(2, '0')}} · ${{event.status}}</strong>${{event.title}}`; item.addEventListener('click', () => {{ current = index; started = performance.now(); playing = false; toggle.textContent = 'Play'; render(0); }}); timeline.appendChild(item); }}); document.getElementById('model-name').textContent = `model ${{sequence.model.selected.name}}`; document.getElementById('source-name').textContent = `gate ${{sequence.model.compatibility.reason}}`;
            function render(progress) {{ const next = (current + 1) % points.length, from = points[current], to = points[next], x = from[0] + (to[0] - from[0]) * progress, y = from[1] + (to[1] - from[1]) * progress; token.setAttribute('cx', x); token.setAttribute('cy', y); tokenLabel.setAttribute('x', x); tokenLabel.setAttribute('y', y + 4); title.textContent = events[current].title; detail.textContent = events[current].detail; metric.textContent = events[current].metric; document.querySelectorAll('.station').forEach((node, index) => {{ node.classList.toggle('active', index === current); node.classList.toggle('done', index < current); }}); timeline.querySelectorAll('button').forEach((node, index) => node.setAttribute('aria-current', index === current ? 'step' : 'false')); }} function animate(now) {{ if (playing) {{ const progress = Math.min(1, (now - started) / duration); render(progress); if (progress >= 1) {{ current = (current + 1) % events.length; started = now; }} }} else {{ render(0); }} requestAnimationFrame(animate); }} toggle.addEventListener('click', () => {{ playing = !playing; started = performance.now(); toggle.textContent = playing ? 'Pause' : 'Play'; }}); document.getElementById('replay').addEventListener('click', () => {{ current = 0; started = performance.now(); playing = true; toggle.textContent = 'Pause'; }}); render(0); requestAnimationFrame(animate);
          </script></body></html>
        """
    )


def main() -> None:
    manifest = build_manifest()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    REEL.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    REEL.write_text(reel_html(manifest), encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    print(f"wrote {REEL.relative_to(ROOT)}")
    print(f"dataset valid: {manifest['dataset']['validation']['valid']}")
    print(f"model enabled: {manifest['model']['compatibility']['enabled']}")
    print(f"model gate: {manifest['model']['compatibility']['reason']}")


if __name__ == "__main__":
    main()
