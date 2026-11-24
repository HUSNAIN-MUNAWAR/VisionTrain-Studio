import json

import pytest

from backend.trainer import experiment_store


def test_experiment_index_is_written_atomically(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    monkeypatch.setattr(experiment_store, "RUNS_DIR", runs)
    monkeypatch.setattr(experiment_store, "INDEX_PATH", runs / "experiments.json")

    record = experiment_store.record_experiment({"model": {"id": "m1"}}, "completed")

    assert record["status"] == "completed"
    assert json.loads((runs / "experiments.json").read_text()) == [record]
    assert list(runs.glob("*.tmp")) == []


def test_malformed_experiment_index_is_not_silently_reset(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    index = runs / "experiments.json"
    index.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(experiment_store, "RUNS_DIR", runs)
    monkeypatch.setattr(experiment_store, "INDEX_PATH", index)

    with pytest.raises(RuntimeError, match="cannot be read safely"):
        experiment_store.list_experiments()

    assert index.read_text(encoding="utf-8") == "{not-json"
