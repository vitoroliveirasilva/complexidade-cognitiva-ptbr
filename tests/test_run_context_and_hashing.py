from __future__ import annotations

import json
from pathlib import Path

from complexidade_cognitiva_ptbr.utils.hashing import build_file_fingerprint, hash_file
from complexidade_cognitiva_ptbr.utils.run_context import RunContext


def test_hash_file_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "dataset.csv"
    path.write_text("id,texto,target\n1,Texto,baixa\n", encoding="utf-8")

    first_hash = hash_file(path)
    second_hash = hash_file(path)

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_build_file_fingerprint_does_not_require_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "ausente.csv"

    fingerprint = build_file_fingerprint(path)

    assert fingerprint.exists is False
    assert fingerprint.sha256 is None
    assert fingerprint.size_bytes is None


def test_run_context_creates_versioned_metadata(app_settings, raw_dataset_file) -> None:
    app_settings.config_path.parent.mkdir(parents=True, exist_ok=True)
    app_settings.config_path.write_text("project:\n  name: teste\n", encoding="utf-8")

    context = RunContext.create(app_settings)

    assert context.run_dir.exists()
    assert context.models_dir.exists()
    assert context.metrics_dir.exists()
    assert context.figures_dir.exists()
    assert context.reports_dir.exists()
    assert context.explainability_dir.exists()
    assert context.config_snapshot_path.exists()
    assert context.environment_path.exists()
    assert context.data_fingerprint_path.exists()
    assert context.execution_log_path.exists()

    fingerprint = json.loads(context.data_fingerprint_path.read_text(encoding="utf-8"))
    assert fingerprint["dataset"]["exists"] is True
    assert fingerprint["dataset"]["sha256"]
    assert fingerprint["dataset"]["path"] == "data/raw/dataset.csv"

    pointer = context.update_latest_pointer()
    assert pointer.read_text(encoding="utf-8").strip() == context.run_id
