from __future__ import annotations

from pathlib import Path

from complexidade_cognitiva_ptbr.utils.artifact_validation import (
    ArtifactValidationError,
    resolve_latest_run_dir,
    validate_artifacts,
)

SMOKE_REQUIRED_FILES = (
    "config_snapshot.yaml",
    "environment.json",
    "data_fingerprint.json",
    "run_manifest.json",
    "metrics/leakage_report.json",
    "metrics/cv_summary.json",
    "models/best_model_bundle.joblib",
    "models/best_experiment.json",
    "metrics/final_metrics.json",
    "reports/final_report_extended.md",
)


def _write_artifacts(base_dir: Path, relative_paths: tuple[str, ...]) -> None:
    for relative_path in relative_paths:
        path = base_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n", encoding="utf-8")


def test_validate_artifacts_smoke_profile_passes(app_settings) -> None:
    run_dir = app_settings.outputs.runs_dir / "20260101_000000"
    _write_artifacts(run_dir, SMOKE_REQUIRED_FILES)

    result = validate_artifacts(app_settings, run_dir=run_dir, profile="smoke")

    assert result.passed is True
    assert result.required_failures == ()
    assert (
        result.to_dict(app_settings.project_root)["summary"]["required_failures"] == 0
    )


def test_validate_artifacts_reports_missing_required_file(app_settings) -> None:
    run_dir = app_settings.outputs.runs_dir / "20260101_000000"
    _write_artifacts(
        run_dir,
        tuple(
            path
            for path in SMOKE_REQUIRED_FILES
            if path != "metrics/final_metrics.json"
        ),
    )

    result = validate_artifacts(app_settings, run_dir=run_dir, profile="smoke")

    assert result.passed is False
    assert any(
        check.relative_path == "metrics/final_metrics.json"
        for check in result.required_failures
    )


def test_resolve_latest_run_dir_uses_pointer(app_settings) -> None:
    run_dir = app_settings.outputs.runs_dir / "20260101_000000"
    run_dir.mkdir(parents=True, exist_ok=True)
    app_settings.outputs.latest_dir.mkdir(parents=True, exist_ok=True)
    (app_settings.outputs.latest_dir / "run_id.txt").write_text(
        "20260101_000000\n", encoding="utf-8"
    )

    assert resolve_latest_run_dir(app_settings) == run_dir


def test_resolve_latest_run_dir_fails_without_pointer(app_settings) -> None:
    try:
        resolve_latest_run_dir(app_settings)
    except ArtifactValidationError as exc:
        assert "Ponteiro" in str(exc)
    else:
        raise AssertionError("Era esperado erro quando o ponteiro latest não existe.")
