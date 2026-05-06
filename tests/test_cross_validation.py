from __future__ import annotations

import json
import pandas as pd
import pytest

from complexidade_cognitiva_ptbr.config.settings import CrossValidationConfig
from complexidade_cognitiva_ptbr.evaluation.cross_validation import (
    CrossValidationError,
    build_cv_summary,
    run_cross_validation,
)


def test_run_cross_validation_generates_artifacts(
    app_settings, feature_datasets
) -> None:
    result = run_cross_validation(app_settings)

    assert result.results_path.exists()
    assert result.summary_path.exists()
    assert result.report_path.exists()
    assert result.figure_path.exists()

    results_df = pd.read_csv(result.results_path)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert not results_df.empty
    assert set(results_df["fold"]) == {1, 2, 3}
    assert summary["summary"]["total_experiments"] == 1
    assert (
        summary["summary"]["best_experiment_id"]
        == "linguistic_metrics__logistic_regression"
    )
    assert (
        result.to_dict(app_settings.project_root)["results_path"]
        == "outputs/metrics/cv_results.csv"
    )


def test_build_cv_summary_computes_aggregate_statistics(app_settings) -> None:
    results_df = pd.DataFrame(
        [
            {
                "experiment_id": "exp",
                "representation": "linguistic_metrics",
                "model_name": "logistic_regression",
                "fold": 1,
                "fit_seconds": 0.1,
                "accuracy": 0.5,
                "f1_macro": 0.4,
            },
            {
                "experiment_id": "exp",
                "representation": "linguistic_metrics",
                "model_name": "logistic_regression",
                "fold": 2,
                "fit_seconds": 0.2,
                "accuracy": 1.0,
                "f1_macro": 0.8,
            },
        ]
    )

    summary = build_cv_summary(results_df, app_settings)

    metrics = summary["experiments"][0]["metrics"]
    assert metrics["accuracy"]["mean"] == 0.75
    assert metrics["f1_macro"]["min"] == 0.4
    assert metrics["f1_macro"]["max"] == 0.8


def test_run_cross_validation_rejects_n_splits_larger_than_smallest_class(
    make_settings, app_settings, feature_datasets
) -> None:
    cv = CrossValidationConfig(
        enabled=True,
        method="stratified_kfold",
        n_splits=20,
        n_repeats=1,
        shuffle=True,
        random_state=42,
        scoring=("accuracy", "f1_macro"),
    )
    settings = make_settings(cross_validation=cv)

    with pytest.raises(CrossValidationError, match="n_splits"):
        run_cross_validation(settings)
