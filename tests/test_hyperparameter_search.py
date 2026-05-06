from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from complexidade_cognitiva_ptbr.models.search import (
    HyperparameterSearchError,
    build_classifier_param_grid,
    run_hyperparameter_search_for_experiment,
)
from complexidade_cognitiva_ptbr.models.train import (
    build_training_pipeline,
    load_training_frames,
    train_model,
    validate_training_frames,
)


def test_build_classifier_param_grid_adds_classifier_prefix(
    app_settings, feature_datasets
) -> None:
    train_df, val_df = load_training_frames(app_settings)
    feature_columns = validate_training_frames(train_df, val_df, app_settings)
    pipeline = build_training_pipeline(
        settings=app_settings,
        representation="linguistic_metrics",
        model_name="logistic_regression",
        feature_columns=feature_columns,
        text_column=app_settings.preprocessing.clean_text_column,
    )

    grid = build_classifier_param_grid(app_settings, pipeline, "logistic_regression")

    assert "classifier__C" in grid
    assert grid["classifier__C"] == (1.0,)


def test_hyperparameter_search_returns_tuned_estimator(
    app_settings, feature_datasets
) -> None:
    train_df, val_df = load_training_frames(app_settings)
    feature_columns = validate_training_frames(train_df, val_df, app_settings)
    pipeline = build_training_pipeline(
        settings=app_settings,
        representation="linguistic_metrics",
        model_name="logistic_regression",
        feature_columns=feature_columns,
        text_column=app_settings.preprocessing.clean_text_column,
    )

    outcome = run_hyperparameter_search_for_experiment(
        settings=app_settings,
        pipeline=pipeline,
        train_df=train_df,
        model_name="logistic_regression",
        representation="linguistic_metrics",
        target_column=app_settings.dataset.target_column,
        text_column=app_settings.preprocessing.clean_text_column,
        feature_columns=feature_columns,
    )

    assert outcome.enabled is True
    assert outcome.best_params["C"] == 1.0
    assert outcome.cv_results
    assert callable(getattr(outcome.best_estimator, "predict", None))


def test_hyperparameter_search_rejects_invalid_space(
    make_settings, app_settings, feature_datasets
) -> None:
    search = replace(
        app_settings.hyperparameter_search,
        search_spaces={"logistic_regression": {"parametro_inexistente": (1, 2)}},
    )
    settings = make_settings(hyperparameter_search=search)
    train_df, val_df = load_training_frames(settings)
    feature_columns = validate_training_frames(train_df, val_df, settings)
    pipeline = build_training_pipeline(
        settings=settings,
        representation="linguistic_metrics",
        model_name="logistic_regression",
        feature_columns=feature_columns,
        text_column=settings.preprocessing.clean_text_column,
    )

    with pytest.raises(HyperparameterSearchError, match="Nenhum parâmetro"):
        build_classifier_param_grid(settings, pipeline, "logistic_regression")


def test_train_model_persists_hyperparameter_artifacts(
    app_settings, feature_datasets
) -> None:
    result = train_model(app_settings)

    assert result.hyperparameter_results_path.exists()
    assert result.best_hyperparameters_path.exists()
    assert result.hyperparameter_report_path.exists()

    payload = json.loads(result.best_hyperparameters_path.read_text(encoding="utf-8"))
    assert payload["enabled"] is True
    assert payload["best_experiment"]["model_name"] == "logistic_regression"

    results_df = pd.read_csv(result.hyperparameter_results_path)
    assert not results_df.empty
    assert "mean_test_score" in results_df.columns
