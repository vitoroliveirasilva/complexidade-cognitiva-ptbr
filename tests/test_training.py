from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from ..src.complexidade_cognitiva_ptbr.features.build_features import get_feature_columns
from ..src.complexidade_cognitiva_ptbr.models.train import (
    TrainingError,
    compute_classification_metrics,
    merge_prepared_and_features,
    prepare_model_frame,
    select_best_experiment,
    train_model,
    validate_training_configuration,
)


def test_compute_classification_metrics_returns_required_metrics() -> None:
    metrics = compute_classification_metrics(
        y_true=["baixa", "media", "alta", "alta"],
        y_pred=["baixa", "media", "media", "alta"],
    )

    assert set(metrics) == {
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "balanced_accuracy",
    }
    assert metrics["accuracy"] == 0.75


def test_compute_classification_metrics_rejects_incompatible_lengths() -> None:
    with pytest.raises(TrainingError, match="incompatível"):
        compute_classification_metrics(["baixa"], [])


def test_select_best_experiment_uses_stable_tie_breaker() -> None:
    experiments = [
        {
            "experiment_id": "first",
            "validation_metrics": {
                "f1_macro": 0.8,
                "f1_weighted": 0.7,
                "accuracy": 0.8,
            },
        },
        {
            "experiment_id": "second",
            "validation_metrics": {
                "f1_macro": 0.8,
                "f1_weighted": 0.7,
                "accuracy": 0.8,
            },
        },
    ]

    best = select_best_experiment(experiments, "f1_macro")

    assert best["experiment_id"] == "first"


def test_select_best_experiment_rejects_missing_metric() -> None:
    with pytest.raises(TrainingError, match="Métrica de seleção"):
        select_best_experiment(
            [{"experiment_id": "x", "validation_metrics": {"accuracy": 1.0}}],
            "f1_macro",
        )


def test_validate_training_configuration_rejects_unsupported_model(
    make_settings, app_settings
) -> None:
    training = replace(app_settings.training, models=("modelo_inexistente",))
    settings = make_settings(training=training)

    with pytest.raises(TrainingError, match="não suportado"):
        validate_training_configuration(settings)


def test_merge_prepared_and_features_rejects_mismatched_rows(
    app_settings, sample_dataset
) -> None:
    prepared = sample_dataset.head(3).copy()
    prepared["texto_limpo"] = prepared["texto"]
    feature_rows = []
    for _, row in prepared.head(2).iterrows():
        feature_rows.append(
            {
                "id": row["id"],
                "target": row["target"],
                **dict.fromkeys(get_feature_columns(), 0),
            }
        )
    features = pd.DataFrame(feature_rows)

    with pytest.raises(TrainingError, match="Falha ao combinar"):
        merge_prepared_and_features(prepared, features, app_settings, "train")


def test_prepare_model_frame_coerces_text_and_numeric_features(
    app_settings, sample_dataset
) -> None:
    frame = sample_dataset.head(2).copy()
    frame["texto_limpo"] = frame["texto"]
    for column in get_feature_columns():
        frame[column] = "1"

    model_frame = prepare_model_frame(
        frame,
        target_column=app_settings.dataset.target_column,
        text_column=app_settings.preprocessing.clean_text_column,
        feature_columns=get_feature_columns(),
    )

    assert app_settings.dataset.target_column not in model_frame.columns
    assert (
        model_frame[app_settings.preprocessing.clean_text_column].tolist()
        == frame["texto_limpo"].tolist()
    )
    assert all(
        pd.api.types.is_numeric_dtype(model_frame[column])
        for column in get_feature_columns()
    )


def test_train_model_persists_best_model_and_experiment(
    app_settings, feature_datasets
) -> None:
    result = train_model(app_settings)

    assert result.best_model_path.exists()
    assert result.best_experiment_path.exists()
    assert result.experiment_results_path.exists()

    payload = json.loads(result.best_experiment_path.read_text(encoding="utf-8"))
    assert payload["best_experiment"]["model_name"] == "logistic_regression"
    assert payload["best_experiment"]["representation"] == "linguistic_metrics"
    assert payload["selection"]["selection_metric"] == "f1_macro"
    assert (
        result.to_dict(app_settings.project_root)["best_model_path"]
        == "outputs/models/best_model.joblib"
    )
