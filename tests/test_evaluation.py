from __future__ import annotations

import json

import pandas as pd
import pytest

from ..src.complexidade_cognitiva_ptbr.evaluation.reporting import (
    EvaluationError,
    build_predictions_dataframe,
    compute_final_metrics,
    evaluate_model,
    load_best_experiment,
    validate_test_frame,
)


def test_compute_final_metrics_returns_required_metrics() -> None:
    metrics = compute_final_metrics(
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


def test_compute_final_metrics_rejects_empty_vectors() -> None:
    with pytest.raises(EvaluationError, match="rótulos reais"):
        compute_final_metrics([], [])


def test_build_predictions_dataframe_preserves_configured_text_columns(
    app_settings, sample_dataset
) -> None:
    test_df = sample_dataset.head(2).copy()
    test_df["texto_limpo"] = test_df["texto"].str.lower()

    output = build_predictions_dataframe(
        test_df=test_df,
        y_true=["baixa", "baixa"],
        y_pred=["baixa", "media"],
        id_column=app_settings.dataset.id_column,
        target_column=app_settings.dataset.target_column,
        extra_text_columns=(
            app_settings.preprocessing.clean_text_column,
            app_settings.dataset.text_column,
        ),
    )

    assert list(output.columns) == [
        "id",
        "target",
        "predicted_target",
        "correct",
        "texto_limpo",
        "texto",
    ]
    assert output["correct"].tolist() == [True, False]


def test_validate_test_frame_rejects_duplicate_ids(
    app_settings, sample_dataset
) -> None:
    test_df = sample_dataset.head(3).copy()
    test_df.loc[1, "id"] = test_df.loc[0, "id"]

    with pytest.raises(EvaluationError, match="identificadores duplicados"):
        validate_test_frame(test_df, app_settings)


def test_load_best_experiment_rejects_invalid_payload(app_settings) -> None:
    app_settings.outputs.model_dir.mkdir(parents=True, exist_ok=True)
    path = app_settings.outputs.model_dir / "best_experiment.json"
    path.write_text(
        json.dumps({"best_experiment": {}, "selection": {}}), encoding="utf-8"
    )

    with pytest.raises(EvaluationError, match="Estrutura obrigatória|incompleto"):
        load_best_experiment(app_settings)


def test_evaluate_model_generates_final_artifacts(
    app_settings, trained_model_artifacts
) -> None:
    result = evaluate_model(app_settings)

    assert result.final_metrics_path.exists()
    assert result.classification_report_json_path.exists()
    assert result.classification_report_txt_path.exists()
    assert result.test_predictions_path.exists()
    assert result.confusion_matrix_path.exists()
    assert result.final_report_path.exists()

    final_metrics = json.loads(result.final_metrics_path.read_text(encoding="utf-8"))
    predictions_df = pd.read_csv(result.test_predictions_path)
    report_text = result.final_report_path.read_text(encoding="utf-8")

    assert final_metrics["evaluation_dataset"]["split"] == "test"
    assert not predictions_df.empty
    assert {"id", "target", "predicted_target", "correct"}.issubset(
        predictions_df.columns
    )
    assert "# Relatório final de avaliação" in report_text
    assert (
        result.to_dict(app_settings.project_root)["final_metrics_path"]
        == "outputs/metrics/final_metrics.json"
    )
