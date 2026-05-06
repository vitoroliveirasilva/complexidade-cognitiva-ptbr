from __future__ import annotations

import json
import joblib
import pandas as pd

from complexidade_cognitiva_ptbr.config.settings import TrainingConfig
from complexidade_cognitiva_ptbr.evaluation.explainability import (
    generate_explainability_artifacts,
)
from complexidade_cognitiva_ptbr.evaluation.reporting import load_test_frame
from complexidade_cognitiva_ptbr.models.train import prepare_model_frame, train_model


def test_generate_tfidf_explainability_artifacts_for_linear_model(
    make_settings,
    app_settings,
    feature_datasets,
) -> None:
    training = TrainingConfig(
        models=("logistic_regression",),
        representations=("tfidf_word",),
        selection_metric="f1_macro",
    )
    settings = make_settings(training=training)
    training_result = train_model(settings)
    model = joblib.load(training_result.best_model_path)
    best_experiment = json.loads(
        training_result.best_experiment_path.read_text(encoding="utf-8")
    )
    test_df = load_test_frame(settings)
    x_test = prepare_model_frame(
        test_df,
        settings.dataset.target_column,
        settings.preprocessing.clean_text_column,
        [],
    )
    y_pred = [str(value) for value in model.predict(x_test)]

    result = generate_explainability_artifacts(
        settings=settings,
        model=model,
        best_experiment=best_experiment,
        test_df=test_df,
        y_pred=y_pred,
        explainability_dir=settings.outputs.latest_dir / "explainability-test",
        figures_dir=settings.outputs.figures_dir,
    )

    assert result.summary["status"] == "supported"
    assert result.top_terms_csv_path.exists()
    assert result.top_terms_json_path.exists()
    assert result.report_path.exists()
    assert result.local_explanations_path.exists()
    assert result.top_terms_figure_paths

    top_terms = pd.read_csv(result.top_terms_csv_path)
    local = pd.read_csv(result.local_explanations_path)
    report_text = result.report_path.read_text(encoding="utf-8")

    assert {"class", "direction", "term", "coefficient"}.issubset(top_terms.columns)
    assert not top_terms.empty
    assert not local.empty
    assert "# Relatório de explicabilidade TF-IDF" in report_text


def test_generate_explainability_records_unsupported_model_without_breaking(
    app_settings,
    trained_model_artifacts,
) -> None:
    model = joblib.load(trained_model_artifacts.best_model_path)
    best_experiment = json.loads(
        trained_model_artifacts.best_experiment_path.read_text(encoding="utf-8")
    )
    test_df = load_test_frame(app_settings)

    result = generate_explainability_artifacts(
        settings=app_settings,
        model=model,
        best_experiment=best_experiment,
        test_df=test_df,
        explainability_dir=app_settings.outputs.latest_dir
        / "explainability-unsupported",
        figures_dir=app_settings.outputs.figures_dir,
    )

    assert result.summary["status"] == "unsupported"
    assert result.top_terms_csv_path.exists()
    assert result.report_path.exists()
    assert not result.top_terms_figure_paths
