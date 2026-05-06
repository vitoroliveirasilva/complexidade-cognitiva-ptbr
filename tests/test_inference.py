from __future__ import annotations

import pandas as pd

from complexidade_cognitiva_ptbr.models.bundle import load_model_bundle
from complexidade_cognitiva_ptbr.models.inference import predict_file, predict_text


def test_train_model_persists_bundle_and_latest_copy(
    app_settings, trained_model_artifacts
) -> None:
    assert trained_model_artifacts.best_bundle_path.exists()

    bundle = load_model_bundle(trained_model_artifacts.best_bundle_path)

    assert bundle.model_name == "logistic_regression"
    assert bundle.representation == "linguistic_metrics"
    assert set(bundle.classes) == {"alta", "baixa", "media"}
    assert bundle.feature_columns
    assert (
        app_settings.outputs.latest_dir / "models" / "best_model_bundle.joblib"
    ).exists()


def test_predict_text_uses_saved_bundle(app_settings, trained_model_artifacts) -> None:
    result = predict_text(
        app_settings,
        "A narrativa simbólica exige inferências complexas do leitor.",
        model_path=trained_model_artifacts.best_bundle_path,
    )

    assert result.predicted_class in {"alta", "baixa", "media"}
    assert result.linguistic_metrics["num_palavras"] > 0
    assert result.model_metadata["model_name"] == "logistic_regression"
    assert result.probabilities is not None
    assert set(result.probabilities) == {"alta", "baixa", "media"}


def test_predict_file_writes_prediction_csv(
    app_settings, trained_model_artifacts, tmp_path
) -> None:
    input_path = tmp_path / "novos_textos.csv"
    output_path = tmp_path / "predicoes.csv"
    pd.DataFrame(
        [
            {"id": "a", "texto": "O menino abriu a porta e sorriu."},
            {
                "id": "b",
                "texto": "A composição intertextual amplia camadas de sentido.",
            },
        ]
    ).to_csv(input_path, index=False, encoding="utf-8")

    saved_path = predict_file(
        app_settings,
        input_path=input_path,
        output_path=output_path,
        model_path=trained_model_artifacts.best_bundle_path,
    )

    assert saved_path == output_path.resolve()
    predictions = pd.read_csv(saved_path)
    assert len(predictions) == 2
    assert set(predictions["predicted_class"]).issubset({"alta", "baixa", "media"})
    assert "metric_num_palavras" in predictions.columns
