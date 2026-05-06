from __future__ import annotations

import json

import pandas as pd

from complexidade_cognitiva_ptbr.evaluation.visualization import (
    save_confusion_matrix_plot,
    save_cv_summary_chart,
    save_experiment_comparison_chart,
    save_feature_distribution_charts,
)


def test_save_confusion_matrix_plot_supports_normalized_output(tmp_path) -> None:
    output_path = save_confusion_matrix_plot(
        y_true=["baixa", "media", "alta", "alta"],
        y_pred=["baixa", "media", "media", "alta"],
        labels=["alta", "baixa", "media"],
        output_path=tmp_path / "confusion_matrix_normalized.png",
        normalized=True,
        dpi=100,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_experiment_and_cv_charts(tmp_path) -> None:
    experiment_results = tmp_path / "experiment_results.csv"
    pd.DataFrame(
        [
            {"experiment_id": "a", "validation_f1_macro": 0.70},
            {"experiment_id": "b", "validation_f1_macro": 0.90},
        ]
    ).to_csv(experiment_results, index=False, encoding="utf-8")
    cv_summary = tmp_path / "cv_summary.json"
    cv_summary.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "experiment_id": "a",
                        "metrics": {"f1_macro": {"mean": 0.7, "std": 0.1}},
                    },
                    {
                        "experiment_id": "b",
                        "metrics": {"f1_macro": {"mean": 0.9, "std": 0.05}},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    experiment_chart = save_experiment_comparison_chart(
        experiment_results_path=experiment_results,
        output_path=tmp_path / "experiment_comparison_f1_macro.png",
        metric="f1_macro",
        dpi=100,
    )
    cv_chart = save_cv_summary_chart(
        cv_summary_path=cv_summary,
        output_path=tmp_path / "cv_summary_f1_macro.png",
        metric="f1_macro",
        dpi=100,
    )

    assert experiment_chart is not None and experiment_chart.exists()
    assert cv_chart is not None and cv_chart.exists()


def test_save_feature_distribution_charts(
    app_settings, sample_dataset, tmp_path
) -> None:
    frame = sample_dataset.head(9).copy()
    frame["num_palavras"] = [5, 6, 5, 7, 5, 6, 8, 6, 7]
    frame["media_palavras_por_sentenca"] = frame["num_palavras"]
    frame["type_token_ratio"] = 0.8
    frame["razao_palavras_longas"] = 0.2
    frame["densidade_lexical_aproximada"] = 0.6

    paths = save_feature_distribution_charts(
        test_df=frame,
        settings=app_settings,
        figures_dir=tmp_path,
    )

    assert paths
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
