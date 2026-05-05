from __future__ import annotations

import json

import pandas as pd

from ..src.complexidade_cognitiva_ptbr.data.preprocessing import prepare_dataset


def test_prepare_dataset_writes_splits_and_report(
    app_settings, raw_dataset_file
) -> None:
    result = prepare_dataset(app_settings)

    assert result.train_path.exists()
    assert result.val_path.exists()
    assert result.test_path.exists()
    assert result.report_path.exists()

    train_df = pd.read_csv(result.train_path)
    val_df = pd.read_csv(result.val_path)
    test_df = pd.read_csv(result.test_path)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert len(train_df) + len(val_df) + len(test_df) == 30
    assert "texto_limpo" in train_df.columns
    assert report["split"]["actual_counts"]["train"] == len(train_df)
    assert report["validation"]["class_distribution"] == {
        "alta": 10,
        "baixa": 10,
        "media": 10,
    }
    assert (
        result.to_dict(app_settings.project_root)["train_path"]
        == "data/processed/train.csv"
    )
