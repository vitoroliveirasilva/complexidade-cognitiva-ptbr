from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from complexidade_cognitiva_ptbr.data.preprocessing import (
    DatasetValidationError,
    build_preprocessed_dataset,
    clean_text,
    split_dataset,
    validate_dataset,
)


def test_clean_text_normalizes_spaces_line_breaks_and_control_chars() -> None:
    text = "  Linha   um.\x00\r\n\r\n\r\n  Linha\t dois.  "

    cleaned = clean_text(text)

    assert cleaned == "Linha um.\n\nLinha dois."


def test_clean_text_can_preserve_internal_spacing_when_requested() -> None:
    text = "  Linha   com\t espaços.\n  Outra linha.  "

    cleaned = clean_text(text, normalize_whitespace=False)

    assert cleaned == "Linha   com\t espaços.\n  Outra linha."


def test_validate_dataset_rejects_missing_required_column(app_settings) -> None:
    df = pd.DataFrame({"id": [1], "texto": ["Texto simples."]})

    with pytest.raises(DatasetValidationError, match="Colunas obrigatórias ausentes"):
        validate_dataset(df, app_settings)


def test_validate_dataset_rejects_blank_required_values(
    app_settings, sample_dataset
) -> None:
    df = sample_dataset.copy()
    df.loc[0, "texto"] = "   "

    with pytest.raises(DatasetValidationError, match=r"valor\(es\) vazio\(s\)"):
        validate_dataset(df, app_settings)


def test_validate_dataset_rejects_duplicate_ids(app_settings, sample_dataset) -> None:
    df = sample_dataset.copy()
    df.loc[1, "id"] = df.loc[0, "id"]

    with pytest.raises(DatasetValidationError, match="identificadores duplicados"):
        validate_dataset(df, app_settings)


def test_validate_dataset_reports_duplicate_text_warning(
    app_settings, sample_dataset
) -> None:
    df = sample_dataset.copy()
    df.loc[1, "texto"] = df.loc[0, "texto"]

    summary = validate_dataset(df, app_settings)

    assert summary["exact_duplicate_text_count"] == 2
    assert summary["exact_duplicate_text_examples"][0]["occurrences"] == 2
    assert summary["warnings"]


def test_build_preprocessed_dataset_adds_clean_text_column(
    app_settings, sample_dataset
) -> None:
    df = sample_dataset.head(3).copy()
    df.loc[0, "texto"] = "  Texto   com   espaços.  "

    prepared, summary = build_preprocessed_dataset(df, app_settings)

    assert "texto_limpo" in prepared.columns
    assert prepared.loc[0, "texto_limpo"] == "Texto com espaços."
    assert summary["changed_rows"] == 1
    assert summary["empty_after_cleaning"] == 0


def test_build_preprocessed_dataset_can_replace_original_text(
    make_settings, app_settings, sample_dataset
) -> None:
    preprocessing = replace(app_settings.preprocessing, preserve_original_text=False)
    settings = make_settings(preprocessing=preprocessing)
    df = sample_dataset.head(3).copy()
    df.loc[0, "texto"] = "  Texto   com   espaços.  "

    prepared, _ = build_preprocessed_dataset(df, settings)

    assert prepared.loc[0, "texto"] == "Texto com espaços."
    assert prepared.loc[0, "texto_limpo"] == "Texto com espaços."


def test_build_preprocessed_dataset_rejects_texts_empty_after_cleaning(
    app_settings, sample_dataset
) -> None:
    df = sample_dataset.head(3).copy()
    df.loc[0, "texto"] = "\x00\x01\x02"

    with pytest.raises(DatasetValidationError, match="ficaram vazios"):
        build_preprocessed_dataset(df, app_settings)


def test_split_dataset_creates_train_val_test(app_settings, sample_dataset) -> None:
    prepared, _ = build_preprocessed_dataset(sample_dataset, app_settings)

    result = split_dataset(prepared, app_settings)

    assert set(result) == {"train", "val", "test", "metadata"}
    assert len(result["train"]) > len(result["val"]) > 0
    assert len(result["test"]) > 0
    assert result["metadata"]["stratified_used"] is True
    assert sum(result["metadata"]["actual_counts"].values()) == len(sample_dataset)
