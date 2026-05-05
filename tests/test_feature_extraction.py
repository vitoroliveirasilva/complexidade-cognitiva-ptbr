from __future__ import annotations

import pandas as pd
import pytest

from complexidade_cognitiva_ptbr.features.build_features import (
    FeatureBuildError,
    build_feature_datasets,
    build_features_dataframe,
    extract_linguistic_features,
    get_feature_columns,
    tokenize_sentences,
    tokenize_words,
)


def test_tokenizers_handle_portuguese_accents_and_apostrophes() -> None:
    text = "Embora cansada, a criança observou a árvore. Depois, d'água sorriu!"

    assert tokenize_sentences(text) == [
        "Embora cansada, a criança observou a árvore.",
        "Depois, d'água sorriu!",
    ]
    assert "criança" in tokenize_words(text)
    assert "árvore" in tokenize_words(text)
    assert "d'água" in tokenize_words(text)


def test_extract_linguistic_features_returns_expected_columns() -> None:
    features = extract_linguistic_features(
        "Embora a narrativa seja breve, ela possui palavras complexas e símbolos.",
        long_word_min_chars=7,
    )

    assert list(features.keys()) == get_feature_columns()
    assert features["num_palavras"] > 0
    assert features["num_sentencas"] == 1
    assert features["frequencia_marcadores_subordinacao"] > 0
    assert 0 <= features["type_token_ratio"] <= 1


def test_extract_linguistic_features_rejects_invalid_long_word_threshold() -> None:
    with pytest.raises(FeatureBuildError, match="long_word_min_chars"):
        extract_linguistic_features("Texto válido.", long_word_min_chars=0)


def test_extract_linguistic_features_handles_empty_text_without_nan() -> None:
    features = extract_linguistic_features("   ")

    assert list(features.keys()) == get_feature_columns()
    assert features["num_palavras"] == 0
    assert features["num_sentencas"] == 0
    assert all(value == value for value in features.values())


def test_build_features_dataframe_preserves_id_target_and_feature_order(
    app_settings, sample_dataset
) -> None:
    df = sample_dataset.head(3).copy()
    df["texto_limpo"] = df["texto"]

    features_df = build_features_dataframe(df, app_settings, split_name="train")

    assert list(features_df.columns) == ["id", "target", *get_feature_columns()]
    assert len(features_df) == 3
    assert features_df[get_feature_columns()].isna().sum().sum() == 0


def test_build_features_dataframe_rejects_duplicate_ids(
    app_settings, sample_dataset
) -> None:
    df = sample_dataset.head(3).copy()
    df["texto_limpo"] = df["texto"]
    df.loc[1, "id"] = df.loc[0, "id"]

    with pytest.raises(FeatureBuildError, match="duplicada"):
        build_features_dataframe(df, app_settings, split_name="train")


def test_build_feature_datasets_writes_feature_artifacts(
    app_settings, prepared_dataset
) -> None:
    result = build_feature_datasets(app_settings)

    assert result.train_features_path.exists()
    assert result.val_features_path.exists()
    assert result.test_features_path.exists()
    assert result.metadata_path.exists()

    train_features = pd.read_csv(result.train_features_path)
    assert ["id", "target", *get_feature_columns()] == list(train_features.columns)
    assert result.metadata["splits"]["train"]["feature_count"] == len(
        get_feature_columns()
    )
