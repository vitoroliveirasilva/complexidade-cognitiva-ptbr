from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from complexidade_cognitiva_ptbr.config.settings import (
    SettingsError,
    load_settings,
)
from complexidade_cognitiva_ptbr.utils.paths import (
    ensure_dir,
    ensure_project_directories,
    find_project_root,
    relative_to_root,
)


def test_managed_directories_are_created(app_settings) -> None:
    directories = ensure_project_directories(app_settings)

    assert len(directories) == 9
    assert all(directory.exists() and directory.is_dir() for directory in directories)


def test_ensure_dir_rejects_existing_file(tmp_path: Path) -> None:
    file_path = tmp_path / "arquivo.txt"
    file_path.write_text("conteúdo", encoding="utf-8")

    with pytest.raises(RuntimeError, match="não é um diretório"):
        ensure_dir(file_path)


def test_relative_to_root_returns_posix_relative_path(app_settings) -> None:
    path = app_settings.outputs.metrics_dir / "final_metrics.json"

    assert (
        relative_to_root(path, app_settings.project_root)
        == "outputs/metrics/final_metrics.json"
    )


def test_find_project_root_uses_project_markers(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    nested = root / "src" / "complexidade_cognitiva_ptbr" / "data"
    nested.mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")

    assert find_project_root(nested) == root


def test_load_settings_parses_valid_yaml(tmp_path: Path) -> None:
    root = tmp_path
    (root / "configs").mkdir()
    (root / "src").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    config_path = root / "configs" / "config.yaml"
    config_path.write_text(
        dedent("""
            project:
              name: complexidade-cognitiva-ptbr
              version: 0.1.0
              description: Teste
              language: pt-BR
              random_state: 42
            dataset:
              input_path: data/raw/dataset.csv
              id_column: id
              text_column: texto
              target_column: target
              split:
                train_size: 0.7
                val_size: 0.15
                test_size: 0.15
                stratify: true
            preprocessing:
              preserve_original_text: true
              clean_text_column: texto_limpo
              normalize_whitespace: true
            features:
              output_dir: data/processed/features
              long_word_min_chars: 7
              tfidf:
                word:
                  enabled: true
                  max_features: 200
                  ngram_range: [1, 2]
                  min_df: 1
                char:
                  enabled: true
                  max_features: 200
                  ngram_range: [3, 5]
                  min_df: 1
            training:
              models: [logistic_regression]
              representations: [linguistic_metrics]
              selection_metric: f1_macro
            outputs:
              processed_dir: data/processed
              model_dir: outputs/models
              metrics_dir: outputs/metrics
              figures_dir: outputs/figures
              reports_dir: outputs/reports
            logging:
              level: INFO
              format: text
            """).strip() + "\n",
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.project_root == root
    assert settings.dataset.input_path == root / "data" / "raw" / "dataset.csv"
    assert settings.training.models == ("logistic_regression",)
    assert settings.cross_validation.enabled is True
    assert settings.outputs.runs_dir == root / "outputs" / "runs"
    assert settings.outputs.latest_dir == root / "outputs" / "latest"


def test_load_settings_rejects_duplicate_required_dataset_columns(
    tmp_path: Path,
) -> None:
    root = tmp_path
    (root / "configs").mkdir()
    (root / "src").mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    config_path = root / "configs" / "config.yaml"
    config_path.write_text(
        dedent("""
            project:
              name: complexidade-cognitiva-ptbr
              version: 0.1.0
              description: Teste
              language: pt-BR
              random_state: 42
            dataset:
              input_path: data/raw/dataset.csv
              id_column: id
              text_column: texto
              target_column: texto
              split:
                train_size: 0.7
                val_size: 0.15
                test_size: 0.15
                stratify: true
            preprocessing:
              preserve_original_text: true
              clean_text_column: texto_limpo
              normalize_whitespace: true
            features:
              output_dir: data/processed/features
              long_word_min_chars: 7
              tfidf:
                word: {enabled: true, max_features: 200, ngram_range: [1, 2], min_df: 1}
                char: {enabled: true, max_features: 200, ngram_range: [3, 5], min_df: 1}
            training:
              models: [logistic_regression]
              representations: [linguistic_metrics]
              selection_metric: f1_macro
            outputs:
              processed_dir: data/processed
              model_dir: outputs/models
              metrics_dir: outputs/metrics
              figures_dir: outputs/figures
              reports_dir: outputs/reports
            logging:
              level: INFO
              format: text
            """).strip() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="distintas|distintos"):
        load_settings(config_path)
