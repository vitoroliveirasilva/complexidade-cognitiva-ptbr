from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, variance
from typing import Any

import pandas as pd

from ..config.settings import AppSettings
from ..data.io import read_csv_dataset, write_csv_dataset, write_json
from ..utils.paths import relative_to_root

_SENTENCE_PATTERN = re.compile(r"[^.!?…]+[.!?…]*", flags=re.UNICODE)
_WORD_PATTERN = re.compile(
    r"\b[0-9A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'’][0-9A-Za-zÀ-ÖØ-öø-ÿ]+)*\b",
    flags=re.UNICODE,
)
_NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_PUNCTUATION_CHARS = frozenset(string.punctuation + "“”‘’—–…«»")

_CONNECTIVES = frozenset(
    {
        "ademais",
        "afinal",
        "ainda",
        "assim",
        "contudo",
        "entretanto",
        "logo",
        "mas",
        "nem",
        "ou",
        "pois",
        "porém",
        "portanto",
        "porque",
        "quando",
        "também",
        "todavia",
        "e",
    }
)

_SUBORDINATION_MARKERS = frozenset(
    {
        "caso",
        "conforme",
        "conquanto",
        "embora",
        "enquanto",
        "onde",
        "para",
        "porque",
        "quando",
        "quanto",
        "que",
        "se",
        "segundo",
    }
)

_APPROXIMATE_FUNCTION_WORDS = frozenset(
    {
        "a",
        "à",
        "ao",
        "aos",
        "as",
        "às",
        "com",
        "da",
        "das",
        "de",
        "do",
        "dos",
        "e",
        "em",
        "entre",
        "esse",
        "essa",
        "isso",
        "na",
        "nas",
        "no",
        "nos",
        "o",
        "os",
        "ou",
        "para",
        "por",
        "que",
        "se",
        "sem",
        "sob",
        "sobre",
        "um",
        "uma",
        "uns",
        "umas",
    }
)

_FEATURE_DESCRIPTIONS = {
    "num_caracteres": "Quantidade total de caracteres do texto limpo.",
    "num_palavras": "Quantidade total de tokens lexicais identificados no texto.",
    "num_sentencas": "Quantidade aproximada de sentenças identificadas por pontuação final.",
    "media_palavras_por_sentenca": "Média de palavras por sentença.",
    "media_caracteres_por_palavra": "Média de caracteres por palavra.",
    "maior_sentenca_palavras": "Comprimento da maior sentença em número de palavras.",
    "variancia_tamanho_sentencas": "Variância do número de palavras por sentença.",
    "palavras_unicas": "Quantidade de palavras únicas em caixa baixa.",
    "type_token_ratio": "Razão entre palavras únicas e total de palavras.",
    "diversidade_lexical": "Medida de diversidade lexical equivalente ao type-token ratio nesta versão.",
    "razao_palavras_longas": "Proporção de palavras com tamanho maior ou igual ao limite configurado.",
    "razao_palavras_repetidas": "Proporção de tokens que aparecem mais de uma vez no texto.",
    "razao_pontuacao": "Proporção de caracteres de pontuação em relação ao total de caracteres.",
    "razao_numeros": "Proporção de tokens numéricos em relação ao total de palavras/tokens.",
    "frequencia_conectivos": "Proporção de conectivos simples em relação ao total de palavras.",
    "frequencia_marcadores_subordinacao": "Proporção de marcadores de subordinação em relação ao total de palavras.",
    "densidade_lexical_aproximada": "Proporção aproximada de palavras de conteúdo em relação ao total de palavras.",
}

__all__ = [
    "FeatureBuildError",
    "FeatureBuildResult",
    "build_feature_datasets",
    "build_features_dataframe",
    "build_features_metadata",
    "extract_linguistic_features",
    "get_feature_columns",
    "tokenize_sentences",
    "tokenize_words",
]


class FeatureBuildError(RuntimeError):
    """Erro gerado durante a construção de features linguísticas"""


# Resultado da geração dos arquivos de features
@dataclass(frozen=True)
class FeatureBuildResult:
    train_features_path: Path
    val_features_path: Path
    test_features_path: Path
    metadata_path: Path
    metadata: dict[str, Any]

    def to_dict(self, project_root: Path | None = None) -> dict[str, Any]:
        
        if project_root is None:
            return {
                "train_features_path": self.train_features_path.as_posix(),
                "val_features_path": self.val_features_path.as_posix(),
                "test_features_path": self.test_features_path.as_posix(),
                "metadata_path": self.metadata_path.as_posix(),
                "metadata": self.metadata,
            }

        return {
            "train_features_path": relative_to_root(self.train_features_path, project_root),
            "val_features_path": relative_to_root(self.val_features_path, project_root),
            "test_features_path": relative_to_root(self.test_features_path, project_root),
            "metadata_path": relative_to_root(self.metadata_path, project_root),
            "metadata": self.metadata,
        }


# Gera e salva features linguísticas para treino, validação e teste
def build_feature_datasets(settings: AppSettings) -> FeatureBuildResult:

    processed_dir = settings.outputs.processed_dir
    input_paths = {
        "train": processed_dir / "train.csv",
        "val": processed_dir / "val.csv",
        "test": processed_dir / "test.csv",
    }

    datasets = {
        split_name: _read_prepared_split(path, split_name)
        for split_name, path in input_paths.items()
    }
    feature_frames = {
        split_name: build_features_dataframe(df, settings, split_name)
        for split_name, df in datasets.items()
    }

    output_dir = settings.features.output_dir
    train_features_path = _write_feature_csv(feature_frames["train"], output_dir / "train_features.csv", "train")
    val_features_path = _write_feature_csv(feature_frames["val"], output_dir / "val_features.csv", "val")
    test_features_path = _write_feature_csv(feature_frames["test"], output_dir / "test_features.csv", "test")

    metadata_path = output_dir / "features_metadata.json"
    metadata = build_features_metadata(
        settings=settings,
        input_paths=input_paths,
        output_paths={
            "train": train_features_path,
            "val": val_features_path,
            "test": test_features_path,
            "metadata_path": metadata_path,
        },
        source_datasets=datasets,
        feature_frames=feature_frames,
    )
    metadata_path = _write_features_metadata(metadata, metadata_path)

    return FeatureBuildResult(
        train_features_path=train_features_path,
        val_features_path=val_features_path,
        test_features_path=test_features_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


# Constrói o DataFrame de features de um subconjunto preparado
def build_features_dataframe(
    df: pd.DataFrame,
    settings: AppSettings,
    split_name: str,
) -> pd.DataFrame:

    id_column = settings.dataset.id_column
    target_column = settings.dataset.target_column
    text_column = _resolve_feature_text_column(df, settings)
    required_columns = (id_column, target_column, text_column)

    _validate_prepared_split(df, required_columns, split_name)
    _validate_unique_keys(df, (id_column,), split_name)
    _validate_text_values(df, text_column, split_name)

    output_rows: list[dict[str, Any]] = []
    selected_columns = df.loc[:, [id_column, target_column, text_column]]

    for row in selected_columns.itertuples(index=False, name=None):
        row_id, target, text = row
        features = extract_linguistic_features(
            text,
            long_word_min_chars=settings.features.long_word_min_chars,
        )
        output_rows.append(
            {
                id_column: row_id,
                target_column: target,
                **features,
            }
        )

    feature_columns = get_feature_columns()
    feature_df = pd.DataFrame(output_rows, columns=[id_column, target_column, *feature_columns])
    _validate_feature_frame(feature_df, feature_columns, split_name)

    return feature_df


# Extrai métricas linguísticas interpretáveis de um texto
def extract_linguistic_features(
    text: object,
    long_word_min_chars: int = 7,
) -> dict[str, float | int]:

    if isinstance(long_word_min_chars, bool) or not isinstance(long_word_min_chars, int):
        raise FeatureBuildError("long_word_min_chars deve ser um inteiro.")
    if long_word_min_chars < 1:
        raise FeatureBuildError("long_word_min_chars deve ser maior ou igual a 1.")

    normalized_text = _coerce_text(text)
    sentences = tokenize_sentences(normalized_text)
    words_original = tokenize_words(normalized_text)
    words = [word.lower() for word in words_original]
    numbers = _NUMBER_PATTERN.findall(normalized_text)

    num_characters = len(normalized_text)
    num_words = len(words)
    num_sentences = len(sentences)
    sentence_lengths = [len(tokenize_words(sentence)) for sentence in sentences]
    sentence_lengths = [length for length in sentence_lengths if length > 0]

    word_lengths = [len(word) for word in words]
    word_counts = Counter(words)
    repeated_token_count = sum(count for count in word_counts.values() if count > 1)
    unique_words = len(word_counts)
    long_words = sum(1 for word in words if len(word) >= long_word_min_chars)
    punctuation_count = sum(1 for char in normalized_text if char in _PUNCTUATION_CHARS)
    connective_count = sum(1 for word in words if word in _CONNECTIVES)
    subordination_count = sum(1 for word in words if word in _SUBORDINATION_MARKERS)
    lexical_content_count = sum(1 for word in words if word not in _APPROXIMATE_FUNCTION_WORDS)

    return {
        "num_caracteres": num_characters,
        "num_palavras": num_words,
        "num_sentencas": num_sentences,
        "media_palavras_por_sentenca": _safe_division(num_words, num_sentences),
        "media_caracteres_por_palavra": _safe_mean(word_lengths),
        "maior_sentenca_palavras": max(sentence_lengths, default=0),
        "variancia_tamanho_sentencas": _safe_variance(sentence_lengths),
        "palavras_unicas": unique_words,
        "type_token_ratio": _safe_division(unique_words, num_words),
        "diversidade_lexical": _safe_division(unique_words, num_words),
        "razao_palavras_longas": _safe_division(long_words, num_words),
        "razao_palavras_repetidas": _safe_division(repeated_token_count, num_words),
        "razao_pontuacao": _safe_division(punctuation_count, num_characters),
        "razao_numeros": _safe_division(len(numbers), max(num_words, len(numbers))),
        "frequencia_conectivos": _safe_division(connective_count, num_words),
        "frequencia_marcadores_subordinacao": _safe_division(subordination_count, num_words),
        "densidade_lexical_aproximada": _safe_division(lexical_content_count, num_words),
    }


# Tokeniza sentenças de forma local, simples e determinística
def tokenize_sentences(text: object) -> list[str]:

    normalized = _coerce_text(text)
    if not normalized:
        return []

    candidates = [match.group(0).strip() for match in _SENTENCE_PATTERN.finditer(normalized)]
    sentences = [sentence for sentence in candidates if sentence]
    if sentences:
        return sentences

    return [normalized]


# Tokeniza palavras preservando caracteres acentuados do português
def tokenize_words(text: object) -> list[str]:

    return _WORD_PATTERN.findall(_coerce_text(text))


# Retorna os nomes das colunas numéricas de features na ordem oficial
def get_feature_columns() -> list[str]:

    return list(_FEATURE_DESCRIPTIONS.keys())


# Monta o arquivo de metadados das features geradas
def build_features_metadata(
    settings: AppSettings,
    input_paths: dict[str, Path],
    output_paths: dict[str, Path],
    source_datasets: dict[str, pd.DataFrame],
    feature_frames: dict[str, pd.DataFrame],
) -> dict[str, Any]:

    feature_columns = get_feature_columns()
    id_column = settings.dataset.id_column
    target_column = settings.dataset.target_column
    required_splits = ("train", "val", "test")

    missing_splits = [
        split
        for split in required_splits
        if split not in input_paths or split not in source_datasets or split not in feature_frames
    ]
    if missing_splits:
        raise FeatureBuildError(
            "Não foi possível montar metadados. Split(s) ausente(s): "
            f"{', '.join(missing_splits)}."
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": {
            "name": settings.project.name,
            "version": settings.project.version,
            "random_state": settings.project.random_state,
        },
        "source": {
            "text_column_used": _resolve_feature_text_column(source_datasets["train"], settings),
            "text_column_by_split": {
                split: _resolve_feature_text_column(source_datasets[split], settings)
                for split in required_splits
            },
            "id_column": id_column,
            "target_column": target_column,
            "input_paths": {
                split: relative_to_root(path, settings.project_root)
                for split, path in input_paths.items()
            },
        },
        "outputs": {
            split: relative_to_root(path, settings.project_root)
            for split, path in output_paths.items()
        },
        "parameters": {
            "long_word_min_chars": settings.features.long_word_min_chars,
            "sentence_tokenization": "regex_local",
            "word_tokenization": "regex_local_unicode",
            "external_downloads_required": False,
        },
        "feature_columns": feature_columns,
        "feature_descriptions": dict(_FEATURE_DESCRIPTIONS),
        "splits": {
            split: {
                "source_rows": int(len(source_datasets[split])),
                "feature_rows": int(len(feature_frames[split])),
                "feature_count": len(feature_columns),
                "class_distribution": _class_distribution(feature_frames[split], target_column),
                "missing_values_by_feature": {
                    column: int(feature_frames[split][column].isna().sum())
                    for column in feature_columns
                },
            }
            for split in required_splits
        },
    }


def _read_prepared_split(path: Path, split_name: str) -> pd.DataFrame:
    try:
        return read_csv_dataset(path)
    except Exception as exc:
        raise FeatureBuildError(
            f"Não foi possível ler o subconjunto preparado '{split_name}'. "
            "Execute primeiro: python scripts/prepare_dataset.py. "
            f"Detalhe: {exc}"
        ) from exc


def _write_feature_csv(df: pd.DataFrame, path: Path, split_name: str) -> Path:
    try:
        return write_csv_dataset(df, path)
    except Exception as exc:
        raise FeatureBuildError(
            f"Não foi possível salvar as features do subconjunto '{split_name}' em {path}: {exc}"
        ) from exc


def _write_features_metadata(metadata: dict[str, Any], path: Path) -> Path:
    try:
        return write_json(metadata, path)
    except Exception as exc:
        raise FeatureBuildError(f"Não foi possível salvar os metadados de features em {path}: {exc}") from exc


def _validate_prepared_split(
    df: pd.DataFrame,
    required_columns: tuple[str, ...],
    split_name: str,
) -> None:
    if df.empty:
        raise FeatureBuildError(f"O subconjunto '{split_name}' está vazio.")

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise FeatureBuildError(
            f"O subconjunto '{split_name}' não possui as colunas necessárias: "
            f"{', '.join(missing_columns)}."
        )

    null_columns = {
        column: int(df[column].isna().sum())
        for column in required_columns
        if int(df[column].isna().sum()) > 0
    }
    if null_columns:
        raise FeatureBuildError(
            f"O subconjunto '{split_name}' possui valores nulos em colunas necessárias: {null_columns}."
        )


def _validate_unique_keys(df: pd.DataFrame, key_columns: tuple[str, ...], split_name: str) -> None:
    duplicated_count = int(df.duplicated(subset=list(key_columns), keep=False).sum())
    if duplicated_count == 0:
        return

    formatted_keys = ", ".join(key_columns)
    raise FeatureBuildError(
        f"O subconjunto '{split_name}' possui {duplicated_count} registro(s) com chave duplicada "
        f"em ({formatted_keys})."
    )


def _validate_text_values(df: pd.DataFrame, text_column: str, split_name: str) -> None:
    normalized_text = df[text_column].map(_coerce_text)
    empty_count = int(normalized_text.eq("").sum())
    if empty_count > 0:
        raise FeatureBuildError(
            f"O subconjunto '{split_name}' possui {empty_count} texto(s) vazio(s) em '{text_column}'."
        )


def _validate_feature_frame(
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    split_name: str,
) -> None:
    if feature_df.empty:
        raise FeatureBuildError(f"Nenhuma feature foi gerada para o subconjunto '{split_name}'.")

    missing_columns = [column for column in feature_columns if column not in feature_df.columns]
    if missing_columns:
        raise FeatureBuildError(
            f"As features do subconjunto '{split_name}' não possuem coluna(s): "
            f"{', '.join(missing_columns)}."
        )

    invalid_columns: dict[str, int] = {}
    for column in feature_columns:
        values = pd.to_numeric(feature_df[column], errors="coerce")
        invalid_count = int((values.isna() | ~values.map(math.isfinite)).sum())
        if invalid_count > 0:
            invalid_columns[column] = invalid_count

    if invalid_columns:
        raise FeatureBuildError(
            f"As features do subconjunto '{split_name}' possuem valores inválidos: {invalid_columns}."
        )


def _resolve_feature_text_column(df: pd.DataFrame, settings: AppSettings) -> str:
    clean_column = settings.preprocessing.clean_text_column
    if clean_column in df.columns:
        return clean_column

    text_column = settings.dataset.text_column
    if text_column in df.columns:
        return text_column

    raise FeatureBuildError(
        f"Não foi encontrada coluna textual para features. Esperado '{clean_column}' ou '{text_column}'."
    )


def _class_distribution(df: pd.DataFrame, target_column: str) -> dict[str, int]:
    if target_column not in df.columns:
        return {}

    counts = df[target_column].astype(str).value_counts(dropna=False).sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _coerce_text(value: object) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def _safe_division(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0

    value = float(numerator) / float(denominator)
    if math.isfinite(value):
        return round(value, 6)
    return 0.0


def _safe_mean(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return round(float(mean(values)), 6)


def _safe_variance(values: list[int | float]) -> float:
    if len(values) < 2:
        return 0.0
    return round(float(variance(values)), 6)
