from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config.settings import AppSettings
from .io import read_csv_dataset, write_csv_dataset, write_json
from ..utils.paths import relative_to_root

_WHITESPACE_PATTERN = re.compile(r"[ \t\f\v]+")
_EXCESSIVE_LINE_BREAKS_PATTERN = re.compile(r"\n{3,}")
_SPACES_AROUND_LINE_BREAKS_PATTERN = re.compile(r" *\n *")
_CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class DatasetValidationError(ValueError):
    """Erro gerado quando o dataset não atende aos requisitos mínimos"""


# Resultado da preparação do dataset
@dataclass(frozen=True)
class PreparationResult:
    train_path: Path
    val_path: Path
    test_path: Path
    report_path: Path
    report: dict[str, Any]

    def to_dict(self, project_root: Path | None = None) -> dict[str, Any]:

        if project_root is None:
            return {
                "train_path": self.train_path.as_posix(),
                "val_path": self.val_path.as_posix(),
                "test_path": self.test_path.as_posix(),
                "report_path": self.report_path.as_posix(),
                "report": self.report,
            }

        return {
            "train_path": relative_to_root(self.train_path, project_root),
            "val_path": relative_to_root(self.val_path, project_root),
            "test_path": relative_to_root(self.test_path, project_root),
            "report_path": relative_to_root(self.report_path, project_root),
            "report": self.report,
        }


# Executa leitura, validação, limpeza, split e persistência do dataset
def prepare_dataset(settings: AppSettings) -> PreparationResult:

    raw_df = read_csv_dataset(settings.dataset.input_path)
    validation_summary = validate_dataset(raw_df, settings)
    prepared_df, cleaning_summary = build_preprocessed_dataset(raw_df, settings)
    split_result = split_dataset(prepared_df, settings)

    processed_dir = settings.outputs.processed_dir
    train_path = write_csv_dataset(split_result["train"], processed_dir / "train.csv")
    val_path = write_csv_dataset(split_result["val"], processed_dir / "val.csv")
    test_path = write_csv_dataset(split_result["test"], processed_dir / "test.csv")

    report = build_preparation_report(
        settings=settings,
        raw_df=raw_df,
        prepared_df=prepared_df,
        validation_summary=validation_summary,
        cleaning_summary=cleaning_summary,
        split_metadata=split_result["metadata"],
        train_df=split_result["train"],
        val_df=split_result["val"],
        test_df=split_result["test"],
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
    )
    report_path = write_json(report, processed_dir / "preparation_report.json")

    return PreparationResult(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        report_path=report_path,
        report=report,
    )


# Valida os requisitos mínimos do dataset bruto
def validate_dataset(df: pd.DataFrame, settings: AppSettings) -> dict[str, Any]:

    if not isinstance(df, pd.DataFrame):
        raise DatasetValidationError("O dataset deve ser um pandas.DataFrame.")

    errors: list[str] = []
    warnings: list[str] = []
    id_column, text_column, target_column = settings.required_dataset_columns

    duplicated_column_names = [str(column) for column in df.columns[df.columns.duplicated()].tolist()]
    if duplicated_column_names:
        errors.append(f"O dataset possui colunas duplicadas: {', '.join(duplicated_column_names[:10])}.")

    if df.empty:
        errors.append("O dataset está vazio.")

    required_columns = tuple(settings.required_dataset_columns)
    if len(set(required_columns)) != len(required_columns):
        errors.append("As colunas obrigatórias configuradas devem possuir nomes distintos.")

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        errors.append(f"Colunas obrigatórias ausentes: {', '.join(missing_columns)}.")
        raise DatasetValidationError(" ".join(errors))

    required_nulls = {
        column: int(df[column].isna().sum()) for column in required_columns
    }
    for column, null_count in required_nulls.items():
        if null_count > 0:
            errors.append(f"A coluna '{column}' possui {null_count} valor(es) nulo(s).")

    blank_summary = _count_blank_required_values(df, required_columns)
    for column, blank_count in blank_summary.items():
        if blank_count > 0:
            errors.append(f"A coluna '{column}' possui {blank_count} valor(es) vazio(s).")

    duplicated_ids = _duplicated_values(df, id_column)
    if duplicated_ids:
        preview = ", ".join(duplicated_ids[:10])
        suffix = "..." if len(duplicated_ids) > 10 else ""
        errors.append(f"A coluna '{id_column}' possui identificadores duplicados: {preview}{suffix}.")

    class_distribution = _class_distribution(df, target_column)
    if len(class_distribution) < 2:
        errors.append(f"O dataset precisa possuir pelo menos duas classes distintas em '{target_column}'.")

    if len(df) < 3:
        errors.append("O dataset precisa possuir pelo menos três linhas para gerar treino, validação e teste.")

    exact_duplicate_text_count = int(df[text_column].duplicated(keep=False).sum())
    if exact_duplicate_text_count > 0:
        warnings.append(
            f"Foram encontrados {exact_duplicate_text_count} registro(s) envolvidos em duplicidade textual exata."
        )

    normalized_duplicate_text_count = _count_normalized_duplicate_texts(df, text_column)
    if normalized_duplicate_text_count > exact_duplicate_text_count:
        warnings.append(
            f"Foram encontrados {normalized_duplicate_text_count} registro(s) envolvidos em duplicidade textual após normalização simples."
        )

    if errors:
        raise DatasetValidationError(" ".join(errors))

    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "required_columns": list(required_columns),
        "required_nulls": required_nulls,
        "blank_required_values": blank_summary,
        "class_distribution": class_distribution,
        "exact_duplicate_text_count": exact_duplicate_text_count,
        "normalized_duplicate_text_count": normalized_duplicate_text_count,
        "exact_duplicate_text_examples": _duplicate_text_examples(df, id_column, text_column),
        "warnings": warnings,
    }


# Cria a versão preparada do dataset com texto limpo
def build_preprocessed_dataset(
    df: pd.DataFrame,
    settings: AppSettings,
) -> tuple[pd.DataFrame, dict[str, Any]]:

    if not isinstance(df, pd.DataFrame):
        raise DatasetValidationError("O dataset deve ser um pandas.DataFrame.")

    text_column = settings.dataset.text_column
    clean_column = settings.preprocessing.clean_text_column
    if text_column not in df.columns:
        raise DatasetValidationError(f"Coluna textual ausente no dataset: {text_column}")
    if not clean_column.strip():
        raise DatasetValidationError("A coluna de texto limpo configurada não pode ser vazia.")

    prepared = df.copy()
    original_texts = prepared[text_column].map(_coerce_text_value)
    cleaned_texts = original_texts.map(
        lambda value: clean_text(
            value,
            normalize_whitespace=settings.preprocessing.normalize_whitespace,
        )
    )

    empty_after_cleaning = int((cleaned_texts.str.len() == 0).sum())
    if empty_after_cleaning > 0:
        raise DatasetValidationError(
            f"Após a limpeza textual, {empty_after_cleaning} texto(s) ficaram vazios."
        )

    changed_rows = int((original_texts != cleaned_texts).sum())
    prepared[clean_column] = cleaned_texts

    if not settings.preprocessing.preserve_original_text and clean_column != text_column:
        prepared[text_column] = cleaned_texts

    return prepared, {
        "text_column": text_column,
        "clean_text_column": clean_column,
        "preserve_original_text": settings.preprocessing.preserve_original_text,
        "normalize_whitespace": settings.preprocessing.normalize_whitespace,
        "changed_rows": changed_rows,
        "empty_after_cleaning": empty_after_cleaning,
    }


# Aplica limpeza textual controlada e não destrutiva
def clean_text(text: str, normalize_whitespace: bool = True) -> str:

    normalized = _coerce_text_value(text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _CONTROL_CHARS_PATTERN.sub("", normalized)

    if normalize_whitespace:
        normalized = _WHITESPACE_PATTERN.sub(" ", normalized)
        normalized = _SPACES_AROUND_LINE_BREAKS_PATTERN.sub("\n", normalized)
        normalized = _EXCESSIVE_LINE_BREAKS_PATTERN.sub("\n\n", normalized)

    return normalized.strip()


# Divide o dataset em treino, validação e teste
def split_dataset(df: pd.DataFrame, settings: AppSettings) -> dict[str, Any]:

    if not isinstance(df, pd.DataFrame):
        raise DatasetValidationError("O dataset preparado deve ser um pandas.DataFrame.")

    split_config = settings.dataset.split
    random_state = settings.project.random_state
    target_column = settings.dataset.target_column

    if target_column not in df.columns:
        raise DatasetValidationError(f"Coluna alvo ausente no dataset preparado: {target_column}")

    estimated_counts = _estimate_split_counts(
        len(df),
        split_config.train_size,
        split_config.val_size,
        split_config.test_size,
    )
    _validate_split_sizes(len(df), estimated_counts)

    stratify_requested = split_config.stratify
    warnings: list[str] = []

    try:
        train_df, val_df, test_df, stratified_used = _execute_split(
            df=df,
            target_column=target_column,
            train_size=split_config.train_size,
            val_size=split_config.val_size,
            test_size=split_config.test_size,
            random_state=random_state,
            stratify=stratify_requested,
        )
    except ValueError as exc:
        if not stratify_requested:
            raise DatasetValidationError(f"Não foi possível dividir o dataset: {exc}") from exc

        warnings.append(
            "Split estratificado inviável para o tamanho/distribuição atual do dataset. "
            f"Foi aplicado split não estratificado. Detalhe original: {exc}"
        )
        try:
            train_df, val_df, test_df, stratified_used = _execute_split(
                df=df,
                target_column=target_column,
                train_size=split_config.train_size,
                val_size=split_config.val_size,
                test_size=split_config.test_size,
                random_state=random_state,
                stratify=False,
            )
        except ValueError as fallback_exc:
            raise DatasetValidationError(f"Não foi possível dividir o dataset: {fallback_exc}") from fallback_exc

    _ensure_non_empty_splits(train_df=train_df, val_df=val_df, test_df=test_df)
    warnings.extend(_split_class_warnings(train_df, val_df, test_df, target_column))

    train_df = _sort_by_identifier(train_df, settings.dataset.id_column)
    val_df = _sort_by_identifier(val_df, settings.dataset.id_column)
    test_df = _sort_by_identifier(test_df, settings.dataset.id_column)

    metadata = {
        "requested": {
            "train_size": split_config.train_size,
            "val_size": split_config.val_size,
            "test_size": split_config.test_size,
            "stratify": stratify_requested,
        },
        "estimated_counts": estimated_counts,
        "actual_counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "actual_percentages": {
            "train": _safe_ratio(len(train_df), len(df)),
            "val": _safe_ratio(len(val_df), len(df)),
            "test": _safe_ratio(len(test_df), len(df)),
        },
        "stratified_used": stratified_used,
        "stratification_warning": warnings[0] if warnings else "",
        "warnings": warnings,
    }

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "metadata": metadata,
    }


# Monta o relatório JSON da preparação dos dados
def build_preparation_report(
    settings: AppSettings,
    raw_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
    validation_summary: dict[str, Any],
    cleaning_summary: dict[str, Any],
    split_metadata: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_path: Path,
    val_path: Path,
    test_path: Path,
) -> dict[str, Any]:

    target_column = settings.dataset.target_column

    warnings = list(validation_summary.get("warnings", []))
    warnings.extend(split_metadata.get("warnings", []))

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": {
            "name": settings.project.name,
            "version": settings.project.version,
            "random_state": settings.project.random_state,
        },
        "input": {
            "path": relative_to_root(settings.dataset.input_path, settings.project_root),
            "rows": int(len(raw_df)),
            "columns": list(raw_df.columns),
        },
        "validation": validation_summary,
        "cleaning": cleaning_summary,
        "split": {
            **split_metadata,
            "class_distribution": {
                "full": _class_distribution(prepared_df, target_column),
                "train": _class_distribution(train_df, target_column),
                "val": _class_distribution(val_df, target_column),
                "test": _class_distribution(test_df, target_column),
            },
        },
        "outputs": {
            "train_path": relative_to_root(train_path, settings.project_root),
            "val_path": relative_to_root(val_path, settings.project_root),
            "test_path": relative_to_root(test_path, settings.project_root),
            "report_path": relative_to_root(
                settings.outputs.processed_dir / "preparation_report.json",
                settings.project_root,
            ),
        },
        "warnings": _deduplicate_preserving_order(warnings),
    }


def _execute_split(
    df: pd.DataFrame,
    target_column: str,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
    stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    full_stratify = df[target_column] if stratify else None
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=full_stratify,
    )

    relative_val_size = val_size / (train_size + val_size)
    train_val_stratify = train_val_df[target_column] if stratify else None
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_state,
        shuffle=True,
        stratify=train_val_stratify,
    )

    return train_df, val_df, test_df, bool(stratify)


def _estimate_split_counts(n_rows: int, train_size: float, val_size: float, test_size: float) -> dict[str, int]:
    if n_rows <= 0:
        return {"train": 0, "val": 0, "test": 0}

    test_count = int(math.ceil(n_rows * test_size))
    train_val_count = n_rows - test_count
    relative_val_size = val_size / (train_size + val_size)
    val_count = int(math.ceil(train_val_count * relative_val_size))
    train_count = train_val_count - val_count

    return {
        "train": train_count,
        "val": val_count,
        "test": test_count,
    }


def _validate_split_sizes(n_rows: int, estimated_counts: dict[str, int]) -> None:
    if any(count <= 0 for count in estimated_counts.values()):
        raise DatasetValidationError(
            "As proporções configuradas geram subconjuntos vazios para o tamanho atual do dataset. "
            f"Linhas: {n_rows}; estimativa: train={estimated_counts['train']}, "
            f"val={estimated_counts['val']}, test={estimated_counts['test']}."
        )


def _ensure_non_empty_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    empty_splits = [
        split_name
        for split_name, split_df in {"train": train_df, "val": val_df, "test": test_df}.items()
        if split_df.empty
    ]
    if empty_splits:
        raise DatasetValidationError(f"Split gerou subconjunto(s) vazio(s): {', '.join(empty_splits)}.")


def _split_class_warnings(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
) -> list[str]:
    warnings: list[str] = []
    train_classes = set(train_df[target_column].astype(str).unique())

    for split_name, split_df in {"val": val_df, "test": test_df}.items():
        split_classes = set(split_df[target_column].astype(str).unique())
        unseen_classes = sorted(split_classes.difference(train_classes))
        if unseen_classes:
            warnings.append(
                f"O subconjunto '{split_name}' possui classe(s) ausente(s) no treino: {', '.join(unseen_classes)}."
            )

    return warnings


def _count_blank_required_values(df: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, int]:
    blank_counts: dict[str, int] = {}
    for column in columns:
        blank_counts[column] = int(df[column].map(_is_blank_value).sum())
    return blank_counts


def _duplicated_values(df: pd.DataFrame, column: str) -> list[str]:
    duplicated_mask = df[column].duplicated(keep=False)
    if not bool(duplicated_mask.any()):
        return []

    duplicated_values = df.loc[duplicated_mask, column].dropna().astype(str).unique().tolist()
    return sorted(duplicated_values)


def _duplicate_text_examples(df: pd.DataFrame, id_column: str, text_column: str) -> list[dict[str, Any]]:
    duplicated_mask = df[text_column].duplicated(keep=False)
    if not bool(duplicated_mask.any()):
        return []

    examples: list[dict[str, Any]] = []
    duplicated_df = df.loc[duplicated_mask, [id_column, text_column]].copy()

    for text_value, group in duplicated_df.groupby(text_column, dropna=False):
        ids = group[id_column].astype(str).tolist()
        examples.append(
            {
                "text_preview": _coerce_text_value(text_value)[:120],
                "ids": ids[:10],
                "occurrences": int(len(group)),
            }
        )
        if len(examples) >= 10:
            break

    return examples


def _count_normalized_duplicate_texts(df: pd.DataFrame, text_column: str) -> int:
    normalized = df[text_column].map(lambda value: clean_text(_coerce_text_value(value)).casefold())
    return int(normalized.duplicated(keep=False).sum())


def _class_distribution(df: pd.DataFrame, target_column: str) -> dict[str, int]:
    if target_column not in df.columns:
        return {}

    counts = df[target_column].map(_coerce_text_value).value_counts(dropna=False).sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _safe_ratio(part: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(part / total, 6)


# Ordena de forma determinística usando o identificador como texto
def _sort_by_identifier(df: pd.DataFrame, id_column: str) -> pd.DataFrame:

    if id_column not in df.columns:
        return df.reset_index(drop=True)

    return df.sort_values(id_column, key=lambda series: series.astype(str)).reset_index(drop=True)


def _coerce_text_value(value: Any) -> str:
    if value is None:
        return ""

    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value)


def _is_blank_value(value: Any) -> bool:
    return _coerce_text_value(value).strip() == ""


def _deduplicate_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
