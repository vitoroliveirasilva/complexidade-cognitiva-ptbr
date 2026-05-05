from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any, TextIO

import pandas as pd


class DataIOError(RuntimeError):
    """Erro gerado em operações de leitura ou escrita de dados"""


# Lê um dataset CSV em UTF-8 e retorna um DataFrame
def read_csv_dataset(path: str | Path) -> pd.DataFrame:

    csv_path = _resolve_input_file(path=path, expected_suffix=".csv", label="dataset")

    try:
        dataframe = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError as exc:
        raise DataIOError(
            "Não foi possível ler o CSV usando UTF-8. Salve o arquivo como UTF-8 e tente novamente."
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise DataIOError(f"O arquivo CSV está vazio: {csv_path}") from exc
    except pd.errors.ParserError as exc:
        raise DataIOError(
            f"Erro ao interpretar o CSV: {csv_path}. Detalhes: {exc}"
        ) from exc
    except OSError as exc:
        raise DataIOError(f"Não foi possível ler o CSV em {csv_path}: {exc}") from exc

    _validate_dataframe_columns(dataframe, csv_path)
    return dataframe


# Salva um DataFrame em CSV sem índice e retorna o caminho final
def write_csv_dataset(df: pd.DataFrame, path: str | Path) -> Path:

    if not isinstance(df, pd.DataFrame):
        raise DataIOError(
            "O conteúdo informado para escrita em CSV deve ser um pandas.DataFrame."
        )

    _validate_dataframe_columns(df, None)
    output_path = _resolve_output_file(path=path, expected_suffix=".csv", label="CSV")

    def writer(file: TextIO) -> None:
        df.to_csv(file, index=False, encoding="utf-8")

    _atomic_write(output_path, writer)
    return output_path


# Salva um dicionário em JSON UTF-8 indentado
def write_json(payload: dict[str, Any], path: str | Path) -> Path:

    if not isinstance(payload, Mapping):
        raise DataIOError(
            "O conteúdo informado para escrita em JSON deve ser um mapeamento/dicionário."
        )

    output_path = _resolve_output_file(path=path, expected_suffix=".json", label="JSON")

    def writer(file: TextIO) -> None:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=_json_default)
        file.write("\n")

    _atomic_write(output_path, writer)
    return output_path


def _resolve_input_file(path: str | Path, expected_suffix: str, label: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DataIOError(f"Caminho inválido para {label}: {path!r}") from exc

    if not resolved.exists():
        raise DataIOError(f"Arquivo não encontrado: {resolved}")
    if not resolved.is_file():
        raise DataIOError(f"O caminho informado não é um arquivo: {resolved}")
    if resolved.suffix.lower() != expected_suffix:
        raise DataIOError(
            f"O arquivo de {label} deve usar a extensão '{expected_suffix}'. Arquivo recebido: {resolved.name}"
        )

    return resolved


def _resolve_output_file(path: str | Path, expected_suffix: str, label: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DataIOError(
            f"Caminho inválido para escrita de {label}: {path!r}"
        ) from exc

    if resolved.exists() and resolved.is_dir():
        raise DataIOError(
            f"O caminho de saída de {label} aponta para um diretório: {resolved}"
        )
    if resolved.suffix.lower() != expected_suffix:
        raise DataIOError(
            f"O arquivo de saída de {label} deve usar a extensão '{expected_suffix}'. Caminho recebido: {resolved}"
        )

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DataIOError(
            f"Não foi possível criar o diretório de saída {resolved.parent}: {exc}"
        ) from exc

    return resolved


def _validate_dataframe_columns(df: pd.DataFrame, source_path: Path | None) -> None:
    duplicated_columns = [
        str(column) for column in df.columns[df.columns.duplicated()].tolist()
    ]
    if duplicated_columns:
        source = f" em {source_path}" if source_path is not None else ""
        preview = ", ".join(duplicated_columns[:10])
        raise DataIOError(f"Foram encontradas colunas duplicadas{source}: {preview}.")


def _atomic_write(path: Path, writer: Any) -> None:
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            writer(temp_file)

        os.replace(temp_path, path)
    except (OSError, TypeError, ValueError) as exc:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise DataIOError(
            f"Não foi possível salvar o arquivo em {path}: {exc}"
        ) from exc


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, set):
        return sorted(value)

    raise TypeError(f"Objeto não serializável em JSON: {type(value).__name__}")
