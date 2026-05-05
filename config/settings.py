from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_CONFIG_MAX_BYTES = 2_000_000
_SPLIT_SUM_TOLERANCE = 1e-6
_ALLOWED_LOGGING_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_ALLOWED_LOGGING_FORMATS = frozenset({"text", "json"})


class SettingsError(ValueError):
    """Erro gerado quando a configuração do projeto é inválida."""


# Configurações gerais do projeto
@dataclass(frozen=True)
class ProjectConfig:
    name: str
    version: str
    description: str
    language: str
    random_state: int


# Configurações de divisão do dataset
@dataclass(frozen=True)
class SplitConfig:
    train_size: float
    val_size: float
    test_size: float
    stratify: bool


# Configurações de entrada e colunas obrigatórias do dataset
@dataclass(frozen=True)
class DatasetConfig:
    input_path: Path
    id_column: str
    text_column: str
    target_column: str
    split: SplitConfig


# Configurações de pré-processamento textual
@dataclass(frozen=True)
class PreprocessingConfig:
    preserve_original_text: bool
    clean_text_column: str
    normalize_whitespace: bool


# Configuração de uma representação TF-IDF
@dataclass(frozen=True)
class TfidfVectorizerConfig:
    enabled: bool
    max_features: int
    ngram_range: tuple[int, int]
    min_df: int


# Configurações de extração de atributos
@dataclass(frozen=True)
class FeaturesConfig:
    output_dir: Path
    long_word_min_chars: int
    tfidf_word: TfidfVectorizerConfig
    tfidf_char: TfidfVectorizerConfig


# Configurações de treinamento e seleção de experimentos
@dataclass(frozen=True)
class TrainingConfig:
    models: tuple[str, ...]
    representations: tuple[str, ...]
    selection_metric: str


# Configurações dos diretórios de saída
@dataclass(frozen=True)
class OutputsConfig:
    processed_dir: Path
    model_dir: Path
    metrics_dir: Path
    figures_dir: Path
    reports_dir: Path


# Configurações de logging
@dataclass(frozen=True)
class LoggingConfig:
    level: str
    format: str


# Configuração consolidada da aplicação
@dataclass(frozen=True)
class AppSettings:
    config_path: Path
    project_root: Path
    project: ProjectConfig
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    features: FeaturesConfig
    training: TrainingConfig
    outputs: OutputsConfig
    logging: LoggingConfig

    @property
    def required_dataset_columns(self) -> tuple[str, str, str]:
        return (
            self.dataset.id_column,
            self.dataset.text_column,
            self.dataset.target_column,
        )

    # Retorna os diretórios que devem existir para execução local
    @property
    def managed_directories(self) -> tuple[Path, ...]:
        return (
            self.dataset.input_path.parent,
            self.outputs.processed_dir,
            self.features.output_dir,
            self.outputs.model_dir,
            self.outputs.metrics_dir,
            self.outputs.figures_dir,
            self.outputs.reports_dir,
        )


# Carrega e valida as configurações do projeto a partir do YAML
def load_settings(config_path: str | Path | None = None) -> AppSettings:

    initial_root = _find_project_root(Path.cwd())
    resolved_config_path = _resolve_config_path(config_path, initial_root)
    project_root = _find_project_root(resolved_config_path.parent)
    raw_config = _load_yaml(resolved_config_path)

    project = _parse_project(raw_config)
    dataset = _parse_dataset(raw_config, project_root)
    preprocessing = _parse_preprocessing(raw_config)
    features = _parse_features(raw_config, project_root)
    training = _parse_training(raw_config)
    outputs = _parse_outputs(raw_config, project_root)
    logging = _parse_logging(raw_config)

    _validate_dataset_columns(dataset)

    return AppSettings(
        config_path=resolved_config_path,
        project_root=project_root,
        project=project,
        dataset=dataset,
        preprocessing=preprocessing,
        features=features,
        training=training,
        outputs=outputs,
        logging=logging,
    )


def _resolve_config_path(config_path: str | Path | None, project_root: Path) -> Path:
    if config_path is None:
        return (project_root / "configs" / "config.yaml").resolve()

    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def _load_yaml(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise SettingsError(f"Arquivo de configuração não encontrado: {config_path}")

    if not config_path.is_file():
        raise SettingsError(
            f"Caminho de configuração não aponta para um arquivo: {config_path}"
        )

    try:
        file_size = config_path.stat().st_size
    except OSError as exc:
        raise SettingsError(
            f"Não foi possível acessar o arquivo de configuração: {config_path}"
        ) from exc

    if file_size > _CONFIG_MAX_BYTES:
        raise SettingsError(
            "Arquivo de configuração muito grande para um YAML de configuração: "
            f"{config_path} ({file_size} bytes)."
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise SettingsError(
            f"YAML de configuração inválido em {config_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise SettingsError(
            f"Erro ao ler o arquivo de configuração: {config_path}"
        ) from exc

    if not isinstance(loaded, dict):
        raise SettingsError(
            "O arquivo de configuração deve conter um mapeamento YAML na raiz."
        )

    return loaded


def _parse_project(raw_config: Mapping[str, Any]) -> ProjectConfig:
    section = _required_mapping(raw_config, "project")
    return ProjectConfig(
        name=_required_str(section, "name", "project"),
        version=_required_str(section, "version", "project"),
        description=_required_str(section, "description", "project"),
        language=_required_str(section, "language", "project"),
        random_state=_required_int(section, "random_state", "project", minimum=0),
    )


def _parse_dataset(raw_config: Mapping[str, Any], project_root: Path) -> DatasetConfig:
    section = _required_mapping(raw_config, "dataset")
    split_section = _required_mapping(section, "split", "dataset")

    split = SplitConfig(
        train_size=_required_float(
            split_section, "train_size", "dataset.split", minimum=0.0, maximum=1.0
        ),
        val_size=_required_float(
            split_section, "val_size", "dataset.split", minimum=0.0, maximum=1.0
        ),
        test_size=_required_float(
            split_section, "test_size", "dataset.split", minimum=0.0, maximum=1.0
        ),
        stratify=_required_bool(split_section, "stratify", "dataset.split"),
    )
    _validate_split(split)

    input_path = _resolve_project_path(
        project_root,
        _required_str(section, "input_path", "dataset"),
    )

    return DatasetConfig(
        input_path=input_path,
        id_column=_required_str(section, "id_column", "dataset"),
        text_column=_required_str(section, "text_column", "dataset"),
        target_column=_required_str(section, "target_column", "dataset"),
        split=split,
    )


def _parse_preprocessing(raw_config: Mapping[str, Any]) -> PreprocessingConfig:
    section = _required_mapping(raw_config, "preprocessing")
    return PreprocessingConfig(
        preserve_original_text=_required_bool(
            section, "preserve_original_text", "preprocessing"
        ),
        clean_text_column=_required_str(section, "clean_text_column", "preprocessing"),
        normalize_whitespace=_required_bool(
            section, "normalize_whitespace", "preprocessing"
        ),
    )


def _parse_features(
    raw_config: Mapping[str, Any], project_root: Path
) -> FeaturesConfig:
    section = _required_mapping(raw_config, "features")
    tfidf_section = _required_mapping(section, "tfidf", "features")

    return FeaturesConfig(
        output_dir=_resolve_project_path(
            project_root,
            _required_str(section, "output_dir", "features"),
        ),
        long_word_min_chars=_required_int(
            section, "long_word_min_chars", "features", minimum=1
        ),
        tfidf_word=_parse_tfidf(
            _required_mapping(tfidf_section, "word", "features.tfidf"),
            "features.tfidf.word",
        ),
        tfidf_char=_parse_tfidf(
            _required_mapping(tfidf_section, "char", "features.tfidf"),
            "features.tfidf.char",
        ),
    )


def _parse_tfidf(
    section: Mapping[str, Any], section_name: str
) -> TfidfVectorizerConfig:
    ngram_range = section.get("ngram_range")
    if not isinstance(ngram_range, Sequence) or isinstance(ngram_range, str):
        raise SettingsError(f"{section_name}.ngram_range deve conter dois inteiros.")

    if len(ngram_range) != 2:
        raise SettingsError(
            f"{section_name}.ngram_range deve conter exatamente dois inteiros."
        )

    start, end = ngram_range
    if not _is_plain_int(start) or not _is_plain_int(end):
        raise SettingsError(f"{section_name}.ngram_range deve conter apenas inteiros.")

    if start <= 0 or end < start:
        raise SettingsError(
            f"{section_name}.ngram_range deve respeitar 1 <= início <= fim."
        )

    return TfidfVectorizerConfig(
        enabled=_required_bool(section, "enabled", section_name),
        max_features=_required_int(section, "max_features", section_name, minimum=1),
        ngram_range=(start, end),
        min_df=_required_int(section, "min_df", section_name, minimum=1),
    )


def _parse_training(raw_config: Mapping[str, Any]) -> TrainingConfig:
    section = _required_mapping(raw_config, "training")
    return TrainingConfig(
        models=_required_unique_str_tuple(section, "models", "training"),
        representations=_required_unique_str_tuple(
            section, "representations", "training"
        ),
        selection_metric=_required_str(section, "selection_metric", "training"),
    )


def _parse_outputs(raw_config: Mapping[str, Any], project_root: Path) -> OutputsConfig:
    section = _required_mapping(raw_config, "outputs")
    return OutputsConfig(
        processed_dir=_resolve_project_path(
            project_root,
            _required_str(section, "processed_dir", "outputs"),
        ),
        model_dir=_resolve_project_path(
            project_root, _required_str(section, "model_dir", "outputs")
        ),
        metrics_dir=_resolve_project_path(
            project_root,
            _required_str(section, "metrics_dir", "outputs"),
        ),
        figures_dir=_resolve_project_path(
            project_root,
            _required_str(section, "figures_dir", "outputs"),
        ),
        reports_dir=_resolve_project_path(
            project_root,
            _required_str(section, "reports_dir", "outputs"),
        ),
    )


def _parse_logging(raw_config: Mapping[str, Any]) -> LoggingConfig:
    section = _required_mapping(raw_config, "logging")
    level = _required_str(section, "level", "logging").upper()
    if level not in _ALLOWED_LOGGING_LEVELS:
        accepted = ", ".join(sorted(_ALLOWED_LOGGING_LEVELS))
        raise SettingsError(
            f"logging.level inválido: {level}. Valores aceitos: {accepted}."
        )

    output_format = _required_str(section, "format", "logging").lower()
    if output_format not in _ALLOWED_LOGGING_FORMATS:
        accepted = ", ".join(sorted(_ALLOWED_LOGGING_FORMATS))
        raise SettingsError(
            f"logging.format inválido: {output_format}. Valores aceitos: {accepted}."
        )

    return LoggingConfig(level=level, format=output_format)


def _validate_split(split: SplitConfig) -> None:
    total = split.train_size + split.val_size + split.test_size
    if abs(total - 1.0) > _SPLIT_SUM_TOLERANCE:
        raise SettingsError(
            "A soma de dataset.split.train_size, val_size e test_size deve ser igual a 1.0. "
            f"Valor atual: {total:.6f}."
        )

    if split.train_size <= 0 or split.val_size <= 0 or split.test_size <= 0:
        raise SettingsError(
            "As proporções de treino, validação e teste devem ser maiores que zero."
        )


def _validate_dataset_columns(dataset: DatasetConfig) -> None:
    columns = dataset.id_column, dataset.text_column, dataset.target_column
    if len(set(columns)) != len(columns):
        raise SettingsError(
            "dataset.id_column, dataset.text_column e dataset.target_column "
            "devem apontar para colunas distintas."
        )


def _required_mapping(
    mapping: Mapping[str, Any],
    key: str,
    parent: str | None = None,
) -> Mapping[str, Any]:
    value = mapping.get(key)
    path = _join_key(parent, key)
    if not isinstance(value, Mapping):
        raise SettingsError(f"Seção obrigatória ausente ou inválida: {path}")
    return value


def _required_str(
    mapping: Mapping[str, Any], key: str, parent: str | None = None
) -> str:
    value = mapping.get(key)
    path = _join_key(parent, key)
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"Campo obrigatório ausente ou inválido: {path}")
    return value.strip()


def _required_int(
    mapping: Mapping[str, Any],
    key: str,
    parent: str | None = None,
    minimum: int | None = None,
) -> int:
    value = mapping.get(key)
    path = _join_key(parent, key)
    if not _is_plain_int(value):
        raise SettingsError(f"Campo obrigatório deve ser inteiro: {path}")
    if minimum is not None and value < minimum:
        raise SettingsError(f"Campo {path} deve ser maior ou igual a {minimum}.")
    return value


def _required_float(
    mapping: Mapping[str, Any],
    key: str,
    parent: str | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = mapping.get(key)
    path = _join_key(parent, key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SettingsError(f"Campo obrigatório deve ser numérico: {path}")

    value_float = float(value)
    if not math.isfinite(value_float):
        raise SettingsError(f"Campo {path} deve conter um número finito.")
    if minimum is not None and value_float < minimum:
        raise SettingsError(f"Campo {path} deve ser maior ou igual a {minimum}.")
    if maximum is not None and value_float > maximum:
        raise SettingsError(f"Campo {path} deve ser menor ou igual a {maximum}.")

    return value_float


def _required_bool(
    mapping: Mapping[str, Any], key: str, parent: str | None = None
) -> bool:
    value = mapping.get(key)
    path = _join_key(parent, key)
    if not isinstance(value, bool):
        raise SettingsError(f"Campo obrigatório deve ser booleano: {path}")
    return value


def _required_unique_str_tuple(
    mapping: Mapping[str, Any],
    key: str,
    parent: str | None = None,
) -> tuple[str, ...]:
    value = mapping.get(key)
    path = _join_key(parent, key)
    if not isinstance(value, Sequence) or isinstance(value, str) or not value:
        raise SettingsError(f"Campo obrigatório deve ser uma lista não vazia: {path}")

    items: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise SettingsError(
                f"Todos os itens de {path} devem ser textos não vazios."
            )

        normalized = item.strip()
        if normalized in seen:
            raise SettingsError(
                f"Item duplicado em {path}: {normalized!r} na posição {index}."
            )

        seen.add(normalized)
        items.append(normalized)

    return tuple(items)


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _find_project_root(start: Path) -> Path:
    current = start.expanduser().resolve()
    candidates = (current, *current.parents)

    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "configs"
        ).is_dir():
            return candidate

    return current


def _join_key(parent: str | None, key: str) -> str:
    return key if parent is None else f"{parent}.{key}"


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
