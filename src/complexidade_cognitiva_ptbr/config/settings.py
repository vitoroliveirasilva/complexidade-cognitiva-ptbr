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
    """Erro gerado quando a configuração do projeto é inválida"""


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


# Configurações de validação cruzada
@dataclass(frozen=True)
class CrossValidationConfig:
    enabled: bool
    method: str
    n_splits: int
    n_repeats: int
    shuffle: bool
    random_state: int
    scoring: tuple[str, ...]


# Configurações de verificação de vazamento experimental
@dataclass(frozen=True)
class LeakageChecksConfig:
    enabled: bool
    check_id_overlap: bool
    check_exact_text_overlap: bool
    check_normalized_text_overlap: bool
    check_near_duplicates: bool
    near_duplicate_threshold: float
    check_target_like_columns: bool
    fail_on_critical_leakage: bool


# Configurações de busca de hiperparâmetros
@dataclass(frozen=True)
class HyperparameterSearchConfig:
    enabled: bool
    strategy: str
    refit_metric: str
    n_jobs: int
    verbose: int
    save_all_results: bool
    search_spaces: dict[str, dict[str, tuple[Any, ...]]]


# Configurações de explicabilidade
@dataclass(frozen=True)
class ExplainabilityConfig:
    enabled: bool
    top_n_terms_per_class: int
    generate_global_tfidf_coefficients: bool
    generate_local_explanations: bool
    sample_predictions_per_class: int
    output_format: tuple[str, ...]


# Configurações de versionamento e rastreabilidade de execuções
@dataclass(frozen=True)
class RunTrackingConfig:
    enabled: bool
    run_id_format: str
    create_latest_pointer: bool
    copy_config_snapshot: bool
    save_environment: bool
    save_dataset_fingerprint: bool
    save_git_commit: bool


# Configurações de inferência local
@dataclass(frozen=True)
class InferenceConfig:
    enabled: bool
    default_model_path: Path
    include_probabilities: bool
    include_linguistic_metrics: bool
    include_explanations: bool


# Configurações de relatórios visuais
@dataclass(frozen=True)
class VisualReportsConfig:
    enabled: bool
    dpi: int
    generate_normalized_confusion_matrix: bool
    generate_experiment_comparison_chart: bool
    generate_cv_summary_chart: bool
    generate_feature_distribution_charts: bool
    generate_top_terms_chart: bool
    generate_prediction_confidence_chart: bool


# Configurações dos diretórios de saída
@dataclass(frozen=True)
class OutputsConfig:
    processed_dir: Path
    model_dir: Path
    metrics_dir: Path
    figures_dir: Path
    reports_dir: Path
    runs_dir: Path
    latest_dir: Path


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
    cross_validation: CrossValidationConfig
    leakage_checks: LeakageChecksConfig
    hyperparameter_search: HyperparameterSearchConfig
    explainability: ExplainabilityConfig
    run_tracking: RunTrackingConfig
    inference: InferenceConfig
    visual_reports: VisualReportsConfig
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
            self.outputs.runs_dir,
            self.outputs.latest_dir,
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
    cross_validation = _parse_cross_validation(raw_config, project.random_state)
    leakage_checks = _parse_leakage_checks(raw_config)
    hyperparameter_search = _parse_hyperparameter_search(raw_config)
    explainability = _parse_explainability(raw_config)
    run_tracking = _parse_run_tracking(raw_config)
    inference = _parse_inference(raw_config, project_root)
    visual_reports = _parse_visual_reports(raw_config)
    outputs = _parse_outputs(raw_config, project_root)
    logging = _parse_logging(raw_config)

    settings = AppSettings(
        config_path=resolved_config_path,
        project_root=project_root,
        project=project,
        dataset=dataset,
        preprocessing=preprocessing,
        features=features,
        training=training,
        cross_validation=cross_validation,
        leakage_checks=leakage_checks,
        hyperparameter_search=hyperparameter_search,
        explainability=explainability,
        run_tracking=run_tracking,
        inference=inference,
        visual_reports=visual_reports,
        outputs=outputs,
        logging=logging,
    )

    _validate_dataset_columns(dataset)

    from .validation import validate_settings

    validate_settings(settings)
    return settings


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
        raise SettingsError(f"YAML de configuração inválido em {config_path}: {exc}") from exc
    except OSError as exc:
        raise SettingsError(f"Erro ao ler o arquivo de configuração: {config_path}") from exc

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
        normalize_whitespace=_required_bool(section, "normalize_whitespace", "preprocessing"),
    )


def _parse_features(raw_config: Mapping[str, Any], project_root: Path) -> FeaturesConfig:
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
        representations=_required_unique_str_tuple(section, "representations", "training"),
        selection_metric=_required_str(section, "selection_metric", "training"),
    )


def _parse_cross_validation(
    raw_config: Mapping[str, Any], default_random_state: int
) -> CrossValidationConfig:
    section = _optional_mapping(raw_config, "cross_validation")
    return CrossValidationConfig(
        enabled=_optional_bool(section, "enabled", True),
        method=_optional_str(section, "method", "stratified_kfold"),
        n_splits=_optional_int(section, "n_splits", 5, minimum=2),
        n_repeats=_optional_int(section, "n_repeats", 1, minimum=1),
        shuffle=_optional_bool(section, "shuffle", True),
        random_state=_optional_int(
            section, "random_state", default_random_state, minimum=0
        ),
        scoring=_optional_unique_str_tuple(
            section,
            "scoring",
            (
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "f1_weighted",
                "balanced_accuracy",
            ),
        ),
    )


def _parse_leakage_checks(raw_config: Mapping[str, Any]) -> LeakageChecksConfig:
    section = _optional_mapping(raw_config, "leakage_checks")
    return LeakageChecksConfig(
        enabled=_optional_bool(section, "enabled", True),
        check_id_overlap=_optional_bool(section, "check_id_overlap", True),
        check_exact_text_overlap=_optional_bool(section, "check_exact_text_overlap", True),
        check_normalized_text_overlap=_optional_bool(
            section, "check_normalized_text_overlap", True
        ),
        check_near_duplicates=_optional_bool(section, "check_near_duplicates", True),
        near_duplicate_threshold=_optional_float(
            section, "near_duplicate_threshold", 0.92, minimum=0.0, maximum=1.0
        ),
        check_target_like_columns=_optional_bool(
            section, "check_target_like_columns", True
        ),
        fail_on_critical_leakage=_optional_bool(
            section, "fail_on_critical_leakage", True
        ),
    )


def _parse_hyperparameter_search(
    raw_config: Mapping[str, Any]
) -> HyperparameterSearchConfig:
    section = _optional_mapping(raw_config, "hyperparameter_search")
    spaces = section.get("search_spaces", _default_search_spaces())
    if not isinstance(spaces, Mapping):
        raise SettingsError("hyperparameter_search.search_spaces deve ser um mapeamento.")

    return HyperparameterSearchConfig(
        enabled=_optional_bool(section, "enabled", True),
        strategy=_optional_str(section, "strategy", "grid"),
        refit_metric=_optional_str(section, "refit_metric", "f1_macro"),
        n_jobs=_optional_int(section, "n_jobs", -1),
        verbose=_optional_int(section, "verbose", 1, minimum=0),
        save_all_results=_optional_bool(section, "save_all_results", True),
        search_spaces=_normalize_search_spaces(spaces),
    )


def _parse_explainability(raw_config: Mapping[str, Any]) -> ExplainabilityConfig:
    section = _optional_mapping(raw_config, "explainability")
    return ExplainabilityConfig(
        enabled=_optional_bool(section, "enabled", True),
        top_n_terms_per_class=_optional_int(
            section, "top_n_terms_per_class", 30, minimum=1
        ),
        generate_global_tfidf_coefficients=_optional_bool(
            section, "generate_global_tfidf_coefficients", True
        ),
        generate_local_explanations=_optional_bool(
            section, "generate_local_explanations", True
        ),
        sample_predictions_per_class=_optional_int(
            section, "sample_predictions_per_class", 10, minimum=1
        ),
        output_format=_optional_unique_str_tuple(section, "output_format", ("json", "csv", "md")),
    )


def _parse_run_tracking(raw_config: Mapping[str, Any]) -> RunTrackingConfig:
    section = _optional_mapping(raw_config, "run_tracking")
    return RunTrackingConfig(
        enabled=_optional_bool(section, "enabled", True),
        run_id_format=_optional_str(section, "run_id_format", "%Y%m%d_%H%M%S"),
        create_latest_pointer=_optional_bool(section, "create_latest_pointer", True),
        copy_config_snapshot=_optional_bool(section, "copy_config_snapshot", True),
        save_environment=_optional_bool(section, "save_environment", True),
        save_dataset_fingerprint=_optional_bool(
            section, "save_dataset_fingerprint", True
        ),
        save_git_commit=_optional_bool(section, "save_git_commit", True),
    )


def _parse_inference(
    raw_config: Mapping[str, Any], project_root: Path
) -> InferenceConfig:
    section = _optional_mapping(raw_config, "inference")
    return InferenceConfig(
        enabled=_optional_bool(section, "enabled", True),
        default_model_path=_resolve_project_path(
            project_root,
            _optional_str(
                section,
                "default_model_path",
                "outputs/latest/models/best_model_bundle.joblib",
            ),
        ),
        include_probabilities=_optional_bool(section, "include_probabilities", True),
        include_linguistic_metrics=_optional_bool(
            section, "include_linguistic_metrics", True
        ),
        include_explanations=_optional_bool(section, "include_explanations", True),
    )


def _parse_visual_reports(raw_config: Mapping[str, Any]) -> VisualReportsConfig:
    section = _optional_mapping(raw_config, "visual_reports")
    return VisualReportsConfig(
        enabled=_optional_bool(section, "enabled", True),
        dpi=_optional_int(section, "dpi", 160, minimum=72),
        generate_normalized_confusion_matrix=_optional_bool(
            section, "generate_normalized_confusion_matrix", True
        ),
        generate_experiment_comparison_chart=_optional_bool(
            section, "generate_experiment_comparison_chart", True
        ),
        generate_cv_summary_chart=_optional_bool(
            section, "generate_cv_summary_chart", True
        ),
        generate_feature_distribution_charts=_optional_bool(
            section, "generate_feature_distribution_charts", True
        ),
        generate_top_terms_chart=_optional_bool(section, "generate_top_terms_chart", True),
        generate_prediction_confidence_chart=_optional_bool(
            section, "generate_prediction_confidence_chart", True
        ),
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
        runs_dir=_resolve_project_path(
            project_root,
            _optional_str(section, "runs_dir", "outputs/runs"),
        ),
        latest_dir=_resolve_project_path(
            project_root,
            _optional_str(section, "latest_dir", "outputs/latest"),
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


def _optional_mapping(
    mapping: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = mapping.get(key, {})
    if not isinstance(value, Mapping):
        raise SettingsError(f"Seção opcional inválida: {key}")
    return value


def _required_str(
    mapping: Mapping[str, Any], key: str, parent: str | None = None
) -> str:
    value = mapping.get(key)
    path = _join_key(parent, key)
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"Campo obrigatório ausente ou inválido: {path}")
    return value.strip()


def _optional_str(mapping: Mapping[str, Any], key: str, default: str) -> str:
    value = mapping.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"Campo deve ser texto não vazio: {key}")
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


def _optional_int(
    mapping: Mapping[str, Any],
    key: str,
    default: int,
    minimum: int | None = None,
) -> int:
    value = mapping.get(key, default)
    if not _is_plain_int(value):
        raise SettingsError(f"Campo deve ser inteiro: {key}")
    if minimum is not None and value < minimum:
        raise SettingsError(f"Campo {key} deve ser maior ou igual a {minimum}.")
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
    return _coerce_float(value, path, minimum=minimum, maximum=maximum)


def _optional_float(
    mapping: Mapping[str, Any],
    key: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = mapping.get(key, default)
    return _coerce_float(value, key, minimum=minimum, maximum=maximum)


def _coerce_float(
    value: object,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
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


def _optional_bool(mapping: Mapping[str, Any], key: str, default: bool) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise SettingsError(f"Campo deve ser booleano: {key}")
    return value


def _required_unique_str_tuple(
    mapping: Mapping[str, Any],
    key: str,
    parent: str | None = None,
) -> tuple[str, ...]:
    value = mapping.get(key)
    path = _join_key(parent, key)
    return _coerce_unique_str_tuple(value, path)


def _optional_unique_str_tuple(
    mapping: Mapping[str, Any],
    key: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    value = mapping.get(key, default)
    return _coerce_unique_str_tuple(value, key)


def _coerce_unique_str_tuple(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str) or not value:
        raise SettingsError(f"Campo obrigatório deve ser uma lista não vazia: {path}")

    items: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise SettingsError(f"Todos os itens de {path} devem ser textos não vazios.")

        normalized = item.strip()
        if normalized in seen:
            raise SettingsError(
                f"Item duplicado em {path}: {normalized!r} na posição {index}."
            )

        seen.add(normalized)
        items.append(normalized)

    return tuple(items)


def _normalize_search_spaces(
    raw_spaces: Mapping[str, Any]
) -> dict[str, dict[str, tuple[Any, ...]]]:
    spaces: dict[str, dict[str, tuple[Any, ...]]] = {}

    for model_name, params in raw_spaces.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise SettingsError("Nome de modelo inválido em hyperparameter_search.search_spaces.")
        if not isinstance(params, Mapping):
            raise SettingsError(
                "Cada item de hyperparameter_search.search_spaces deve mapear "
                f"parâmetros para listas. Item inválido: {model_name!r}."
            )

        normalized_params: dict[str, tuple[Any, ...]] = {}
        for param_name, values in params.items():
            if not isinstance(param_name, str) or not param_name.strip():
                raise SettingsError(
                    f"Nome de parâmetro inválido no espaço de busca de {model_name}."
                )
            if not isinstance(values, Sequence) or isinstance(values, str) or not values:
                raise SettingsError(
                    "Cada parâmetro de busca deve conter uma lista não vazia. "
                    f"Parâmetro inválido: {model_name}.{param_name}."
                )
            normalized_params[param_name.strip()] = tuple(values)

        spaces[model_name.strip()] = normalized_params

    return spaces


def _default_search_spaces() -> dict[str, dict[str, list[Any]]]:
    return {
        "logistic_regression": {
            "C": [0.1, 1.0, 3.0, 10.0],
            "max_iter": [1000, 2000],
            "class_weight": [None, "balanced"],
        },
        "linear_svm": {
            "C": [0.1, 1.0, 3.0, 10.0],
            "class_weight": [None, "balanced"],
        },
        "random_forest": {
            "n_estimators": [100, 300],
            "max_depth": [None, 10, 30],
            "min_samples_split": [2, 5],
            "class_weight": [None, "balanced"],
        },
    }


def _resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _find_project_root(start: Path) -> Path:
    current = start.expanduser().resolve()
    candidates = (current, *current.parents)

    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate

    return current


def _join_key(parent: str | None, key: str) -> str:
    return key if parent is None else f"{parent}.{key}"


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
