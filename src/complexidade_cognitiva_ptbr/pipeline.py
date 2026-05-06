from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from .config.settings import AppSettings, load_settings
from .data.preprocessing import (
    PreparationResult,
    prepare_dataset,
)
from .data.leakage import LeakageCheckResult, run_leakage_checks
from .evaluation.reporting import (
    EvaluationResult,
    evaluate_model,
)
from .evaluation.cross_validation import CrossValidationResult, run_cross_validation
from .features.build_features import (
    FeatureBuildResult,
    build_feature_datasets,
)
from .models.train import TrainingResult, train_model
from .utils.logging_utils import (
    configure_logging,
    get_logger,
)
from .utils.paths import (
    ensure_project_directories,
    relative_to_root,
)
from .utils.run_context import RunContext

_JSON_INDENT = 2
_ResultT = TypeVar("_ResultT")


# Normalização de tipos comuns para serialização JSON, como Path e objetos com métodos item() ou tolist()
def _json_default(value: object) -> object:

    if isinstance(value, Path):
        return value.as_posix()

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except (TypeError, ValueError):
            pass

    return str(value)


# Converte payloads do pipeline para JSON legível e tolerante a tipos externos
def _to_pretty_json(payload: dict[str, Any]) -> str:
    
    return json.dumps(
        payload, ensure_ascii=False, indent=_JSON_INDENT, default=_json_default
    )


# Valida e normaliza o nome da etapa usado nos relatórios e logs
def _normalize_stage_name(stage: str) -> str:

    normalized_stage = stage.strip()
    if not normalized_stage:
        msg = "O nome da etapa do pipeline não pode ser vazio."
        raise ValueError(msg)
    return normalized_stage


# Resultado da validação estrutural do pipeline
@dataclass(frozen=True)
class PipelineBootstrapResult:

    project_name: str
    project_version: str
    stage: str
    config_path: str
    project_root: str
    managed_directories: tuple[str, ...]
    required_dataset_columns: tuple[str, ...]
    random_state: int
    run_context: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:

        return {
            "project_name": self.project_name,
            "project_version": self.project_version,
            "stage": self.stage,
            "config_path": self.config_path,
            "project_root": self.project_root,
            "managed_directories": list(self.managed_directories),
            "required_dataset_columns": list(self.required_dataset_columns),
            "random_state": self.random_state,
            "run_context": self.run_context,
        }

    def to_json(self) -> str:
        
        return _to_pretty_json(self.to_dict())


# Resultado serializável de uma etapa executada pelo pipeline
@dataclass(frozen=True)
class PipelineStageResult:

    stage: str
    bootstrap: PipelineBootstrapResult
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        
        return {
            "stage": self.stage,
            "bootstrap": self.bootstrap.to_dict(),
            "payload": self.payload,
        }

    def to_json(self) -> str:

        return _to_pretty_json(self.to_dict())


# Resultado serializável da execução integrada do pipeline disponível
@dataclass(frozen=True)
class PipelineRunResult:

    stage: str
    stages: tuple[PipelineStageResult, ...]

    def to_dict(self) -> dict[str, Any]:

        return {
            "stage": self.stage,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def to_json(self) -> str:

        return _to_pretty_json(self.to_dict())


# Pipeline principal do projeto
class CognitiveComplexityPipeline:

    def __init__(self, settings: AppSettings) -> None:
        if settings is None:
            msg = "As configurações do pipeline não foram informadas."
            raise ValueError(msg)

        self.settings = settings
        self.run_context = RunContext.create(settings)
        self.logger = get_logger(self.__class__.__name__)

    # Constrói o pipeline a partir do arquivo de configuração
    @classmethod
    def from_config(
        cls, config_path: str | Path | None = None
    ) -> CognitiveComplexityPipeline:

        settings = load_settings(config_path)
        configure_logging(settings.logging.level, settings.logging.format)
        return cls(settings)

    # Valida a infraestrutura compartilhada do projeto e garante diretórios
    def bootstrap(self, stage: str = "base") -> PipelineBootstrapResult:

        normalized_stage = _normalize_stage_name(stage)
        directories = ensure_project_directories(self.settings)
        relative_directories = tuple(
            relative_to_root(directory, self.settings.project_root)
            for directory in directories
        )

        result = PipelineBootstrapResult(
            project_name=self.settings.project.name,
            project_version=self.settings.project.version,
            stage=normalized_stage,
            config_path=relative_to_root(
                self.settings.config_path, self.settings.project_root
            ),
            project_root=self.settings.project_root.as_posix(),
            managed_directories=relative_directories,
            required_dataset_columns=tuple(self.settings.required_dataset_columns),
            random_state=self.settings.project.random_state,
            run_context=self.run_context.to_dict(self.settings.project_root),
        )

        self.logger.info(
            "Estrutura validada para o estágio '%s' com %s diretórios gerenciados.",
            normalized_stage,
            len(relative_directories),
        )
        return result

    # Executa uma etapa com logs consistentes e preservação da exceção original
    def _execute_stage(
        self,
        stage: str,
        start_message: str,
        success_message: str,
        action: Callable[[], _ResultT],
    ) -> _ResultT:

        self.logger.info(start_message)
        try:
            result = action()
        except Exception:
            self.logger.exception("Falha ao executar o estágio '%s'.", stage)
            raise

        self.logger.info(success_message)
        return result

    # Executa a etapa de preparação dos dados
    def prepare_dataset(self) -> PipelineStageResult:

        stage = "prepare_dataset"
        bootstrap = self.bootstrap(stage=stage)
        input_path = relative_to_root(
            self.settings.dataset.input_path, self.settings.project_root
        )

        result: PreparationResult = self._execute_stage(
            stage=stage,
            start_message=f"Iniciando preparação do dataset bruto em '{input_path}'.",
            success_message="Preparação concluída.",
            action=lambda: prepare_dataset(self.settings),
        )
        self.logger.info(
            "Arquivos gerados: %s, %s, %s e %s.",
            relative_to_root(result.train_path, self.settings.project_root),
            relative_to_root(result.val_path, self.settings.project_root),
            relative_to_root(result.test_path, self.settings.project_root),
            relative_to_root(result.report_path, self.settings.project_root),
        )

        return PipelineStageResult(
            stage=stage,
            bootstrap=bootstrap,
            payload=result.to_dict(self.settings.project_root),
        )

    # Executa as verificações de vazamento entre treino, validação e teste
    def detect_leakage(self) -> PipelineStageResult:

        stage = "detect_leakage"
        bootstrap = self.bootstrap(stage=stage)

        result: LeakageCheckResult = self._execute_stage(
            stage=stage,
            start_message="Iniciando verificações de vazamento experimental entre splits.",
            success_message="Verificações de vazamento concluídas.",
            action=lambda: run_leakage_checks(
                self.settings,
                metrics_dir=self.run_context.metrics_dir,
                reports_dir=self.run_context.reports_dir,
            ),
        )
        self.logger.info(
            "Relatório de vazamento salvo em %s.",
            relative_to_root(result.report_json_path, self.settings.project_root),
        )

        return PipelineStageResult(
            stage=stage,
            bootstrap=bootstrap,
            payload=result.to_dict(self.settings.project_root),
        )

    # Executa a etapa de extração de features linguísticas
    def build_features(self) -> PipelineStageResult:

        stage = "build_features"
        bootstrap = self.bootstrap(stage=stage)

        result: FeatureBuildResult = self._execute_stage(
            stage=stage,
            start_message="Iniciando geração de métricas linguísticas interpretáveis.",
            success_message="Features concluídas.",
            action=lambda: build_feature_datasets(self.settings),
        )
        self.logger.info(
            "Arquivos gerados: %s, %s, %s e %s.",
            relative_to_root(result.train_features_path, self.settings.project_root),
            relative_to_root(result.val_features_path, self.settings.project_root),
            relative_to_root(result.test_features_path, self.settings.project_root),
            relative_to_root(result.metadata_path, self.settings.project_root),
        )

        return PipelineStageResult(
            stage=stage,
            bootstrap=bootstrap,
            payload=result.to_dict(self.settings.project_root),
        )

    # Executa validação cruzada estratificada no conjunto treino+validação
    def cross_validate(self) -> PipelineStageResult:

        stage = "cross_validate"
        bootstrap = self.bootstrap(stage=stage)

        result: CrossValidationResult = self._execute_stage(
            stage=stage,
            start_message="Iniciando validação cruzada estratificada.",
            success_message="Validação cruzada concluída.",
            action=lambda: run_cross_validation(
                self.settings,
                metrics_dir=self.run_context.metrics_dir,
                reports_dir=self.run_context.reports_dir,
                figures_dir=self.run_context.figures_dir,
            ),
        )
        self.logger.info(
            "Resultados de validação cruzada salvos em %s.",
            relative_to_root(result.results_path, self.settings.project_root),
        )

        return PipelineStageResult(
            stage=stage,
            bootstrap=bootstrap,
            payload=result.to_dict(self.settings.project_root),
        )

    # Executa a etapa de treinamento e seleção do melhor experimento
    def train_model(self) -> PipelineStageResult:

        stage = "train_model"
        bootstrap = self.bootstrap(stage=stage)

        result: TrainingResult = self._execute_stage(
            stage=stage,
            start_message="Iniciando treinamento e comparação de modelos supervisionados.",
            success_message="Treinamento concluído.",
            action=lambda: train_model(
                    self.settings,
                    model_dir=self.run_context.models_dir,
                    metrics_dir=self.run_context.metrics_dir,
                    reports_dir=self.run_context.reports_dir,
                    latest_dir=self.run_context.latest_dir,
                ),
        )
        self.logger.info(
            "Melhor modelo salvo em %s.",
            relative_to_root(result.best_model_path, self.settings.project_root),
        )

        return PipelineStageResult(
            stage=stage,
            bootstrap=bootstrap,
            payload=result.to_dict(self.settings.project_root),
        )

    # Executa a avaliação final do melhor modelo no conjunto de teste
    def evaluate_model(self) -> PipelineStageResult:

        stage = "evaluate_model"
        bootstrap = self.bootstrap(stage=stage)

        result: EvaluationResult = self._execute_stage(
            stage=stage,
            start_message="Iniciando avaliação final do melhor modelo no conjunto de teste.",
            success_message="Avaliação concluída.",
            action=lambda: evaluate_model(
                    self.settings,
                    model_dir=self.run_context.models_dir,
                    metrics_dir=self.run_context.metrics_dir,
                    figures_dir=self.run_context.figures_dir,
                    reports_dir=self.run_context.reports_dir,
                    explainability_dir=self.run_context.explainability_dir,
                ),
        )
        self.logger.info(
            "Relatório final salvo em %s.",
            relative_to_root(result.final_report_path, self.settings.project_root),
        )

        return PipelineStageResult(
            stage=stage,
            bootstrap=bootstrap,
            payload=result.to_dict(self.settings.project_root),
        )

    # Executa o pipeline completo de preparação, features, treino e avaliação
    def run(self) -> PipelineRunResult:

        stages = (
            self.prepare_dataset(),
            self.detect_leakage(),
            self.build_features(),
            self.cross_validate(),
            self.train_model(),
            self.evaluate_model(),
        )
        if self.settings.run_tracking.create_latest_pointer:
            self.run_context.update_latest_pointer()
        return PipelineRunResult(stage="run_pipeline", stages=stages)
