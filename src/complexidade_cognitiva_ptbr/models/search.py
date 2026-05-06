from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from ..config.settings import AppSettings
from ..data.io import write_json
from ..utils.paths import relative_to_root


class HyperparameterSearchError(RuntimeError):
    """Erro gerado quando a busca de hiperparâmetros não pode ser concluída"""


# Resultado em memória da busca de hiperparâmetros de um experimento
@dataclass(frozen=True)
class SearchOutcome:

    enabled: bool
    best_estimator: Pipeline
    best_params: dict[str, Any]
    best_score: float | None
    cv_results: list[dict[str, Any]]
    search_metadata: dict[str, Any]


# Artefatos consolidados de busca de hiperparâmetros
@dataclass(frozen=True)
class HyperparameterSearchArtifacts:

    results_path: Path
    best_params_path: Path
    report_path: Path
    best_hyperparameters: dict[str, Any]

    def to_dict(self, project_root: Path | None = None) -> dict[str, Any]:
        if project_root is None:
            return {
                "results_path": self.results_path.as_posix(),
                "best_params_path": self.best_params_path.as_posix(),
                "report_path": self.report_path.as_posix(),
                "best_hyperparameters": self.best_hyperparameters,
            }

        return {
            "results_path": relative_to_root(self.results_path, project_root),
            "best_params_path": relative_to_root(self.best_params_path, project_root),
            "report_path": relative_to_root(self.report_path, project_root),
            "best_hyperparameters": self.best_hyperparameters,
        }


# Executa busca configurável para um único experimento, usando somente o conjunto de treino
def run_hyperparameter_search_for_experiment(
    settings: AppSettings,
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    *,
    model_name: str,
    representation: str,
    target_column: str,
    text_column: str,
    feature_columns: list[str],
) -> SearchOutcome:
    if not settings.hyperparameter_search.enabled:
        return SearchOutcome(
            enabled=False,
            best_estimator=pipeline,
            best_params={},
            best_score=None,
            cv_results=[],
            search_metadata={"enabled": False, "reason": "hyperparameter_search.enabled=false"},
        )

    param_grid = build_classifier_param_grid(settings, pipeline, model_name)
    if not param_grid:
        return SearchOutcome(
            enabled=True,
            best_estimator=pipeline,
            best_params={},
            best_score=None,
            cv_results=[],
            search_metadata={
                "enabled": True,
                "skipped": True,
                "reason": f"Nenhum hiperparâmetro configurado para {model_name}.",
            },
        )

    x_train = _prepare_model_frame(train_df, target_column, text_column, feature_columns)
    y_train = train_df[target_column].astype(str)
    cv = _build_search_cv(settings, y_train)
    search = _build_search_estimator(settings, pipeline, param_grid, cv)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            search.fit(x_train, y_train)
    except Exception as exc:
        raise HyperparameterSearchError(
            f"Falha na busca de hiperparâmetros para {representation}__{model_name}: {exc}"
        ) from exc

    best_params = _strip_classifier_prefix(dict(search.best_params_))
    best_score = float(search.best_score_) if math.isfinite(float(search.best_score_)) else None
    cv_results = build_cv_results_rows(
        search.cv_results_,
        experiment_id=f"{representation}__{model_name}",
        representation=representation,
        model_name=model_name,
    )

    return SearchOutcome(
        enabled=True,
        best_estimator=search.best_estimator_,
        best_params=best_params,
        best_score=round(best_score, 6) if best_score is not None else None,
        cv_results=cv_results,
        search_metadata={
            "enabled": True,
            "strategy": settings.hyperparameter_search.strategy,
            "refit_metric": settings.hyperparameter_search.refit_metric,
            "cv_n_splits": cv.n_splits,
            "candidate_count": len(cv_results),
            "best_params": best_params,
            "best_score": round(best_score, 6) if best_score is not None else None,
        },
    )


# Normaliza os espaços de busca do YAML para nomes de parâmetro aceitos pelo Pipeline
def build_classifier_param_grid(
    settings: AppSettings,
    pipeline: Pipeline,
    model_name: str,
) -> dict[str, tuple[Any, ...]]:
    configured_space = settings.hyperparameter_search.search_spaces.get(model_name, {})
    if not isinstance(configured_space, Mapping):
        raise HyperparameterSearchError(
            f"Espaço de busca inválido para o modelo '{model_name}'."
        )

    valid_params = set(pipeline.get_params(deep=True))
    normalized: dict[str, tuple[Any, ...]] = {}
    ignored: list[str] = []

    for raw_name, raw_values in configured_space.items():
        parameter_name = raw_name if str(raw_name).startswith("classifier__") else f"classifier__{raw_name}"
        if parameter_name not in valid_params:
            ignored.append(str(raw_name))
            continue
        values = tuple(raw_values) if isinstance(raw_values, tuple | list) else (raw_values,)
        if not values:
            continue
        normalized[parameter_name] = values

    if configured_space and not normalized:
        raise HyperparameterSearchError(
            f"Nenhum parâmetro configurado para '{model_name}' é aceito pelo Pipeline. "
            f"Parâmetros ignorados: {', '.join(ignored)}."
        )

    return normalized


# Converte cv_results_ do sklearn para linhas serializáveis e estáveis
def build_cv_results_rows(
    cv_results: dict[str, Any],
    *,
    experiment_id: str,
    representation: str,
    model_name: str,
) -> list[dict[str, Any]]:
    params_list = cv_results.get("params", [])
    rows: list[dict[str, Any]] = []

    for index, params in enumerate(params_list):
        row: dict[str, Any] = {
            "experiment_id": experiment_id,
            "representation": representation,
            "model_name": model_name,
            "rank": _safe_cv_value(cv_results, "rank_test_score", index),
            "mean_test_score": _safe_cv_value(cv_results, "mean_test_score", index),
            "std_test_score": _safe_cv_value(cv_results, "std_test_score", index),
            "mean_fit_time": _safe_cv_value(cv_results, "mean_fit_time", index),
            "std_fit_time": _safe_cv_value(cv_results, "std_fit_time", index),
            "params": json.dumps(_strip_classifier_prefix(dict(params)), ensure_ascii=False, sort_keys=True),
        }
        for key, values in cv_results.items():
            if key.startswith("mean_test_") or key.startswith("std_test_") or key.startswith("rank_test_"):
                row[key] = _safe_cv_value(cv_results, key, index)
        rows.append(row)

    return rows


# Persiste resultados consolidados da busca em CSV, JSON e Markdown
def persist_hyperparameter_search_artifacts(
    search_rows: list[dict[str, Any]],
    best_hyperparameters: dict[str, Any],
    *,
    metrics_dir: Path,
    reports_dir: Path,
) -> HyperparameterSearchArtifacts:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    results_path = metrics_dir / "hyperparameter_search_results.csv"
    best_params_path = metrics_dir / "best_hyperparameters.json"
    report_path = reports_dir / "hyperparameter_search_report.md"

    pd.DataFrame(search_rows).to_csv(results_path, index=False, encoding="utf-8")
    write_json(best_hyperparameters, best_params_path)
    report_path.write_text(_build_search_markdown(best_hyperparameters), encoding="utf-8")

    return HyperparameterSearchArtifacts(
        results_path=results_path,
        best_params_path=best_params_path,
        report_path=report_path,
        best_hyperparameters=best_hyperparameters,
    )


def build_disabled_search_artifacts(
    *,
    settings: AppSettings,
    metrics_dir: Path,
    reports_dir: Path,
) -> HyperparameterSearchArtifacts:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "enabled": False,
        "reason": "hyperparameter_search.enabled=false",
        "project": {"name": settings.project.name, "version": settings.project.version},
    }
    return persist_hyperparameter_search_artifacts([], payload, metrics_dir=metrics_dir, reports_dir=reports_dir)


def _prepare_model_frame(
    df: pd.DataFrame,
    target_column: str,
    text_column: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    frame = df.drop(columns=[target_column]).copy()
    if text_column in frame.columns:
        frame[text_column] = frame[text_column].astype("string").fillna("").astype(str)
    for column in feature_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _build_search_cv(settings: AppSettings, y_train: pd.Series) -> StratifiedKFold:
    class_counts = y_train.astype(str).value_counts()
    if class_counts.empty or len(class_counts) < 2:
        raise HyperparameterSearchError("A busca exige pelo menos duas classes no treino.")

    min_class_count = int(class_counts.min())
    n_splits = min(settings.cross_validation.n_splits, min_class_count)
    if n_splits < 2:
        raise HyperparameterSearchError(
            "A menor classe do treino precisa ter pelo menos 2 amostras para busca com CV."
        )

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=settings.cross_validation.shuffle,
        random_state=settings.cross_validation.random_state if settings.cross_validation.shuffle else None,
    )


def _build_search_estimator(
    settings: AppSettings,
    pipeline: Pipeline,
    param_grid: dict[str, tuple[Any, ...]],
    cv: StratifiedKFold,
) -> GridSearchCV | RandomizedSearchCV:
    common_kwargs = {
        "estimator": pipeline,
        "scoring": settings.hyperparameter_search.refit_metric,
        "cv": cv,
        "refit": True,
        "n_jobs": settings.hyperparameter_search.n_jobs,
        "verbose": settings.hyperparameter_search.verbose,
        "error_score": "raise",
        "return_train_score": False,
    }

    if settings.hyperparameter_search.strategy == "randomized":
        candidate_count = len(list(ParameterGrid(param_grid)))
        n_iter = max(1, min(10, candidate_count))
        return RandomizedSearchCV(
            param_distributions=param_grid,
            n_iter=n_iter,
            random_state=settings.project.random_state,
            **common_kwargs,
        )

    return GridSearchCV(param_grid=param_grid, **common_kwargs)


def _strip_classifier_prefix(params: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in params.items():
        clean[str(key).removeprefix("classifier__")] = _json_ready(value)
    return clean


def _safe_cv_value(cv_results: dict[str, Any], key: str, index: int) -> Any:
    values = cv_results.get(key)
    if values is None:
        return None
    try:
        value = values[index]
    except (IndexError, TypeError):
        return None
    return _json_ready(value)


def _json_ready(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, int | str | bool) or value is None:
        return value
    return str(value)


def _build_search_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Relatório de busca de hiperparâmetros",
        "",
        f"Gerado em UTC: {payload.get('generated_at_utc', 'n/d')}",
        "",
        f"Busca habilitada: {payload.get('enabled')}",
        f"Métrica de refit: {payload.get('refit_metric', 'n/d')}",
        "",
        "## Melhor configuração",
        "",
    ]
    best = payload.get("best_experiment")
    if isinstance(best, Mapping):
        lines.extend(
            [
                f"- Experimento: `{best.get('experiment_id')}`",
                f"- Representação: `{best.get('representation')}`",
                f"- Modelo: `{best.get('model_name')}`",
                f"- Score médio CV: `{best.get('best_search_score')}`",
                f"- Hiperparâmetros: `{json.dumps(best.get('best_params', {}), ensure_ascii=False, sort_keys=True)}`",
            ]
        )
    else:
        lines.append("Nenhuma busca executada ou nenhuma configuração selecionada.")

    lines.extend(["", "## Observação metodológica", ""])
    lines.append(
        "A busca é executada apenas no conjunto de treino, com validação cruzada interna. "
        "O conjunto de teste permanece isolado até a avaliação final."
    )
    lines.append("")
    return "\n".join(lines)
