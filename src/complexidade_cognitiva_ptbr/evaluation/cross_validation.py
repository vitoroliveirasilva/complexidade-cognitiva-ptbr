from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from ..config.settings import AppSettings
from ..data.io import write_json
from ..models.train import (
    TrainingError,
    build_training_pipeline,
    compute_classification_metrics,
    load_training_frames,
    prepare_model_frame,
    validate_training_frames,
)
from ..utils.paths import relative_to_root


class CrossValidationError(RuntimeError):
    """Erro gerado quando a validação cruzada não pode ser executada"""


# Artefatos e resumo da validação cruzada
@dataclass(frozen=True)
class CrossValidationResult:

    results_path: Path
    summary_path: Path
    report_path: Path
    figure_path: Path
    summary: dict[str, Any]

    def to_dict(self, project_root: Path | None = None) -> dict[str, Any]:
        if project_root is None:
            return {
                "results_path": self.results_path.as_posix(),
                "summary_path": self.summary_path.as_posix(),
                "report_path": self.report_path.as_posix(),
                "figure_path": self.figure_path.as_posix(),
                "summary": self.summary,
            }
        return {
            "results_path": relative_to_root(self.results_path, project_root),
            "summary_path": relative_to_root(self.summary_path, project_root),
            "report_path": relative_to_root(self.report_path, project_root),
            "figure_path": relative_to_root(self.figure_path, project_root),
            "summary": self.summary,
        }


# Executa CV estratificada sobre treino+validação, mantendo o teste final isolado
def run_cross_validation(
    settings: AppSettings,
    *,
    metrics_dir: Path | None = None,
    reports_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> CrossValidationResult:
    metrics_dir = metrics_dir or settings.outputs.metrics_dir
    reports_dir = reports_dir or settings.outputs.reports_dir
    figures_dir = figures_dir or settings.outputs.figures_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not settings.cross_validation.enabled:
        summary = _disabled_summary(settings)
        return _persist_outputs(pd.DataFrame(), summary, metrics_dir, reports_dir, figures_dir)

    try:
        train_df, val_df = load_training_frames(settings)
        feature_columns = validate_training_frames(train_df, val_df, settings)
    except TrainingError as exc:
        raise CrossValidationError(f"Não foi possível carregar dados para CV: {exc}") from exc

    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    target_column = settings.dataset.target_column
    text_column = _resolve_text_column(combined_df, settings)
    _validate_cv_input(combined_df, target_column, settings.cross_validation.n_splits)

    rows: list[dict[str, Any]] = []
    splitter = _build_splitter(settings)

    x_all = combined_df.reset_index(drop=True)
    y_all = x_all[target_column].astype(str).reset_index(drop=True)

    for representation in settings.training.representations:
        for model_name in settings.training.models:
            experiment_id = f"{representation}__{model_name}"
            for fold_index, (train_idx, val_idx) in enumerate(splitter.split(x_all, y_all), start=1):
                fold_train = x_all.iloc[train_idx].copy()
                fold_val = x_all.iloc[val_idx].copy()
                pipeline = build_training_pipeline(
                    settings=settings,
                    representation=representation,
                    model_name=model_name,
                    feature_columns=feature_columns,
                    text_column=text_column,
                )
                started = time.perf_counter()
                try:
                    x_train = prepare_model_frame(
                        fold_train, target_column, text_column, feature_columns
                    )
                    y_train = fold_train[target_column].astype(str)
                    x_val = prepare_model_frame(
                        fold_val, target_column, text_column, feature_columns
                    )
                    y_val = fold_val[target_column].astype(str)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)
                        pipeline.fit(x_train, y_train)
                        predictions = [str(value) for value in pipeline.predict(x_val)]
                except Exception as exc:
                    raise CrossValidationError(
                        f"Falha na CV do experimento '{experiment_id}', fold {fold_index}: {exc}"
                    ) from exc

                elapsed = round(time.perf_counter() - started, 6)
                metrics = compute_classification_metrics(y_val.tolist(), predictions)
                row = {
                    "experiment_id": experiment_id,
                    "representation": representation,
                    "model_name": model_name,
                    "fold": fold_index,
                    "train_rows": int(len(fold_train)),
                    "validation_rows": int(len(fold_val)),
                    "fit_seconds": elapsed,
                    "train_class_distribution": json.dumps(
                        _class_distribution(fold_train, target_column), ensure_ascii=False
                    ),
                    "validation_class_distribution": json.dumps(
                        _class_distribution(fold_val, target_column), ensure_ascii=False
                    ),
                }
                for metric_name in settings.cross_validation.scoring:
                    if metric_name in metrics:
                        row[metric_name] = metrics[metric_name]
                rows.append(row)

    results_df = pd.DataFrame(rows)
    summary = build_cv_summary(results_df, settings)
    return _persist_outputs(results_df, summary, metrics_dir, reports_dir, figures_dir)


# Resume resultados por experimento com média, desvio, mínimo e máximo por métrica
def build_cv_summary(results_df: pd.DataFrame, settings: AppSettings) -> dict[str, Any]:
    if results_df.empty:
        return _disabled_summary(settings)

    metrics = [metric for metric in settings.cross_validation.scoring if metric in results_df.columns]
    experiments: list[dict[str, Any]] = []
    grouped = results_df.groupby(["experiment_id", "representation", "model_name"], dropna=False)
    for (experiment_id, representation, model_name), group in grouped:
        metric_summary: dict[str, dict[str, float]] = {}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="raise")
            metric_summary[metric] = {
                "mean": round(float(values.mean()), 6),
                "std": round(float(values.std(ddof=0)), 6),
                "min": round(float(values.min()), 6),
                "max": round(float(values.max()), 6),
            }
        experiments.append(
            {
                "experiment_id": str(experiment_id),
                "representation": str(representation),
                "model_name": str(model_name),
                "folds": int(len(group)),
                "metrics": metric_summary,
                "fit_seconds_total": round(float(group["fit_seconds"].sum()), 6),
                "fit_seconds_mean": round(float(group["fit_seconds"].mean()), 6),
            }
        )

    selection_metric = settings.training.selection_metric
    ranked = sorted(
        experiments,
        key=lambda item: (
            item["metrics"].get(selection_metric, {}).get("mean", -1.0),
            item["metrics"].get("f1_weighted", {}).get("mean", -1.0),
            item["metrics"].get("accuracy", {}).get("mean", -1.0),
        ),
        reverse=True,
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": {"name": settings.project.name, "version": settings.project.version},
        "config": {
            "enabled": settings.cross_validation.enabled,
            "method": settings.cross_validation.method,
            "n_splits": settings.cross_validation.n_splits,
            "n_repeats": settings.cross_validation.n_repeats,
            "shuffle": settings.cross_validation.shuffle,
            "random_state": settings.cross_validation.random_state,
            "scoring": list(settings.cross_validation.scoring),
            "selection_metric": settings.training.selection_metric,
        },
        "summary": {
            "total_rows": int(len(results_df)),
            "total_experiments": int(len(experiments)),
            "best_experiment_id": ranked[0]["experiment_id"] if ranked else None,
            "best_selection_metric_mean": ranked[0]["metrics"].get(selection_metric, {}).get("mean")
            if ranked
            else None,
        },
        "experiments": ranked,
    }


def _persist_outputs(
    results_df: pd.DataFrame,
    summary: dict[str, Any],
    metrics_dir: Path,
    reports_dir: Path,
    figures_dir: Path,
) -> CrossValidationResult:
    results_path = metrics_dir / "cv_results.csv"
    summary_path = write_json(summary, metrics_dir / "cv_summary.json")
    report_path = reports_dir / "cv_report.md"
    figure_path = figures_dir / "cv_metrics_comparison.png"

    results_df.to_csv(results_path, index=False, encoding="utf-8")
    report_path.write_text(_build_cv_markdown(summary), encoding="utf-8")
    _save_cv_figure(summary, figure_path)

    return CrossValidationResult(results_path, summary_path, report_path, figure_path, summary)


def _build_splitter(settings: AppSettings):
    cv = settings.cross_validation
    if cv.method == "repeated_stratified_kfold":
        return RepeatedStratifiedKFold(
            n_splits=cv.n_splits,
            n_repeats=cv.n_repeats,
            random_state=cv.random_state,
        )
    return StratifiedKFold(
        n_splits=cv.n_splits,
        shuffle=cv.shuffle,
        random_state=cv.random_state if cv.shuffle else None,
    )


def _validate_cv_input(df: pd.DataFrame, target_column: str, n_splits: int) -> None:
    if df.empty:
        raise CrossValidationError("O conjunto usado para CV está vazio.")
    class_counts = df[target_column].astype(str).value_counts()
    if len(class_counts) < 2:
        raise CrossValidationError("A validação cruzada exige pelo menos duas classes.")
    min_class_count = int(class_counts.min())
    if min_class_count < n_splits:
        raise CrossValidationError(
            "cross_validation.n_splits é maior que a menor classe disponível. "
            f"n_splits={n_splits}, menor_classe={min_class_count}."
        )


def _resolve_text_column(df: pd.DataFrame, settings: AppSettings) -> str:
    clean = settings.preprocessing.clean_text_column
    if clean in df.columns:
        return clean
    fallback = settings.dataset.text_column
    if fallback in df.columns:
        return fallback
    raise CrossValidationError(f"Nenhuma coluna textual encontrada: {clean} ou {fallback}.")


def _class_distribution(df: pd.DataFrame, target_column: str) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in df[target_column]
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
        .items()
    }


def _build_cv_markdown(summary: dict[str, Any]) -> str:
    status = "disabled" if summary.get("summary", {}).get("total_experiments", 0) == 0 else "executed"
    lines = [
        "# Relatório de validação cruzada",
        "",
        f"Gerado em UTC: `{summary.get('generated_at_utc', '')}`",
        "",
        "## Resumo",
        "",
        f"- Status: `{status}`",
        f"- Experimentos avaliados: {summary.get('summary', {}).get('total_experiments', 0)}",
        f"- Melhor experimento: `{summary.get('summary', {}).get('best_experiment_id')}`",
        f"- Média da métrica de seleção: `{summary.get('summary', {}).get('best_selection_metric_mean')}`",
        "",
        "## Experimentos",
        "",
    ]
    experiments = summary.get("experiments", [])
    if not experiments:
        lines.append("Validação cruzada desabilitada ou sem resultados.")
    for experiment in experiments:
        lines.append(f"### `{experiment['experiment_id']}`")
        lines.append("")
        for metric_name, values in experiment.get("metrics", {}).items():
            lines.append(
                f"- {metric_name}: média={values['mean']}, desvio={values['std']}, "
                f"mín={values['min']}, máx={values['max']}"
            )
        lines.append("")
    return "\n".join(lines)


def _save_cv_figure(summary: dict[str, Any], output_path: Path) -> None:
    experiments = summary.get("experiments", [])
    metric = summary.get("config", {}).get("selection_metric", "f1_macro")
    labels = [item["experiment_id"] for item in experiments]
    means = [item.get("metrics", {}).get(metric, {}).get("mean", 0.0) for item in experiments]
    stds = [item.get("metrics", {}).get(metric, {}).get("std", 0.0) for item in experiments]

    plt.figure(figsize=(max(8, len(labels) * 0.9), 5))
    if labels:
        plt.bar(range(len(labels)), means, yerr=stds)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.ylim(0, 1.05)
        plt.ylabel(metric)
        plt.title("Validação cruzada por experimento")
    else:
        plt.text(0.5, 0.5, "Validação cruzada sem resultados", ha="center", va="center")
        plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _disabled_summary(settings: AppSettings) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": {"name": settings.project.name, "version": settings.project.version},
        "config": {
            "enabled": False,
            "method": settings.cross_validation.method,
            "n_splits": settings.cross_validation.n_splits,
            "n_repeats": settings.cross_validation.n_repeats,
            "scoring": list(settings.cross_validation.scoring),
            "selection_metric": settings.training.selection_metric,
        },
        "summary": {
            "total_rows": 0,
            "total_experiments": 0,
            "best_experiment_id": None,
            "best_selection_metric_mean": None,
        },
        "experiments": [],
    }


__all__ = [
    "CrossValidationError",
    "CrossValidationResult",
    "build_cv_summary",
    "run_cross_validation",
]
