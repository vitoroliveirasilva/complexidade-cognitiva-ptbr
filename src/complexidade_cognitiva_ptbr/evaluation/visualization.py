from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

from ..config.settings import AppSettings
from ..features.build_features import get_feature_columns


class VisualizationError(RuntimeError):
    """Erro gerado quando um relatório visual não pode ser gerado"""


# Gera matriz de confusão absoluta ou normalizada
def save_confusion_matrix_plot(
    *,
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    output_path: Path,
    normalized: bool = False,
    dpi: int = 160,
) -> Path:
    _validate_vectors(y_true, y_pred, labels)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display_matrix: Any = matrix.astype(float)
    number_format = "d"
    title = "Matriz de confusão absoluta - conjunto de teste"
    if normalized:
        row_sums = display_matrix.sum(axis=1, keepdims=True)
        display_matrix = display_matrix / row_sums.clip(min=1.0)
        number_format = ".2f"
        title = "Matriz de confusão normalizada - conjunto de teste"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_width = max(6.0, min(18.0, len(labels) * 1.4))
    fig_height = max(5.0, min(16.0, len(labels) * 1.2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    try:
        image = ax.imshow(display_matrix)
        ax.set_title(title)
        ax.set_xlabel("Classe predita")
        ax.set_ylabel("Classe real")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for row_index in range(display_matrix.shape[0]):
            for column_index in range(display_matrix.shape[1]):
                value = display_matrix[row_index, column_index]
                label = format(int(value), number_format) if number_format == "d" else format(float(value), number_format)
                ax.text(column_index, row_index, label, ha="center", va="center")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return output_path


# Gráfico de barras comparando experimentos pela métrica configurada
def save_experiment_comparison_chart(
    *,
    experiment_results_path: Path,
    output_path: Path,
    metric: str = "f1_macro",
    dpi: int = 160,
) -> Path | None:
    if not experiment_results_path.exists():
        return None
    df = pd.read_csv(experiment_results_path)
    metric_column = f"validation_{metric}"
    if df.empty or metric_column not in df.columns:
        return None

    label_column = "experiment_id" if "experiment_id" in df.columns else df.columns[0]
    plot_df = df.sort_values(metric_column, ascending=True).tail(20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height = max(5.0, min(14.0, 0.35 * len(plot_df) + 1.8))
    fig, ax = plt.subplots(figsize=(10.0, height))
    try:
        ax.barh(plot_df[label_column].astype(str), pd.to_numeric(plot_df[metric_column], errors="coerce").fillna(0.0))
        ax.set_title(f"Comparação de experimentos por {metric}")
        ax.set_xlabel(metric)
        ax.set_ylabel("Experimento")
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return output_path


# Gráfico de validação cruzada com média da métrica de seleção por experimento
def save_cv_summary_chart(
    *,
    cv_summary_path: Path,
    output_path: Path,
    metric: str = "f1_macro",
    dpi: int = 160,
) -> Path | None:
    if not cv_summary_path.exists():
        return None
    try:
        payload = json.loads(cv_summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    experiments = payload.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        return None

    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        metrics = experiment.get("metrics", {}) if isinstance(experiment, dict) else {}
        metric_payload = metrics.get(metric, {}) if isinstance(metrics, dict) else {}
        if "mean" not in metric_payload:
            continue
        rows.append(
            {
                "experiment_id": str(experiment.get("experiment_id", "")),
                "mean": float(metric_payload.get("mean", 0.0)),
                "std": float(metric_payload.get("std", 0.0)),
            }
        )
    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("mean", ascending=True).tail(20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height = max(5.0, min(14.0, 0.35 * len(df) + 1.8))
    fig, ax = plt.subplots(figsize=(10.0, height))
    try:
        ax.barh(df["experiment_id"], df["mean"], xerr=df["std"])
        ax.set_title(f"Resumo de validação cruzada por {metric}")
        ax.set_xlabel(f"{metric} médio")
        ax.set_ylabel("Experimento")
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return output_path


# Gera boxplots de features linguísticas por classe
def save_feature_distribution_charts(
    *,
    test_df: pd.DataFrame,
    settings: AppSettings,
    figures_dir: Path,
) -> list[Path]:
    if test_df.empty:
        return []
    target_column = settings.dataset.target_column
    if target_column not in test_df.columns:
        return []

    preferred = [
        "num_palavras",
        "media_palavras_por_sentenca",
        "type_token_ratio",
        "razao_palavras_longas",
        "densidade_lexical_aproximada",
    ]
    available_features = [feature for feature in preferred if feature in test_df.columns]
    if not available_features:
        available_features = [feature for feature in get_feature_columns()[:5] if feature in test_df.columns]

    paths: list[Path] = []
    figures_dir.mkdir(parents=True, exist_ok=True)
    for feature in available_features:
        output_path = figures_dir / f"feature_distribution_{feature}.png"
        plot_df = test_df[[target_column, feature]].copy()
        plot_df[feature] = pd.to_numeric(plot_df[feature], errors="coerce")
        plot_df = plot_df.dropna(subset=[feature])
        if plot_df.empty:
            continue
        labels = sorted(plot_df[target_column].astype(str).unique())
        values = [plot_df.loc[plot_df[target_column].astype(str) == label, feature].tolist() for label in labels]
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        try:
            try:
                ax.boxplot(values, tick_labels=labels)
            except TypeError:
                ax.boxplot(values, labels=labels)
            ax.set_title(f"Distribuição de {feature} por classe")
            ax.set_xlabel("Classe")
            ax.set_ylabel(feature)
            fig.tight_layout()
            fig.savefig(output_path, dpi=settings.visual_reports.dpi, bbox_inches="tight")
            paths.append(output_path)
        finally:
            plt.close(fig)
    return paths


# Gera histograma de confiança das predições quando o modelo expõe probabilidade/score
def save_prediction_confidence_chart(
    *,
    model: Any,
    x_test: pd.DataFrame,
    labels: list[str],
    output_path: Path,
    dpi: int = 160,
) -> Path | None:
    confidences = _prediction_confidences(model, x_test, labels)
    if not confidences:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    try:
        bins = min(10, max(3, len(confidences)))
        ax.hist(confidences, bins=bins)
        ax.set_title("Distribuição de confiança das predições")
        ax.set_xlabel("Confiança")
        ax.set_ylabel("Quantidade")
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return output_path


def _prediction_confidences(model: Any, x_test: pd.DataFrame, labels: list[str]) -> list[float]:
    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        try:
            matrix = predict_proba(x_test)
            return [round(float(max(row)), 6) for row in matrix]
        except Exception:
            return []

    decision_function = getattr(model, "decision_function", None)
    if not callable(decision_function):
        return []
    try:
        raw_scores = decision_function(x_test)
    except Exception:
        return []

    rows = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
    if rows and not isinstance(rows[0], list):
        rows = [[-float(value), float(value)] for value in rows]

    confidences: list[float] = []
    for row in rows:
        values = [float(value) for value in row]
        probabilities = _softmax_like(values)
        if probabilities:
            confidences.append(round(float(max(probabilities)), 6))
    return confidences


def _softmax_like(values: list[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    if total <= 0:
        return []
    return [value / total for value in exps]


def _validate_vectors(y_true: list[str], y_pred: list[str], labels: list[str]) -> None:
    if not y_true or not y_pred:
        raise VisualizationError("Vetores de avaliação vazios.")
    if len(y_true) != len(y_pred):
        raise VisualizationError("Vetores de avaliação com tamanhos incompatíveis.")
    if not labels:
        raise VisualizationError("Lista de classes vazia.")
