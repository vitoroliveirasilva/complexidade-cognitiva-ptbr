from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ..config.settings import AppSettings
from ..data.io import read_csv_dataset, write_csv_dataset, write_json
from ..models.train import merge_prepared_and_features, prepare_model_frame
from ..utils.paths import relative_to_root
from .explainability import ExplainabilityResult, generate_explainability_artifacts
from .visualization import (
    save_confusion_matrix_plot,
    save_cv_summary_chart,
    save_experiment_comparison_chart,
    save_feature_distribution_charts,
    save_prediction_confidence_chart,
)


class EvaluationError(RuntimeError):
    """Erro gerado quando a avaliação final não pode ser concluída"""


# Resultado da avaliação final do modelo
@dataclass(frozen=True)
class EvaluationResult:

    final_metrics_path: Path
    classification_report_json_path: Path
    classification_report_txt_path: Path
    test_predictions_path: Path
    confusion_matrix_path: Path
    confusion_matrix_normalized_path: Path | None
    final_report_path: Path
    final_report_extended_path: Path
    visual_report_paths: tuple[Path, ...]
    explainability_result: ExplainabilityResult | None
    final_metrics: dict[str, Any]

    def to_dict(self, project_root: Path | None = None) -> dict[str, Any]:

        if project_root is None:
            return {
                "final_metrics_path": self.final_metrics_path.as_posix(),
                "classification_report_json_path": self.classification_report_json_path.as_posix(),
                "classification_report_txt_path": self.classification_report_txt_path.as_posix(),
                "test_predictions_path": self.test_predictions_path.as_posix(),
                "confusion_matrix_path": self.confusion_matrix_path.as_posix(),
                "confusion_matrix_normalized_path": self.confusion_matrix_normalized_path.as_posix() if self.confusion_matrix_normalized_path else None,
                "final_report_path": self.final_report_path.as_posix(),
                "final_report_extended_path": self.final_report_extended_path.as_posix(),
                "visual_report_paths": [path.as_posix() for path in self.visual_report_paths],
                "explainability_result": self.explainability_result.to_dict() if self.explainability_result else None,
                "final_metrics": self.final_metrics,
            }

        return {
            "final_metrics_path": relative_to_root(self.final_metrics_path, project_root),
            "classification_report_json_path": relative_to_root(
                self.classification_report_json_path,
                project_root,
            ),
            "classification_report_txt_path": relative_to_root(
                self.classification_report_txt_path,
                project_root,
            ),
            "test_predictions_path": relative_to_root(self.test_predictions_path, project_root),
            "confusion_matrix_path": relative_to_root(self.confusion_matrix_path, project_root),
            "confusion_matrix_normalized_path": relative_to_root(self.confusion_matrix_normalized_path, project_root) if self.confusion_matrix_normalized_path else None,
            "final_report_path": relative_to_root(self.final_report_path, project_root),
            "final_report_extended_path": relative_to_root(self.final_report_extended_path, project_root),
            "visual_report_paths": [relative_to_root(path, project_root) for path in self.visual_report_paths],
            "explainability_result": self.explainability_result.to_dict(project_root) if self.explainability_result else None,
            "final_metrics": self.final_metrics,
        }


# Avalia o melhor modelo no conjunto de teste e salva os artefatos finais
def evaluate_model(
    settings: AppSettings,
    *,
    model_dir: Path | None = None,
    metrics_dir: Path | None = None,
    figures_dir: Path | None = None,
    reports_dir: Path | None = None,
    explainability_dir: Path | None = None,
) -> EvaluationResult:

    model = load_best_model(settings, model_dir=model_dir)
    best_experiment = load_best_experiment(settings, model_dir=model_dir)
    test_df = load_test_frame(settings)
    validate_test_frame(test_df, settings)

    target_column = settings.dataset.target_column
    id_column = settings.dataset.id_column
    text_column = _resolve_evaluation_text_column(test_df, settings)
    feature_columns = _resolve_feature_columns_from_test(test_df)

    x_test = prepare_model_frame(test_df, target_column, text_column, feature_columns)
    y_true = _series_to_labels(test_df[target_column])
    y_pred = _predict_labels(model, x_test)
    labels = _resolve_labels(y_true, y_pred)

    final_metrics = build_final_metrics_payload(
        settings=settings,
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        test_df=test_df,
        best_experiment=best_experiment,
    )

    metrics_dir = metrics_dir or settings.outputs.metrics_dir
    figures_dir = figures_dir or settings.outputs.figures_dir
    reports_dir = reports_dir or settings.outputs.reports_dir
    explainability_dir = explainability_dir or settings.outputs.latest_dir / "explainability"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    explainability_dir.mkdir(parents=True, exist_ok=True)

    final_metrics_path = write_json(final_metrics, metrics_dir / "final_metrics.json")
    classification_report_json_path = write_json(
        build_classification_report_json(y_true, y_pred, labels),
        metrics_dir / "classification_report.json",
    )
    classification_report_txt_path = write_text_file(
        build_classification_report_text(y_true, y_pred, labels),
        metrics_dir / "classification_report.txt",
    )
    test_predictions_path = write_csv_dataset(
        build_predictions_dataframe(
            test_df=test_df,
            y_true=y_true,
            y_pred=y_pred,
            id_column=id_column,
            target_column=target_column,
            extra_text_columns=(settings.preprocessing.clean_text_column, settings.dataset.text_column),
        ),
        metrics_dir / "test_predictions.csv",
    )

    visual_paths: list[Path] = []
    confusion_matrix_path = save_confusion_matrix_plot(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        output_path=figures_dir / "confusion_matrix_absolute.png",
        normalized=False,
        dpi=settings.visual_reports.dpi,
    )
    visual_paths.append(confusion_matrix_path)
    legacy_confusion_path = save_confusion_matrix_figure(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        output_path=figures_dir / "confusion_matrix.png",
    )

    confusion_matrix_normalized_path: Path | None = None
    if settings.visual_reports.enabled and settings.visual_reports.generate_normalized_confusion_matrix:
        confusion_matrix_normalized_path = save_confusion_matrix_plot(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            output_path=figures_dir / "confusion_matrix_normalized.png",
            normalized=True,
            dpi=settings.visual_reports.dpi,
        )
        visual_paths.append(confusion_matrix_normalized_path)

    if settings.visual_reports.enabled and settings.visual_reports.generate_experiment_comparison_chart:
        experiment_chart = save_experiment_comparison_chart(
            experiment_results_path=(model_dir or settings.outputs.model_dir) / "experiment_results.csv",
            output_path=figures_dir / f"experiment_comparison_{settings.training.selection_metric}.png",
            metric=settings.training.selection_metric,
            dpi=settings.visual_reports.dpi,
        )
        if experiment_chart:
            visual_paths.append(experiment_chart)

    if settings.visual_reports.enabled and settings.visual_reports.generate_cv_summary_chart:
        cv_chart = save_cv_summary_chart(
            cv_summary_path=metrics_dir / "cv_summary.json",
            output_path=figures_dir / f"cv_summary_{settings.training.selection_metric}.png",
            metric=settings.training.selection_metric,
            dpi=settings.visual_reports.dpi,
        )
        if cv_chart:
            visual_paths.append(cv_chart)

    if settings.visual_reports.enabled and settings.visual_reports.generate_feature_distribution_charts:
        visual_paths.extend(
            save_feature_distribution_charts(
                test_df=test_df,
                settings=settings,
                figures_dir=figures_dir,
            )
        )

    if settings.visual_reports.enabled and settings.visual_reports.generate_prediction_confidence_chart:
        confidence_chart = save_prediction_confidence_chart(
            model=model,
            x_test=x_test,
            labels=labels,
            output_path=figures_dir / "prediction_confidence_distribution.png",
            dpi=settings.visual_reports.dpi,
        )
        if confidence_chart:
            visual_paths.append(confidence_chart)

    explainability_result = generate_explainability_artifacts(
        settings=settings,
        model=model,
        best_experiment=best_experiment,
        test_df=test_df,
        y_pred=y_pred,
        explainability_dir=explainability_dir,
        figures_dir=figures_dir,
    )
    visual_paths.extend(explainability_result.top_terms_figure_paths)

    output_paths = {
        "final_metrics": final_metrics_path,
        "classification_report_json": classification_report_json_path,
        "classification_report_txt": classification_report_txt_path,
        "test_predictions": test_predictions_path,
        "confusion_matrix": legacy_confusion_path,
        "confusion_matrix_absolute": confusion_matrix_path,
        "final_report_extended": reports_dir / "final_report_extended.md",
        "explainability_report": explainability_result.report_path,
    }
    if confusion_matrix_normalized_path is not None:
        output_paths["confusion_matrix_normalized"] = confusion_matrix_normalized_path

    final_report_path = write_text_file(
        build_final_report_markdown(
            settings=settings,
            final_metrics=final_metrics,
            best_experiment=best_experiment,
            labels=labels,
            output_paths=output_paths,
        ),
        reports_dir / "final_report.md",
    )
    final_report_extended_path = write_text_file(
        build_final_report_extended_markdown(
            settings=settings,
            final_metrics=final_metrics,
            best_experiment=best_experiment,
            labels=labels,
            output_paths={**output_paths, "final_report": final_report_path},
            visual_paths=tuple(dict.fromkeys(visual_paths)),
            explainability_result=explainability_result,
            metrics_dir=metrics_dir,
            reports_dir=reports_dir,
        ),
        reports_dir / "final_report_extended.md",
    )

    return EvaluationResult(
        final_metrics_path=final_metrics_path,
        classification_report_json_path=classification_report_json_path,
        classification_report_txt_path=classification_report_txt_path,
        test_predictions_path=test_predictions_path,
        confusion_matrix_path=legacy_confusion_path,
        confusion_matrix_normalized_path=confusion_matrix_normalized_path,
        final_report_path=final_report_path,
        final_report_extended_path=final_report_extended_path,
        visual_report_paths=tuple(dict.fromkeys(visual_paths)),
        explainability_result=explainability_result,
        final_metrics=final_metrics,
    )

# Carrega o melhor pipeline serializado pela etapa de treinamento
def load_best_model(settings: AppSettings, *, model_dir: Path | None = None) -> Any:

    model_path = (model_dir or settings.outputs.model_dir) / "best_model.joblib"
    if not model_path.exists():
        raise EvaluationError(
            f"Modelo treinado não encontrado: {model_path}. Execute primeiro scripts/train_model.py."
        )
    if not model_path.is_file():
        raise EvaluationError(f"O caminho do modelo não aponta para um arquivo: {model_path}")
    if model_path.stat().st_size <= 0:
        raise EvaluationError(f"O arquivo do modelo está vazio: {model_path}")

    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise EvaluationError(f"Não foi possível carregar o modelo em {model_path}: {exc}") from exc

    if not callable(getattr(model, "predict", None)):
        raise EvaluationError("O artefato carregado não possui método predict compatível com sklearn.")

    return model


# Carrega o relatório do melhor experimento selecionado em validação
def load_best_experiment(settings: AppSettings, *, model_dir: Path | None = None) -> dict[str, Any]:

    path = (model_dir or settings.outputs.model_dir) / "best_experiment.json"
    if not path.exists():
        raise EvaluationError(
            f"Arquivo do melhor experimento não encontrado: {path}. Execute primeiro scripts/train_model.py."
        )
    if not path.is_file():
        raise EvaluationError(f"O caminho do melhor experimento não aponta para um arquivo: {path}")
    if path.stat().st_size <= 0:
        raise EvaluationError(f"O arquivo do melhor experimento está vazio: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except UnicodeDecodeError as exc:
        raise EvaluationError(f"Arquivo JSON não está em UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Arquivo JSON inválido em {path}: {exc}") from exc
    except OSError as exc:
        raise EvaluationError(f"Não foi possível ler {path}: {exc}") from exc

    return _validate_best_experiment_payload(payload, path)


# Carrega e combina o conjunto de teste preparado com suas features
def load_test_frame(settings: AppSettings) -> pd.DataFrame:

    processed_dir = settings.outputs.processed_dir
    features_dir = settings.features.output_dir

    try:
        test_prepared = read_csv_dataset(processed_dir / "test.csv")
        test_features = read_csv_dataset(features_dir / "test_features.csv")
        return merge_prepared_and_features(test_prepared, test_features, settings, "test")
    except Exception as exc:
        if isinstance(exc, EvaluationError):
            raise
        raise EvaluationError(
            "Não foi possível carregar/combinar o conjunto de teste. "
            "Execute antes as etapas de preparação, features e treinamento. "
            f"Detalhe: {exc}"
        ) from exc


# Valida se o conjunto de teste possui o mínimo necessário para avaliação
def validate_test_frame(test_df: pd.DataFrame, settings: AppSettings) -> None:

    target_column = settings.dataset.target_column
    id_column = settings.dataset.id_column

    if test_df.empty:
        raise EvaluationError("O conjunto de teste está vazio.")

    missing_columns = [column for column in (id_column, target_column) if column not in test_df.columns]
    if missing_columns:
        raise EvaluationError(
            "O conjunto de teste não possui coluna(s) obrigatória(s): "
            + ", ".join(missing_columns)
            + "."
        )

    null_ids = int(test_df[id_column].isna().sum())
    if null_ids > 0:
        raise EvaluationError(f"O conjunto de teste possui {null_ids} identificador(es) nulo(s).")

    duplicate_ids = test_df[id_column].duplicated(keep=False)
    if bool(duplicate_ids.any()):
        examples = test_df.loc[duplicate_ids, id_column].astype(str).unique().tolist()[:10]
        raise EvaluationError(
            f"O conjunto de teste possui identificadores duplicados: {', '.join(examples)}."
        )

    null_targets = int(test_df[target_column].isna().sum())
    if null_targets > 0:
        raise EvaluationError(f"O conjunto de teste possui {null_targets} target(s) nulo(s).")

    blank_targets = int(test_df[target_column].map(_is_blank_value).sum())
    if blank_targets > 0:
        raise EvaluationError(f"O conjunto de teste possui {blank_targets} target(s) vazio(s).")

    unique_classes = sorted(test_df[target_column].astype(str).unique())
    if len(unique_classes) < 1:
        raise EvaluationError("O conjunto de teste não possui classes válidas.")


# Monta o JSON principal de métricas finais
def build_final_metrics_payload(
    settings: AppSettings,
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    test_df: pd.DataFrame,
    best_experiment: dict[str, Any],
) -> dict[str, Any]:

    _validate_prediction_vectors(y_true, y_pred)
    selected_experiment = _extract_selected_experiment(best_experiment)
    target_column = settings.dataset.target_column
    metrics = compute_final_metrics(y_true, y_pred)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": {
            "name": settings.project.name,
            "version": settings.project.version,
            "random_state": settings.project.random_state,
        },
        "evaluation_dataset": {
            "split": "test",
            "rows": int(len(test_df)),
            "class_distribution": class_distribution(test_df, target_column),
            "labels": labels,
        },
        "selected_experiment": selected_experiment,
        "test_metrics": metrics,
        "prediction_distribution": prediction_distribution(y_pred),
    }


# Calcula métricas finais de classificação multiclasse
def compute_final_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:

    _validate_prediction_vectors(y_true, y_pred)

    return {
        "accuracy": _round_metric(accuracy_score(y_true, y_pred)),
        "precision_macro": _round_metric(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": _round_metric(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": _round_metric(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": _round_metric(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": _round_metric(balanced_accuracy_score(y_true, y_pred)),
    }


# Gera o classification report em formato JSON
def build_classification_report_json(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict[str, Any]:

    _validate_prediction_vectors(y_true, y_pred)
    if not labels:
        raise EvaluationError("A lista de rótulos para o relatório de classificação está vazia.")

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return _round_nested_floats(report)


# Gera o classification report textual
def build_classification_report_text(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> str:

    _validate_prediction_vectors(y_true, y_pred)
    if not labels:
        raise EvaluationError("A lista de rótulos para o relatório de classificação está vazia.")

    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )


# Monta CSV com predições do conjunto de teste
def build_predictions_dataframe(
    test_df: pd.DataFrame,
    y_true: list[str],
    y_pred: list[str],
    id_column: str,
    target_column: str,
    extra_text_columns: Sequence[str] | None = None,
) -> pd.DataFrame:

    _validate_prediction_vectors(y_true, y_pred)

    missing_columns = [column for column in (id_column, target_column) if column not in test_df.columns]
    if missing_columns:
        raise EvaluationError(
            "Não foi possível montar predições. Coluna(s) ausente(s): "
            + ", ".join(missing_columns)
            + "."
        )

    if len(test_df) != len(y_true):
        raise EvaluationError(
            "Quantidade de registros do teste difere da quantidade de predições: "
            f"test_df={len(test_df)}, y_true={len(y_true)}, y_pred={len(y_pred)}."
        )

    output = pd.DataFrame(
        {
            id_column: test_df[id_column].tolist(),
            target_column: y_true,
            "predicted_target": y_pred,
            "correct": [true == pred for true, pred in zip(y_true, y_pred, strict=True)],
        }
    )

    candidate_text_columns = tuple(extra_text_columns or ()) + ("texto_limpo", "texto")
    for column in dict.fromkeys(candidate_text_columns):
        if column in test_df.columns and column not in output.columns:
            output[column] = _series_to_safe_strings(test_df[column])

    return output


# Salva a matriz de confusão como imagem PNG
def save_confusion_matrix_figure(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    output_path: Path,
) -> Path:

    _validate_prediction_vectors(y_true, y_pred)
    if not labels:
        raise EvaluationError("A lista de rótulos para a matriz de confusão está vazia.")

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure_width = max(6.0, min(18.0, len(labels) * 1.4))
    figure_height = max(5.0, min(16.0, len(labels) * 1.2))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    try:
        image = ax.imshow(matrix)
        ax.set_title("Matriz de confusão - conjunto de teste")
        ax.set_xlabel("Classe predita")
        ax.set_ylabel("Classe real")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                ax.text(
                    column_index,
                    row_index,
                    str(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                )

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    except OSError as exc:
        raise EvaluationError(f"Não foi possível salvar a matriz de confusão em {output_path}: {exc}") from exc
    finally:
        plt.close(fig)

    return output_path


# Gera relatório final em Markdown para uso acadêmico
def build_final_report_markdown(
    settings: AppSettings,
    final_metrics: dict[str, Any],
    best_experiment: dict[str, Any],
    labels: list[str],
    output_paths: dict[str, Path],
) -> str:

    selected = _required_mapping(final_metrics, "selected_experiment")
    metrics = _required_mapping(final_metrics, "test_metrics")
    validation_metrics = _required_mapping(selected, "validation_metrics")
    dataset_info = _required_mapping(final_metrics, "evaluation_dataset")
    class_distribution_payload = _required_mapping(dataset_info, "class_distribution")

    lines = [
        "# Relatório final de avaliação",
        "",
        "## Identificação",
        "",
        f"- Projeto: `{_markdown_inline(settings.project.name)}`",
        f"- Versão: `{_markdown_inline(settings.project.version)}`",
        f"- Gerado em UTC: `{_markdown_inline(str(final_metrics['generated_at_utc']))}`",
        f"- Seed configurada: `{settings.project.random_state}`",
        "",
        "## Experimento selecionado",
        "",
        f"- Experimento: `{_markdown_inline(str(selected['experiment_id']))}`",
        f"- Representação: `{_markdown_inline(str(selected['representation']))}`",
        f"- Modelo: `{_markdown_inline(str(selected['model_name']))}`",
        f"- Métrica de seleção: `{_markdown_inline(str(selected['selection_metric']))}`",
        f"- Pontuação em validação: `{selected['best_validation_score']}`",
        "",
        "### Métricas em validação",
        "",
        "| Métrica | Valor |",
        "| --- | ---: |",
    ]

    for metric_name, metric_value in validation_metrics.items():
        lines.append(f"| {_markdown_cell(str(metric_name))} | {metric_value} |")

    labels_text = ", ".join(_markdown_inline(str(label)) for label in labels)
    lines.extend(
        [
            "",
            "## Avaliação no conjunto de teste",
            "",
            f"- Total de registros avaliados: `{dataset_info['rows']}`",
            f"- Classes avaliadas: `{labels_text}`",
            "",
            "### Distribuição de classes no teste",
            "",
            "| Classe | Quantidade |",
            "| --- | ---: |",
        ]
    )

    for label, count in class_distribution_payload.items():
        lines.append(f"| {_markdown_cell(str(label))} | {count} |")

    lines.extend(
        [
            "",
            "### Métricas finais",
            "",
            "| Métrica | Valor |",
            "| --- | ---: |",
        ]
    )

    for metric_name, metric_value in metrics.items():
        lines.append(f"| {_markdown_cell(str(metric_name))} | {metric_value} |")

    lines.extend(["", "## Artefatos gerados", ""])

    for name, path in output_paths.items():
        lines.append(f"- `{_markdown_inline(name)}`: `{relative_to_root(path, settings.project_root)}`")

    lines.extend(
        [
            "",
            "## Observações para discussão acadêmica",
            "",
            "A avaliação final utiliza o melhor experimento escolhido no conjunto de validação e aplica esse modelo ao conjunto de teste, preservando a separação entre treino, validação e teste.",
            "",
            "As métricas macro ajudam a observar o comportamento médio entre classes, enquanto o F1 ponderado considera a distribuição real de exemplos por classe.",
            "",
            "A matriz de confusão permite analisar visualmente quais classes foram mais confundidas pelo modelo, apoiando a discussão dos resultados no TCC.",
            "",
            "As métricas linguísticas interpretáveis permanecem disponíveis nos arquivos de features e podem ser usadas para explicar parte do comportamento dos modelos treinados.",
            "",
            "## Referência do relatório de treino",
            "",
            f"- Arquivo do melhor experimento: `{relative_to_root(settings.outputs.model_dir / 'best_experiment.json', settings.project_root)}`",
        ]
    )

    training_context = best_experiment.get("training_context")
    if isinstance(training_context, Mapping):
        feature_columns = training_context.get("feature_columns", [])
        feature_count = len(feature_columns) if isinstance(feature_columns, Sequence) else 0
        lines.append(f"- Quantidade de features linguísticas consideradas no treino: `{feature_count}`")

    return "\n".join(lines) + "\n"

# Gera relatório consolidado com evidências de CV, leakage, hiperparâmetros, explicabilidade e gráficos
def build_final_report_extended_markdown(
    settings: AppSettings,
    final_metrics: dict[str, Any],
    best_experiment: dict[str, Any],
    labels: list[str],
    output_paths: dict[str, Path],
    visual_paths: tuple[Path, ...],
    explainability_result: ExplainabilityResult | None,
    metrics_dir: Path,
    reports_dir: Path,
) -> str:
    selected = _required_mapping(final_metrics, "selected_experiment")
    metrics = _required_mapping(final_metrics, "test_metrics")
    dataset_info = _required_mapping(final_metrics, "evaluation_dataset")
    run_dir = reports_dir.parent if reports_dir.name == "reports" else reports_dir

    leakage_payload = _read_optional_json(metrics_dir / "leakage_report.json")
    cv_summary = _read_optional_json(metrics_dir / "cv_summary.json")
    best_hyperparameters = _read_optional_json(metrics_dir / "best_hyperparameters.json")
    data_fingerprint = _read_optional_json(run_dir / "data_fingerprint.json")

    lines = [
        "# Relatório final estendido",
        "",
        "## Identificação da execução",
        "",
        f"- Projeto: `{_markdown_inline(settings.project.name)}`",
        f"- Versão: `{_markdown_inline(settings.project.version)}`",
        f"- Gerado em UTC: `{_markdown_inline(str(final_metrics['generated_at_utc']))}`",
        f"- Seed configurada: `{settings.project.random_state}`",
        f"- Classes avaliadas: `{_markdown_inline(', '.join(labels))}`",
        "",
        "## Dataset e rastreabilidade",
        "",
        f"- Split avaliado: `{dataset_info.get('split')}`",
        f"- Registros no teste: `{dataset_info.get('rows')}`",
    ]

    dataset_hash = None
    if isinstance(data_fingerprint, dict):
        dataset = data_fingerprint.get("dataset")
        if isinstance(dataset, dict):
            dataset_hash = dataset.get("sha256") or dataset.get("hash_sha256")
            dataset_path = dataset.get("path")
            if dataset_path:
                lines.append(f"- Dataset bruto: `{_markdown_inline(str(dataset_path))}`")
            if dataset_hash:
                lines.append(f"- SHA-256 do dataset bruto: `{_markdown_inline(str(dataset_hash))}`")
    if not dataset_hash:
        lines.append("- Fingerprint do dataset: não localizado nesta execução.")

    lines.extend(
        [
            "",
            "## Melhor experimento",
            "",
            f"- Experimento: `{_markdown_inline(str(selected['experiment_id']))}`",
            f"- Representação: `{_markdown_inline(str(selected['representation']))}`",
            f"- Modelo: `{_markdown_inline(str(selected['model_name']))}`",
            f"- Métrica de seleção: `{_markdown_inline(str(selected['selection_metric']))}`",
            f"- Pontuação de validação: `{selected['best_validation_score']}`",
            "",
            "## Métricas finais no teste",
            "",
            "| Métrica | Valor |",
            "| --- | ---: |",
        ]
    )
    for metric_name, metric_value in metrics.items():
        lines.append(f"| {_markdown_cell(str(metric_name))} | {metric_value} |")

    lines.extend(["", "## Validação cruzada", ""])
    if isinstance(cv_summary, dict) and cv_summary.get("summary"):
        summary = cv_summary["summary"]
        lines.append(f"- Experimentos avaliados em CV: `{summary.get('total_experiments')}`")
        lines.append(f"- Melhor experimento em CV: `{summary.get('best_experiment_id')}`")
        lines.append(f"- Média da métrica de seleção: `{summary.get('best_selection_metric_mean')}`")
    else:
        lines.append("- Resumo de validação cruzada não localizado ou etapa desabilitada.")

    lines.extend(["", "## Vazamento de dados", ""])
    if isinstance(leakage_payload, dict):
        summary = leakage_payload.get("summary", {})
        critical = summary.get("critical_findings", leakage_payload.get("critical_findings", 0)) if isinstance(summary, dict) else 0
        warnings = summary.get("warning_findings", leakage_payload.get("warning_findings", 0)) if isinstance(summary, dict) else 0
        status = leakage_payload.get("status") or summary.get("status", "registrado") if isinstance(summary, dict) else "registrado"
        lines.append(f"- Status: `{status}`")
        lines.append(f"- Achados críticos: `{critical}`")
        lines.append(f"- Alertas: `{warnings}`")
    else:
        lines.append("- Relatório de vazamento não localizado nesta execução.")

    lines.extend(["", "## Hiperparâmetros", ""])
    if isinstance(best_hyperparameters, dict):
        best = best_hyperparameters.get("best_experiment", {})
        params = best.get("best_params", {}) if isinstance(best, dict) else {}
        lines.append(f"- Estratégia: `{best_hyperparameters.get('strategy', 'n/a')}`")
        if params:
            lines.extend(["", "| Parâmetro | Valor |", "| --- | --- |"])
            for name, value in params.items():
                lines.append(f"| {_markdown_cell(str(name))} | `{_markdown_inline(str(value))}` |")
        else:
            lines.append("- Busca sem parâmetros selecionados ou etapa desabilitada.")
    else:
        lines.append("- Arquivo de melhores hiperparâmetros não localizado.")

    lines.extend(["", "## Explicabilidade", ""])
    if explainability_result is not None:
        summary = explainability_result.summary
        lines.append(f"- Status: `{summary.get('status')}`")
        lines.append(f"- Features TF-IDF explicáveis: `{summary.get('tfidf_features', 0)}`")
        lines.append(f"- Relatório: `{relative_to_root(explainability_result.report_path, settings.project_root)}`")
        if summary.get("reason"):
            lines.append(f"- Observação: {summary.get('reason')}")
    else:
        lines.append("- Explicabilidade não executada.")

    lines.extend(["", "## Gráficos e artefatos visuais", ""])
    if visual_paths:
        for path in visual_paths:
            lines.append(f"- `{relative_to_root(path, settings.project_root)}`")
    else:
        lines.append("- Nenhum gráfico adicional foi gerado nesta execução.")

    lines.extend(["", "## Arquivos principais", ""])
    for name, path in output_paths.items():
        lines.append(f"- `{_markdown_inline(name)}`: `{relative_to_root(path, settings.project_root)}`")

    lines.extend(
        [
            "",
            "## Observações metodológicas automáticas",
            "",
            "O conjunto de teste é usado apenas na avaliação final, depois da seleção do melhor experimento por validação.",
            "",
            "Os artefatos de validação cruzada, vazamento, hiperparâmetros e explicabilidade complementam as métricas finais, reduzindo o risco de interpretar um resultado alto sem evidências experimentais.",
            "",
            "Quando o modelo selecionado não for linear com TF-IDF, a etapa de explicabilidade registra a limitação em vez de interromper a execução.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _resolve_evaluation_text_column(test_df: pd.DataFrame, settings: AppSettings) -> str:
    if settings.preprocessing.clean_text_column in test_df.columns:
        return settings.preprocessing.clean_text_column
    return settings.dataset.text_column


def _resolve_feature_columns_from_test(test_df: pd.DataFrame) -> list[str]:
    return [column for column in _known_feature_columns() if column in test_df.columns]


def _known_feature_columns() -> list[str]:
    try:
        from ..features.build_features import get_feature_columns

        return get_feature_columns()
    except Exception:
        return []


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


# Salva arquivo textual em UTF-8 de forma atômica
def write_text_file(content: str, path: Path) -> Path:

    if not isinstance(content, str):
        raise EvaluationError("O conteúdo textual para escrita deve ser uma string.")

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            delete=False,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)
        temp_path.replace(output_path)
    except OSError as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise EvaluationError(f"Não foi possível salvar o arquivo textual em {output_path}: {exc}") from exc

    return output_path


# Retorna distribuição de classes ordenada por rótulo
def class_distribution(df: pd.DataFrame, target_column: str) -> dict[str, int]:

    if target_column not in df.columns:
        raise EvaluationError(f"Coluna target ausente para distribuição de classes: {target_column}")

    counts = df[target_column].astype(str).value_counts(dropna=False).sort_index()
    return {str(label): int(count) for label, count in counts.items()}


# Retorna distribuição das classes preditas
def prediction_distribution(predictions: list[str]) -> dict[str, int]:

    if not predictions:
        return {}

    series = pd.Series(predictions, dtype="string").value_counts(dropna=False).sort_index()
    return {str(label): int(count) for label, count in series.items()}


def _predict_labels(model: Any, x_test: pd.DataFrame) -> list[str]:
    try:
        predictions = model.predict(x_test)
    except Exception as exc:
        raise EvaluationError(f"Não foi possível gerar predições no conjunto de teste: {exc}") from exc

    return [str(value) for value in predictions]


def _series_to_labels(series: pd.Series) -> list[str]:
    return [str(value) for value in series.tolist()]


def _series_to_safe_strings(series: pd.Series) -> list[str]:
    return ["" if pd.isna(value) else str(value) for value in series.tolist()]


def _resolve_labels(y_true: list[str], y_pred: list[str]) -> list[str]:
    _validate_prediction_vectors(y_true, y_pred)
    return sorted(set(y_true).union(y_pred))


def _validate_prediction_vectors(y_true: Sequence[str], y_pred: Sequence[str]) -> None:
    if not y_true:
        raise EvaluationError("A avaliação não possui rótulos reais.")
    if not y_pred:
        raise EvaluationError("A avaliação não possui predições.")
    if len(y_true) != len(y_pred):
        raise EvaluationError(
            "Quantidade de rótulos reais e predições difere: "
            f"y_true={len(y_true)}, y_pred={len(y_pred)}."
        )


def _validate_best_experiment_payload(payload: Any, path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise EvaluationError(f"{path.name} deve conter um objeto JSON na raiz.")

    best_experiment = _required_mapping(payload, "best_experiment")
    selection = _required_mapping(payload, "selection")
    validation_metrics = _required_mapping(best_experiment, "validation_metrics")

    required_best_keys = ("experiment_id", "representation", "model_name")
    missing_best = [key for key in required_best_keys if not _non_empty_value(best_experiment.get(key))]
    if missing_best:
        raise EvaluationError(
            f"best_experiment.json possui best_experiment incompleto. Campo(s): {', '.join(missing_best)}."
        )

    required_selection_keys = ("selection_metric", "best_validation_score")
    missing_selection = [key for key in required_selection_keys if key not in selection]
    if missing_selection:
        raise EvaluationError(
            f"best_experiment.json possui selection incompleto. Campo(s): {', '.join(missing_selection)}."
        )

    selection_metric = str(selection["selection_metric"])
    if selection_metric not in validation_metrics:
        available = ", ".join(sorted(str(key) for key in validation_metrics))
        raise EvaluationError(
            f"Métrica de seleção '{selection_metric}' não está presente em validation_metrics. "
            f"Disponíveis: {available}."
        )

    return payload


def _extract_selected_experiment(best_experiment_payload: dict[str, Any]) -> dict[str, Any]:
    best_experiment = _required_mapping(best_experiment_payload, "best_experiment")
    selection = _required_mapping(best_experiment_payload, "selection")
    validation_metrics = _required_mapping(best_experiment, "validation_metrics")

    return {
        "experiment_id": str(best_experiment["experiment_id"]),
        "representation": str(best_experiment["representation"]),
        "model_name": str(best_experiment["model_name"]),
        "selection_metric": str(selection["selection_metric"]),
        "best_validation_score": selection["best_validation_score"],
        "validation_metrics": dict(validation_metrics),
    }


def _required_mapping(mapping: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise EvaluationError(f"Estrutura obrigatória ausente ou inválida: {key}")
    return value


def _non_empty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _is_blank_value(value: Any) -> bool:
    return isinstance(value, str) and value.strip() == ""


def _round_metric(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    return round(number, 6)


def _round_nested_floats(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _round_nested_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_nested_floats(item) for item in value]
    if isinstance(value, float):
        return _round_metric(value)
    return value


def _markdown_inline(value: str) -> str:
    return value.replace("`", "\\`").replace("\n", " ").strip()


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()
