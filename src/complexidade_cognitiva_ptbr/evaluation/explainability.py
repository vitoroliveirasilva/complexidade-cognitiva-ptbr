from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from sklearn.pipeline import Pipeline

from ..config.settings import AppSettings
from ..data.io import write_csv_dataset, write_json
from ..models.train import prepare_model_frame
from ..utils.paths import relative_to_root


class ExplainabilityError(RuntimeError):
    """Erro gerado quando a explicabilidade não pode ser concluída"""


# Artefatos gerados pela explicabilidade do melhor modelo
@dataclass(frozen=True)
class ExplainabilityResult:

    top_terms_csv_path: Path
    top_terms_json_path: Path
    report_path: Path
    local_explanations_path: Path
    top_terms_figure_paths: tuple[Path, ...]
    summary: dict[str, Any]

    def to_dict(self, project_root: Path | None = None) -> dict[str, Any]:
        def fmt(path: Path) -> str:
            return (
                path.as_posix()
                if project_root is None
                else relative_to_root(path, project_root)
            )

        return {
            "top_terms_csv_path": fmt(self.top_terms_csv_path),
            "top_terms_json_path": fmt(self.top_terms_json_path),
            "report_path": fmt(self.report_path),
            "local_explanations_path": fmt(self.local_explanations_path),
            "top_terms_figure_paths": [
                fmt(path) for path in self.top_terms_figure_paths
            ],
            "summary": self.summary,
        }


@dataclass(frozen=True)
class _LinearTfidfContext:
    model: Pipeline
    transformer: Any
    classifier: Any
    classes: list[str]
    feature_names: list[str]
    tfidf_indices: list[int]
    terms_by_index: dict[int, str]
    coefficients_by_class: dict[str, list[float]]


# Gera explicabilidade global/local para modelos lineares com TF-IDF
def generate_explainability_artifacts(
    settings: AppSettings,
    *,
    model: Pipeline,
    best_experiment: dict[str, Any],
    test_df: pd.DataFrame,
    y_pred: list[str] | None = None,
    explainability_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> ExplainabilityResult:
    explainability_dir = (
        explainability_dir or settings.outputs.latest_dir / "explainability"
    )
    figures_dir = figures_dir or settings.outputs.figures_dir
    explainability_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    top_terms_csv_path = explainability_dir / "tfidf_top_terms_by_class.csv"
    top_terms_json_path = explainability_dir / "tfidf_top_terms_by_class.json"
    report_path = explainability_dir / "tfidf_explainability_report.md"
    local_explanations_path = explainability_dir / "local_explanations_sample.csv"

    if not settings.explainability.enabled:
        summary = _build_disabled_summary(
            settings, reason="explainability.enabled está desabilitado."
        )
        _persist_empty_outputs(
            top_terms_csv_path,
            top_terms_json_path,
            local_explanations_path,
            report_path,
            summary,
            settings,
        )
        return ExplainabilityResult(
            top_terms_csv_path,
            top_terms_json_path,
            report_path,
            local_explanations_path,
            (),
            summary,
        )

    context = inspect_linear_tfidf_context(model)
    if context is None:
        summary = _build_disabled_summary(
            settings,
            reason="O melhor modelo não combina TF-IDF com classificador linear explicável.",
            best_experiment=best_experiment,
        )
        _persist_empty_outputs(
            top_terms_csv_path,
            top_terms_json_path,
            local_explanations_path,
            report_path,
            summary,
            settings,
        )
        return ExplainabilityResult(
            top_terms_csv_path,
            top_terms_json_path,
            report_path,
            local_explanations_path,
            (),
            summary,
        )

    top_terms = extract_global_tfidf_terms(
        context,
        top_n=settings.explainability.top_n_terms_per_class,
    )
    top_terms_df = pd.DataFrame(top_terms)
    top_terms_csv_path = write_csv_dataset(top_terms_df, top_terms_csv_path)

    summary = build_explainability_summary(
        settings=settings,
        context=context,
        best_experiment=best_experiment,
        top_terms=top_terms,
    )
    top_terms_json_path = write_json(
        {"summary": summary, "terms": top_terms}, top_terms_json_path
    )

    figure_paths: tuple[Path, ...] = ()
    if (
        settings.visual_reports.enabled
        and settings.visual_reports.generate_top_terms_chart
    ):
        figure_paths = tuple(
            save_top_terms_figures(
                top_terms_df,
                figures_dir=figures_dir,
                dpi=settings.visual_reports.dpi,
            )
        )

    local_df = build_local_explanations_sample(
        settings=settings,
        context=context,
        test_df=test_df,
        y_pred=y_pred,
    )
    local_explanations_path = write_csv_dataset(local_df, local_explanations_path)

    report_path.write_text(
        build_explainability_markdown(
            summary, top_terms, figure_paths, local_explanations_path, settings
        ),
        encoding="utf-8",
    )
    return ExplainabilityResult(
        top_terms_csv_path=top_terms_csv_path,
        top_terms_json_path=top_terms_json_path,
        report_path=report_path,
        local_explanations_path=local_explanations_path,
        top_terms_figure_paths=figure_paths,
        summary=summary,
    )


# Extrai o contexto explicável de um Pipeline sklearn já ajustado
def inspect_linear_tfidf_context(model: Pipeline) -> _LinearTfidfContext | None:
    if not isinstance(model, Pipeline):
        return None
    if "features" not in model.named_steps or "classifier" not in model.named_steps:
        return None

    transformer = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    coef = getattr(classifier, "coef_", None)
    classes = [str(value) for value in getattr(classifier, "classes_", [])]
    if coef is None or not classes:
        return None

    try:
        raw_feature_names = transformer.get_feature_names_out()
    except Exception:
        return None

    feature_names = [str(name) for name in raw_feature_names]
    tfidf_indices: list[int] = []
    terms_by_index: dict[int, str] = {}
    for index, feature_name in enumerate(feature_names):
        if _is_tfidf_feature_name(feature_name):
            tfidf_indices.append(index)
            terms_by_index[index] = _clean_feature_term(feature_name)

    if not tfidf_indices:
        return None

    coef_matrix = _coefficient_matrix_for_classes(coef, classes)
    if coef_matrix is None:
        return None

    coefficients_by_class = {
        class_name: [float(value) for value in coef_matrix[class_index]]
        for class_index, class_name in enumerate(classes)
    }
    return _LinearTfidfContext(
        model=model,
        transformer=transformer,
        classifier=classifier,
        classes=classes,
        feature_names=feature_names,
        tfidf_indices=tfidf_indices,
        terms_by_index=terms_by_index,
        coefficients_by_class=coefficients_by_class,
    )


# Mapeia coeficientes do classificador linear para termos TF-IDF por classe
def extract_global_tfidf_terms(
    context: _LinearTfidfContext,
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    if top_n < 1:
        raise ExplainabilityError("top_n deve ser maior ou igual a 1.")

    rows: list[dict[str, Any]] = []
    for class_name in context.classes:
        coefficients = context.coefficients_by_class[class_name]
        ranked_positive = sorted(
            context.tfidf_indices,
            key=lambda index: coefficients[index],
            reverse=True,
        )[:top_n]
        ranked_negative = sorted(
            context.tfidf_indices,
            key=lambda index: coefficients[index],
        )[:top_n]

        for direction, indices in (
            ("positive", ranked_positive),
            ("negative", ranked_negative),
        ):
            for rank, feature_index in enumerate(indices, start=1):
                rows.append(
                    {
                        "class": class_name,
                        "direction": direction,
                        "rank": rank,
                        "term": context.terms_by_index[feature_index],
                        "feature_name": context.feature_names[feature_index],
                        "coefficient": round(float(coefficients[feature_index]), 8),
                    }
                )
    return rows


# Gera amostra de explicações locais usando TF-IDF * coeficiente da classe predita
def build_local_explanations_sample(
    settings: AppSettings,
    context: _LinearTfidfContext,
    test_df: pd.DataFrame,
    y_pred: list[str] | None = None,
) -> pd.DataFrame:
    if not settings.explainability.generate_local_explanations:
        return _empty_local_explanations_frame(settings)
    if test_df.empty:
        return _empty_local_explanations_frame(settings)

    target_column = settings.dataset.target_column
    id_column = settings.dataset.id_column
    feature_columns = _feature_columns_from_best_context(context)
    text_column = _resolve_text_column_for_local(settings, test_df)

    x_test = prepare_model_frame(test_df, target_column, text_column, feature_columns)
    predictions = y_pred or [str(value) for value in context.model.predict(x_test)]
    rows: list[dict[str, Any]] = []
    seen_by_class = {class_name: 0 for class_name in context.classes}
    max_per_class = settings.explainability.sample_predictions_per_class

    for row_position, predicted_class in enumerate(predictions):
        predicted_class = str(predicted_class)
        if predicted_class not in seen_by_class:
            continue
        if seen_by_class[predicted_class] >= max_per_class:
            continue

        row_frame = x_test.iloc[[row_position]].copy()
        contributions = _local_tfidf_contributions(context, row_frame, predicted_class)
        if not contributions:
            continue

        seen_by_class[predicted_class] += 1
        source_row = test_df.iloc[row_position]
        for rank, contribution in enumerate(contributions[:10], start=1):
            rows.append(
                {
                    "sample_rank_within_class": seen_by_class[predicted_class],
                    "row_position": int(row_position),
                    "id": source_row.get(id_column, ""),
                    "true_class": str(source_row.get(target_column, "")),
                    "predicted_class": predicted_class,
                    "term_rank": rank,
                    "term": contribution["term"],
                    "tfidf_value": contribution["tfidf_value"],
                    "coefficient": contribution["coefficient"],
                    "contribution": contribution["contribution"],
                }
            )

    if not rows:
        return _empty_local_explanations_frame(settings)
    return pd.DataFrame(rows)


# Salva gráficos horizontais de top termos positivos por classe
def save_top_terms_figures(
    top_terms_df: pd.DataFrame, *, figures_dir: Path, dpi: int
) -> list[Path]:
    if top_terms_df.empty:
        return []
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    positives = top_terms_df[top_terms_df["direction"] == "positive"].copy()
    for class_name, group in positives.groupby("class", dropna=False):
        safe_class = _safe_filename(str(class_name))
        output_path = figures_dir / f"top_tfidf_terms_{safe_class}.png"
        ordered = group.sort_values("rank", ascending=False)
        fig_height = max(4.0, min(14.0, 0.35 * len(ordered) + 1.5))
        fig, ax = plt.subplots(figsize=(9.0, fig_height))
        try:
            ax.barh(ordered["term"].astype(str), ordered["coefficient"].astype(float))
            ax.set_title(f"Top termos TF-IDF positivos - classe {class_name}")
            ax.set_xlabel("Coeficiente")
            ax.set_ylabel("Termo")
            fig.tight_layout()
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            paths.append(output_path)
        finally:
            plt.close(fig)
    return paths


def build_explainability_summary(
    *,
    settings: AppSettings,
    context: _LinearTfidfContext,
    best_experiment: dict[str, Any],
    top_terms: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = best_experiment.get("best_experiment", best_experiment)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "supported",
        "project": {"name": settings.project.name, "version": settings.project.version},
        "selected_experiment": {
            "experiment_id": str(selected.get("experiment_id", "")),
            "representation": str(selected.get("representation", "")),
            "model_name": str(selected.get("model_name", "")),
        },
        "classes": list(context.classes),
        "total_features": len(context.feature_names),
        "tfidf_features": len(context.tfidf_indices),
        "top_n_terms_per_class": settings.explainability.top_n_terms_per_class,
        "rows_generated": len(top_terms),
        "supported_reason": "Modelo linear com coeficientes e termos TF-IDF disponíveis.",
    }


def build_explainability_markdown(
    summary: dict[str, Any],
    top_terms: list[dict[str, Any]],
    figure_paths: tuple[Path, ...],
    local_explanations_path: Path,
    settings: AppSettings,
) -> str:
    lines = [
        "# Relatório de explicabilidade TF-IDF",
        "",
        "## Status",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Gerado em UTC: `{summary.get('generated_at_utc')}`",
    ]
    reason = summary.get("reason") or summary.get("supported_reason")
    if reason:
        lines.append(f"- Observação: {reason}")

    if summary.get("status") != "supported":
        lines.extend(
            [
                "",
                "O modelo selecionado não expõe simultaneamente vocabulário TF-IDF e coeficientes lineares. O pipeline não quebrou; apenas registrou a indisponibilidade da explicação global por coeficientes.",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "",
            "## Modelo explicado",
            "",
            f"- Experimento: `{summary['selected_experiment']['experiment_id']}`",
            f"- Representação: `{summary['selected_experiment']['representation']}`",
            f"- Modelo: `{summary['selected_experiment']['model_name']}`",
            f"- Features totais: `{summary['total_features']}`",
            f"- Features TF-IDF: `{summary['tfidf_features']}`",
            "",
            "## Top termos positivos por classe",
            "",
        ]
    )

    by_class: dict[str, list[dict[str, Any]]] = {}
    for row in top_terms:
        if row.get("direction") == "positive":
            by_class.setdefault(str(row.get("class")), []).append(row)

    for class_name, rows in by_class.items():
        lines.extend(
            [
                f"### Classe `{_markdown_inline(class_name)}`",
                "",
                "| Rank | Termo | Coeficiente |",
                "| ---: | --- | ---: |",
            ]
        )
        for row in sorted(rows, key=lambda item: int(item["rank"]))[
            : settings.explainability.top_n_terms_per_class
        ]:
            lines.append(
                f"| {row['rank']} | {_markdown_cell(str(row['term']))} | {row['coefficient']} |"
            )
        lines.append("")

    lines.extend(["## Artefatos", ""])
    lines.append(
        f"- Explicações locais: `{relative_to_root(local_explanations_path, settings.project_root)}`"
    )
    for path in figure_paths:
        lines.append(f"- Gráfico: `{relative_to_root(path, settings.project_root)}`")
    return "\n".join(lines).rstrip() + "\n"


def _build_disabled_summary(
    settings: AppSettings,
    *,
    reason: str,
    best_experiment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = (best_experiment or {}).get("best_experiment", best_experiment or {})
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "unsupported",
        "reason": reason,
        "project": {"name": settings.project.name, "version": settings.project.version},
        "selected_experiment": {
            "experiment_id": str(selected.get("experiment_id", "")),
            "representation": str(selected.get("representation", "")),
            "model_name": str(selected.get("model_name", "")),
        },
        "classes": [],
        "total_features": 0,
        "tfidf_features": 0,
        "rows_generated": 0,
    }


def _persist_empty_outputs(
    csv_path: Path,
    json_path: Path,
    local_path: Path,
    report_path: Path,
    summary: dict[str, Any],
    settings: AppSettings,
) -> None:
    write_csv_dataset(
        pd.DataFrame(
            columns=[
                "class",
                "direction",
                "rank",
                "term",
                "feature_name",
                "coefficient",
            ]
        ),
        csv_path,
    )
    write_json({"summary": summary, "terms": []}, json_path)
    write_csv_dataset(
        pd.DataFrame(
            columns=["id", "true_class", "predicted_class", "term", "contribution"]
        ),
        local_path,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_explainability_markdown(summary, [], (), local_path, settings),
        encoding="utf-8",
    )


def _coefficient_matrix_for_classes(
    coef: Any, classes: list[str]
) -> list[list[float]] | None:
    rows = getattr(coef, "tolist", lambda: coef)()
    if not isinstance(rows, list) or not rows:
        return None

    if len(classes) == 2 and len(rows) == 1:
        positive = [float(value) for value in rows[0]]
        negative = [-value for value in positive]
        return [negative, positive]

    if len(rows) != len(classes):
        return None
    return [[float(value) for value in row] for row in rows]


def _is_tfidf_feature_name(feature_name: str) -> bool:
    return feature_name.startswith("tfidf_word__") or feature_name.startswith(
        "tfidf_char__"
    )


def _clean_feature_term(feature_name: str) -> str:
    return feature_name.split("__", 1)[1] if "__" in feature_name else feature_name


def _feature_columns_from_best_context(context: _LinearTfidfContext) -> list[str]:
    columns: list[str] = []
    for name in context.feature_names:
        if name.startswith("linguistic_metrics__"):
            columns.append(name.split("__", 1)[1])
    return columns


def _resolve_text_column_for_local(settings: AppSettings, test_df: pd.DataFrame) -> str:
    if settings.preprocessing.clean_text_column in test_df.columns:
        return settings.preprocessing.clean_text_column
    return settings.dataset.text_column


def _local_tfidf_contributions(
    context: _LinearTfidfContext,
    row_frame: pd.DataFrame,
    predicted_class: str,
) -> list[dict[str, Any]]:
    coefficients = context.coefficients_by_class.get(predicted_class)
    if coefficients is None:
        return []
    try:
        transformed = context.transformer.transform(row_frame)
    except Exception:
        return []

    vector = transformed.getrow(0) if sparse.issparse(transformed) else transformed[0]
    rows: list[dict[str, Any]] = []
    if sparse.issparse(vector):
        coo = vector.tocoo()
        pairs = zip(coo.col.tolist(), coo.data.tolist(), strict=False)
    else:
        values = list(vector)
        pairs = (
            (idx, value) for idx, value in enumerate(values) if float(value) != 0.0
        )

    for feature_index, raw_value in pairs:
        if feature_index not in context.terms_by_index:
            continue
        tfidf_value = float(raw_value)
        coefficient = float(coefficients[feature_index])
        contribution = tfidf_value * coefficient
        if not math.isfinite(contribution):
            continue
        rows.append(
            {
                "term": context.terms_by_index[feature_index],
                "tfidf_value": round(tfidf_value, 8),
                "coefficient": round(coefficient, 8),
                "contribution": round(contribution, 8),
            }
        )
    return sorted(rows, key=lambda item: item["contribution"], reverse=True)


def _empty_local_explanations_frame(settings: AppSettings) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sample_rank_within_class",
            "row_position",
            settings.dataset.id_column,
            "true_class",
            "predicted_class",
            "term_rank",
            "term",
            "tfidf_value",
            "coefficient",
            "contribution",
        ]
    )


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-zÀ-ÖØ-öø-ÿ_-]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "classe"


def _markdown_inline(value: str) -> str:
    return value.replace("`", "\\`").replace("\n", " ").strip()


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()
