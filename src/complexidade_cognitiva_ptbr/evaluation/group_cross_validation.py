from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DEFAULT_GROUP_CANDIDATES = [
    "obra",
    "titulo",
    "título",
    "title",
    "work",
    "autor",
    "author",
    "fonte",
    "source",
]

DEFAULT_LABEL_ORDER = ["baixa", "media", "alta"]


@dataclass(frozen=True)
class GroupCrossValidationConfig:
    enabled: bool = True
    preferred_group_column: str = "obra"
    fallback_group_columns: list[str] = field(
        default_factory=lambda: DEFAULT_GROUP_CANDIDATES.copy()
    )
    fallback_strategy: str = "row_id"
    n_splits: int = 5
    min_groups_per_fold: int = 1
    estimator: str = "logistic_regression"
    C: float = 1.0
    max_iter: int = 1000
    class_weight: str | None = "balanced"
    max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1
    lowercase: bool = True
    fail_on_error: bool = False
    output_json: str | None = None
    output_md: str | None = None


@dataclass(frozen=True)
class DatasetColumns:
    input_path: Path
    id_column: str = "id"
    text_column: str = "texto"
    target_column: str = "target"


@dataclass(frozen=True)
class GroupColumnSelection:
    column: str
    strategy: str
    groups: pd.Series
    warnings: list[str]


# Carrega YAML de forma tolerante para permitir uso em testes e scripts avulsos
def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuração não encontrada: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


# Normaliza valores de grupo para evitar que espaços/capitalização criem grupos falsos
def normalize_group_value(value: Any) -> str:
    text = str(value).strip()
    return " ".join(text.split()).casefold()


# Valida se a coluna contém grupos utilizáveis
def usable_group_count(series: pd.Series) -> int:
    normalized = series.dropna().astype(str).map(normalize_group_value)
    normalized = normalized[normalized != ""]
    return int(normalized.nunique())


# Escolhe a coluna de agrupamento respeitando preferência e fallbacks metodológicos
def select_group_column(
    df: pd.DataFrame,
    *,
    preferred_group_column: str = "obra",
    fallback_group_columns: Iterable[str] | None = None,
    fallback_strategy: str = "row_id",
) -> GroupColumnSelection:
    warnings: list[str] = []
    candidates: list[str] = []

    if preferred_group_column:
        candidates.append(preferred_group_column)
    for column in fallback_group_columns or DEFAULT_GROUP_CANDIDATES:
        if column not in candidates:
            candidates.append(column)

    for column in candidates:
        if column not in df.columns:
            continue
        count = usable_group_count(df[column])
        if count >= 2:
            groups = df[column].astype(str).map(normalize_group_value)
            warnings.append(
                f"Agrupamento selecionado pela coluna '{column}' com {count} grupos distintos."
            )
            strategy = (
                "preferred_column"
                if column == preferred_group_column
                else "fallback_column"
            )
            return GroupColumnSelection(
                column=column, strategy=strategy, groups=groups, warnings=warnings
            )
        warnings.append(
            f"Coluna candidata '{column}' ignorada por possuir menos de 2 grupos distintos utilizáveis."
        )

    if fallback_strategy == "row_id":
        groups = pd.Series(
            [f"row_{idx}" for idx in range(len(df))], index=df.index, dtype="object"
        )
        warnings.append(
            "Nenhuma coluna de obra/autor/fonte utilizável foi encontrada. "
            "Foi aplicado fallback por identificador sintético de linha. "
            "Esse modo evita sobreposição entre folds, mas não avalia generalização por obra ou autor."
        )
        return GroupColumnSelection(
            column="__row_group__",
            strategy="synthetic_row_id",
            groups=groups,
            warnings=warnings,
        )

    raise ValueError(
        "Nenhuma coluna de agrupamento utilizável foi encontrada e o fallback por linha está desabilitado."
    )


# Ajusta o número de folds ao total real de grupos disponíveis
def effective_split_count(requested_splits: int, n_groups: int) -> int:
    if requested_splits < 2:
        raise ValueError("group_cross_validation.n_splits deve ser maior ou igual a 2.")
    if n_groups < 2:
        raise ValueError("A avaliação por grupo exige ao menos 2 grupos distintos.")
    return min(int(requested_splits), int(n_groups))


# Constrói um classificador leve para avaliação complementar, sem reutilizar folds ajustados fora da dobra
def build_estimator(config: GroupCrossValidationConfig, random_state: int) -> Pipeline:
    vectorizer = TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        lowercase=config.lowercase,
    )

    estimator_name = config.estimator.strip().lower()
    if estimator_name in {"linear_svm", "svm", "linear_svc"}:
        classifier = LinearSVC(
            C=config.C,
            class_weight=config.class_weight,
            random_state=random_state,
        )
    elif estimator_name in {"logistic_regression", "logreg", "lr"}:
        classifier = LogisticRegression(
            C=config.C,
            max_iter=config.max_iter,
            class_weight=config.class_weight,
            random_state=random_state,
            solver="lbfgs",
        )
    else:
        raise ValueError(
            "Estimador não suportado para avaliação por grupo: "
            f"{config.estimator!r}. Use 'logistic_regression' ou 'linear_svm'."
        )

    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )


# Métricas macro por dobra com zero_division=0 para datasets pequenos/smoke
def metrics_for(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true, y_pred, labels=labels, average="macro", zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
    }


# Estatísticas agregadas entre folds
def summarize_fold_metrics(
    folds: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for metric in metric_names:
        values = [float(fold["metrics"][metric]) for fold in folds]
        means[metric] = float(np.mean(values)) if values else math.nan
        stds[metric] = float(np.std(values, ddof=0)) if values else math.nan
    return means, stds


# Converte valores numpy/pandas para JSON puro
def to_builtin(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    return value


# Gera relatório Markdown objetivo e versionável
def markdown_report(result: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Avaliação complementar por agrupamento",
        "",
        "Este relatório apresenta uma avaliação complementar por grupos, projetada para reduzir o risco de que o modelo aprenda padrões específicos de uma mesma obra, autor ou fonte textual.",
        "",
        "## Resumo",
        "",
        f"- Status: `{result['status']}`",
        f"- Arquivo avaliado: `{result['input_path']}`",
        f"- Coluna de agrupamento: `{result['group_column']}`",
        f"- Estratégia de agrupamento: `{result['group_strategy']}`",
        f"- Linhas avaliadas: `{result['rows']}`",
        f"- Grupos distintos: `{result['groups']}`",
        f"- Folds solicitados: `{result['requested_n_splits']}`",
        f"- Folds efetivos: `{result['effective_n_splits']}`",
        "",
        "## Métricas agregadas",
        "",
        "| métrica | média | desvio padrão |",
        "| --- | ---: | ---: |",
    ]
    for metric, mean in result["metrics_mean"].items():
        std = result["metrics_std"].get(metric, 0.0)
        lines.append(f"| `{metric}` | {mean:.6f} | {std:.6f} |")

    lines.extend(
        [
            "",
            "## Métricas por fold",
            "",
            "| fold | treino | validação | grupos treino | grupos validação | accuracy | precision macro | recall macro | f1 macro |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for fold in result["folds"]:
        metrics = fold["metrics"]
        lines.append(
            "| {fold} | {train_rows} | {validation_rows} | {train_groups} | {validation_groups} | {accuracy:.6f} | {precision:.6f} | {recall:.6f} | {f1:.6f} |".format(
                fold=fold["fold"],
                train_rows=fold["train_rows"],
                validation_rows=fold["validation_rows"],
                train_groups=fold["train_groups"],
                validation_groups=fold["validation_groups"],
                accuracy=metrics["accuracy"],
                precision=metrics["precision_macro"],
                recall=metrics["recall_macro"],
                f1=metrics["f1_macro"],
            )
        )

    lines.extend(["", "## Matriz de confusão agregada", ""])
    labels = result["labels"]
    lines.append(
        "Linhas representam classes reais e colunas representam classes preditas."
    )
    lines.append("")
    lines.append(
        "| real \\ predito | " + " | ".join(f"`{label}`" for label in labels) + " |"
    )
    lines.append("| --- | " + " | ".join("---:" for _ in labels) + " |")
    for label, row in zip(labels, result["aggregated_confusion_matrix"]):
        lines.append(
            "| `{}` | {} |".format(label, " | ".join(str(int(v)) for v in row))
        )

    lines.extend(
        [
            "",
            "## Interpretação metodológica",
            "",
            "A avaliação tradicional estratificada mede o desempenho em partições balanceadas por classe. A avaliação por agrupamento acrescenta uma restrição mais forte: exemplos do mesmo grupo não podem aparecer simultaneamente em treino e validação dentro do mesmo fold.",
            "",
            "Quando a coluna utilizada é `obra`, a métrica indica melhor a capacidade de generalização para obras não vistas durante o treinamento daquele fold. Quando o fallback utiliza `autor`, a avaliação aproxima a generalização para autores não vistos. Quando o fallback sintético por linha é usado, o resultado deve ser interpretado apenas como execução degradada e segura, não como evidência de generalização por obra ou autor.",
            "",
            "## Alertas e observações",
            "",
        ]
    )
    warnings = result.get("warnings") or []
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- Nenhum alerta registrado.")

    return "\n".join(lines) + "\n"


# Avaliação principal por GroupKFold
def run_group_cross_validation(
    *,
    input_path: str | Path,
    output_json: str | Path,
    output_md: str | Path,
    text_column: str = "texto",
    target_column: str = "target",
    id_column: str = "id",
    preferred_group_column: str = "obra",
    fallback_group_columns: Iterable[str] | None = None,
    fallback_strategy: str = "row_id",
    n_splits: int = 5,
    estimator: str = "logistic_regression",
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: str | None = "balanced",
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    random_state: int = 42,
) -> dict[str, Any]:
    data_path = Path(input_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {data_path}")

    df = pd.read_csv(data_path)
    required_columns = {text_column, target_column}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(
            f"Colunas obrigatórias ausentes para avaliação por grupo: {missing}"
        )

    df = df.copy()
    df[text_column] = df[text_column].fillna("").astype(str)
    df[target_column] = df[target_column].fillna("").astype(str)
    df = df[
        (df[text_column].str.strip() != "") & (df[target_column].str.strip() != "")
    ].reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("A avaliação por grupo exige ao menos 2 registros válidos.")

    group_selection = select_group_column(
        df,
        preferred_group_column=preferred_group_column,
        fallback_group_columns=fallback_group_columns,
        fallback_strategy=fallback_strategy,
    )
    groups = group_selection.groups.reset_index(drop=True)
    n_groups = int(groups.nunique())
    effective_n_splits = effective_split_count(n_splits, n_groups)

    labels = [
        label
        for label in DEFAULT_LABEL_ORDER
        if label in set(df[target_column].astype(str))
    ]
    labels.extend(sorted(set(df[target_column].astype(str)) - set(labels)))
    if len(labels) < 2:
        raise ValueError("A avaliação por grupo exige ao menos 2 classes distintas.")

    config = GroupCrossValidationConfig(
        preferred_group_column=preferred_group_column,
        fallback_group_columns=list(fallback_group_columns or DEFAULT_GROUP_CANDIDATES),
        fallback_strategy=fallback_strategy,
        n_splits=n_splits,
        estimator=estimator,
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )

    splitter = GroupKFold(n_splits=effective_n_splits)
    aggregated_cm = np.zeros((len(labels), len(labels)), dtype=int)
    folds: list[dict[str, Any]] = []
    all_warnings = list(group_selection.warnings)
    if effective_n_splits < n_splits:
        all_warnings.append(
            f"Número de folds reduzido de {n_splits} para {effective_n_splits} porque há apenas {n_groups} grupos distintos."
        )

    x = df[text_column].astype(str)
    y = df[target_column].astype(str)

    for fold_number, (train_idx, validation_idx) in enumerate(
        splitter.split(x, y, groups), start=1
    ):
        train_groups = set(groups.iloc[train_idx].astype(str))
        validation_groups = set(groups.iloc[validation_idx].astype(str))
        overlap = train_groups & validation_groups
        if overlap:
            raise RuntimeError(
                f"Vazamento de grupo detectado no fold {fold_number}: {sorted(overlap)[:5]}"
            )

        model = build_estimator(config, random_state=random_state)
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        predictions = [str(value) for value in model.predict(x.iloc[validation_idx])]
        truth = [str(value) for value in y.iloc[validation_idx]]
        fold_metrics = metrics_for(truth, predictions, labels)
        fold_cm = confusion_matrix(truth, predictions, labels=labels)
        aggregated_cm += fold_cm

        folds.append(
            {
                "fold": fold_number,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(validation_idx)),
                "train_groups": int(len(train_groups)),
                "validation_groups": int(len(validation_groups)),
                "group_overlap_count": int(len(overlap)),
                "metrics": fold_metrics,
                "confusion_matrix": fold_cm.tolist(),
            }
        )

    metrics_mean, metrics_std = summarize_fold_metrics(folds)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "input_path": str(data_path),
        "rows": int(len(df)),
        "id_column": id_column,
        "text_column": text_column,
        "target_column": target_column,
        "group_column": group_selection.column,
        "group_strategy": group_selection.strategy,
        "groups": n_groups,
        "requested_n_splits": int(n_splits),
        "effective_n_splits": int(effective_n_splits),
        "estimator": estimator,
        "representation": "tfidf_word",
        "labels": labels,
        "folds": folds,
        "metrics_mean": metrics_mean,
        "metrics_std": metrics_std,
        "aggregated_confusion_matrix": aggregated_cm.tolist(),
        "warnings": all_warnings,
    }

    output_json_path = Path(output_json)
    output_md_path = Path(output_md)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(to_builtin(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    output_md_path.write_text(markdown_report(result), encoding="utf-8")
    return to_builtin(result)


# Monta execução a partir de configs/config.yaml ou configs/ci_smoke_config.yaml
def run_group_cross_validation_from_config(
    *,
    config_path: str | Path,
    input_path: str | Path | None = None,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
    preferred_group_column: str | None = None,
    n_splits: int | None = None,
) -> dict[str, Any]:
    raw_config = load_yaml_config(config_path)
    project = raw_config.get("project", {})
    dataset = raw_config.get("dataset", {})
    outputs = raw_config.get("outputs", {})
    features = raw_config.get("features", {})
    tfidf_word = features.get("tfidf", {}).get("word", {})
    group_config = raw_config.get("group_cross_validation", {}) or {}

    enabled = bool(group_config.get("enabled", True))
    if not enabled:
        metrics_dir = Path(outputs.get("metrics_dir", "outputs/metrics"))
        reports_dir = Path(outputs.get("reports_dir", "outputs/reports"))
        resolved_json = Path(
            output_json
            or group_config.get("output_json")
            or metrics_dir / "group_cross_validation_results.json"
        )
        resolved_md = Path(
            output_md
            or group_config.get("output_md")
            or reports_dir / "group_cross_validation_report.md"
        )
        result = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "disabled",
            "input_path": str(
                input_path or dataset.get("input_path", "data/raw/dataset.csv")
            ),
            "group_column": None,
            "group_strategy": None,
            "rows": 0,
            "groups": 0,
            "requested_n_splits": int(n_splits or group_config.get("n_splits", 5)),
            "effective_n_splits": 0,
            "labels": [],
            "folds": [],
            "metrics_mean": {},
            "metrics_std": {},
            "aggregated_confusion_matrix": [],
            "warnings": ["Avaliação por agrupamento desabilitada na configuração."],
        }
        resolved_json.parent.mkdir(parents=True, exist_ok=True)
        resolved_md.parent.mkdir(parents=True, exist_ok=True)
        resolved_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        resolved_md.write_text(markdown_report(result), encoding="utf-8")
        return result

    metrics_dir = Path(outputs.get("metrics_dir", "outputs/metrics"))
    reports_dir = Path(outputs.get("reports_dir", "outputs/reports"))
    resolved_input = Path(
        input_path or dataset.get("input_path", "data/raw/dataset.csv")
    )
    resolved_json = Path(
        output_json
        or group_config.get("output_json")
        or metrics_dir / "group_cross_validation_results.json"
    )
    resolved_md = Path(
        output_md
        or group_config.get("output_md")
        or reports_dir / "group_cross_validation_report.md"
    )

    ngram_raw = group_config.get("ngram_range", tfidf_word.get("ngram_range", [1, 2]))
    ngram_range = (int(ngram_raw[0]), int(ngram_raw[1]))

    return run_group_cross_validation(
        input_path=resolved_input,
        output_json=resolved_json,
        output_md=resolved_md,
        text_column=dataset.get("text_column", "texto"),
        target_column=dataset.get("target_column", "target"),
        id_column=dataset.get("id_column", "id"),
        preferred_group_column=preferred_group_column
        or group_config.get("preferred_group_column", "obra"),
        fallback_group_columns=group_config.get(
            "fallback_group_columns", DEFAULT_GROUP_CANDIDATES
        ),
        fallback_strategy=group_config.get("fallback_strategy", "row_id"),
        n_splits=int(n_splits or group_config.get("n_splits", 5)),
        estimator=group_config.get("estimator", "logistic_regression"),
        C=float(group_config.get("C", 1.0)),
        max_iter=int(group_config.get("max_iter", 1000)),
        class_weight=group_config.get("class_weight", "balanced"),
        max_features=int(
            group_config.get("max_features", tfidf_word.get("max_features", 5000))
        ),
        ngram_range=ngram_range,
        min_df=int(group_config.get("min_df", tfidf_word.get("min_df", 1))),
        random_state=int(project.get("random_state", 42)),
    )
