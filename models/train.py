from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pandas.errors import MergeError
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ..config.settings import AppSettings, TfidfVectorizerConfig
from ..data.io import read_csv_dataset, write_json
from ..features.build_features import get_feature_columns
from ..utils.paths import relative_to_root

SUPPORTED_MODELS = frozenset({"logistic_regression", "linear_svm", "random_forest"})
SUPPORTED_REPRESENTATIONS = frozenset(
    {
        "linguistic_metrics",
        "tfidf_word",
        "tfidf_char",
        "tfidf_word_plus_linguistic_metrics",
    }
)
SUPPORTED_SELECTION_METRICS = frozenset(
    {
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "balanced_accuracy",
    }
)


class TrainingError(RuntimeError):
    """Erro gerado quando a etapa de treinamento não pode ser concluída"""


# Resultado da etapa de treinamento
@dataclass(frozen=True)
class TrainingResult:
    best_model_path: Path
    best_experiment_path: Path
    experiment_results_path: Path
    best_experiment: dict[str, Any]
    experiments: list[dict[str, Any]]

    def to_dict(self, project_root: Path | None = None) -> dict[str, Any]:

        if project_root is None:
            return {
                "best_model_path": self.best_model_path.as_posix(),
                "best_experiment_path": self.best_experiment_path.as_posix(),
                "experiment_results_path": self.experiment_results_path.as_posix(),
                "best_experiment": self.best_experiment,
                "experiments": self.experiments,
            }

        return {
            "best_model_path": relative_to_root(self.best_model_path, project_root),
            "best_experiment_path": relative_to_root(self.best_experiment_path, project_root),
            "experiment_results_path": relative_to_root(self.experiment_results_path, project_root),
            "best_experiment": self.best_experiment,
            "experiments": self.experiments,
        }


# Executa treinamento, comparação em validação e persistência do melhor modelo
def train_model(settings: AppSettings) -> TrainingResult:
    
    validate_training_configuration(settings)

    train_df, val_df = load_training_frames(settings)
    feature_columns = validate_training_frames(train_df, val_df, settings)
    text_column = resolve_training_text_column(train_df, settings)

    experiments: list[dict[str, Any]] = []
    trained_models: dict[str, Pipeline] = {}

    for representation in settings.training.representations:
        for model_name in settings.training.models:
            experiment_id = f"{representation}__{model_name}"
            pipeline = build_training_pipeline(
                settings=settings,
                representation=representation,
                model_name=model_name,
                feature_columns=feature_columns,
                text_column=text_column,
            )

            try:
                metrics, predictions = fit_and_score_experiment(
                    pipeline=pipeline,
                    train_df=train_df,
                    val_df=val_df,
                    target_column=settings.dataset.target_column,
                    text_column=text_column,
                    feature_columns=feature_columns,
                )
            except TrainingError:
                raise
            except Exception as exc:
                raise TrainingError(
                    f"Falha ao treinar ou avaliar o experimento '{experiment_id}': {exc}"
                ) from exc

            experiment = {
                "experiment_id": experiment_id,
                "representation": representation,
                "model_name": model_name,
                "selection_metric": settings.training.selection_metric,
                "validation_metrics": metrics,
                "validation_prediction_distribution": _prediction_distribution(predictions),
            }
            experiments.append(experiment)
            trained_models[experiment_id] = pipeline

    if not experiments:
        raise TrainingError("Nenhum experimento foi treinado. Verifique training.models e training.representations.")

    best_experiment = select_best_experiment(experiments, settings.training.selection_metric)
    best_model = trained_models[best_experiment["experiment_id"]]

    model_dir = settings.outputs.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = model_dir / "best_model.joblib"
    best_experiment_path = model_dir / "best_experiment.json"
    experiment_results_path = model_dir / "experiment_results.csv"

    persist_model(best_model, best_model_path)
    best_experiment_payload = build_best_experiment_payload(
        settings=settings,
        best_experiment=best_experiment,
        experiments=experiments,
        feature_columns=feature_columns,
        train_df=train_df,
        val_df=val_df,
        best_model_path=best_model_path,
        experiment_results_path=experiment_results_path,
    )
    persist_best_experiment_payload(best_experiment_payload, best_experiment_path)
    persist_experiment_results(
        experiments=experiments,
        path=experiment_results_path,
        selection_metric=settings.training.selection_metric,
    )

    return TrainingResult(
        best_model_path=best_model_path,
        best_experiment_path=best_experiment_path,
        experiment_results_path=experiment_results_path,
        best_experiment=best_experiment_payload,
        experiments=experiments,
    )


# Valida nomes de modelos, representações e métrica de seleção configurados
def validate_training_configuration(settings: AppSettings) -> None:
    
    unsupported_models = sorted(set(settings.training.models).difference(SUPPORTED_MODELS))
    if unsupported_models:
        raise TrainingError(
            "Modelo(s) não suportado(s): "
            f"{', '.join(unsupported_models)}. Suportados: {', '.join(sorted(SUPPORTED_MODELS))}."
        )

    unsupported_representations = sorted(
        set(settings.training.representations).difference(SUPPORTED_REPRESENTATIONS)
    )
    if unsupported_representations:
        raise TrainingError(
            "Representação(ões) não suportada(s): "
            f"{', '.join(unsupported_representations)}. "
            f"Suportadas: {', '.join(sorted(SUPPORTED_REPRESENTATIONS))}."
        )

    if settings.training.selection_metric not in SUPPORTED_SELECTION_METRICS:
        raise TrainingError(
            f"Métrica de seleção não suportada: {settings.training.selection_metric}. "
            f"Suportadas: {', '.join(sorted(SUPPORTED_SELECTION_METRICS))}."
        )

    if "tfidf_word" in settings.training.representations and not settings.features.tfidf_word.enabled:
        raise TrainingError("A representação 'tfidf_word' está configurada, mas features.tfidf.word está desabilitado.")

    if "tfidf_char" in settings.training.representations and not settings.features.tfidf_char.enabled:
        raise TrainingError("A representação 'tfidf_char' está configurada, mas features.tfidf.char está desabilitado.")

    if (
        "tfidf_word_plus_linguistic_metrics" in settings.training.representations
        and not settings.features.tfidf_word.enabled
    ):
        raise TrainingError(
            "A representação 'tfidf_word_plus_linguistic_metrics' exige features.tfidf.word habilitado."
        )


# Carrega datasets preparados e features para treino e validação
def load_training_frames(settings: AppSettings) -> tuple[pd.DataFrame, pd.DataFrame]:

    processed_dir = settings.outputs.processed_dir
    features_dir = settings.features.output_dir

    train_prepared = read_required_csv(processed_dir / "train.csv", "dataset preparado de treino")
    val_prepared = read_required_csv(processed_dir / "val.csv", "dataset preparado de validação")
    train_features = read_required_csv(features_dir / "train_features.csv", "features de treino")
    val_features = read_required_csv(features_dir / "val_features.csv", "features de validação")

    train_df = merge_prepared_and_features(train_prepared, train_features, settings, "train")
    val_df = merge_prepared_and_features(val_prepared, val_features, settings, "val")
    return train_df, val_df


# Lê um CSV obrigatório adicionando contexto às falhas de entrada
def read_required_csv(path: Path, description: str) -> pd.DataFrame:

    if not path.exists():
        raise TrainingError(f"Arquivo obrigatório não encontrado para {description}: {path}")
    if not path.is_file():
        raise TrainingError(f"Caminho informado para {description} não é um arquivo: {path}")

    try:
        return read_csv_dataset(path)
    except Exception as exc:
        raise TrainingError(f"Não foi possível ler {description} em {path}: {exc}") from exc


# Combina texto preparado com features numéricas usando id e target
def merge_prepared_and_features(
    prepared_df: pd.DataFrame,
    features_df: pd.DataFrame,
    settings: AppSettings,
    split_name: str,
) -> pd.DataFrame:

    id_column = settings.dataset.id_column
    target_column = settings.dataset.target_column

    required_prepared = {id_column, target_column, settings.dataset.text_column}
    missing_prepared = sorted(required_prepared.difference(prepared_df.columns))
    if missing_prepared:
        raise TrainingError(
            f"O split preparado '{split_name}' não possui coluna(s): {', '.join(missing_prepared)}."
        )

    feature_columns = get_feature_columns()
    required_features = {id_column, target_column, *feature_columns}
    missing_features = sorted(required_features.difference(features_df.columns))
    if missing_features:
        raise TrainingError(
            f"O arquivo de features '{split_name}' não possui coluna(s): {', '.join(missing_features)}."
        )

    _raise_if_duplicate_keys(prepared_df, [id_column], f"split preparado '{split_name}'")
    _raise_if_duplicate_keys(features_df, [id_column], f"features '{split_name}'")

    text_columns = [id_column, target_column, settings.dataset.text_column]
    clean_column = settings.preprocessing.clean_text_column
    if clean_column in prepared_df.columns and clean_column not in text_columns:
        text_columns.append(clean_column)

    try:
        merged = prepared_df[text_columns].merge(
            features_df,
            on=[id_column, target_column],
            how="inner",
            validate="one_to_one",
        )
    except MergeError as exc:
        raise TrainingError(f"Falha ao combinar split '{split_name}': {exc}") from exc

    if len(merged) != len(prepared_df) or len(merged) != len(features_df):
        raise TrainingError(
            f"Falha ao combinar split '{split_name}': prepared={len(prepared_df)}, "
            f"features={len(features_df)}, combinado={len(merged)}. "
            "Verifique divergências de id, target ou tipos entre os arquivos."
        )

    return merged


# Valida requisitos mínimos para treinar e avaliar em validação
def validate_training_frames(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    settings: AppSettings,
) -> list[str]:

    target_column = settings.dataset.target_column
    text_column = resolve_training_text_column(train_df, settings)
    feature_columns = get_feature_columns()

    if not feature_columns:
        raise TrainingError("Nenhuma feature linguística foi configurada para treinamento.")

    for split_name, df in {"train": train_df, "val": val_df}.items():
        if df.empty:
            raise TrainingError(f"O conjunto '{split_name}' está vazio.")

        required = {target_column, text_column, *feature_columns}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise TrainingError(
                f"O conjunto '{split_name}' não possui coluna(s) obrigatória(s): {', '.join(missing)}."
            )

        _validate_target_values(df, target_column, split_name)
        _validate_text_values(df, text_column, split_name)
        _validate_numeric_feature_values(df, feature_columns, split_name)

    train_classes = set(train_df[target_column].astype(str).unique())
    val_classes = set(val_df[target_column].astype(str).unique())
    if len(train_classes) < 2:
        raise TrainingError("O conjunto de treino precisa possuir pelo menos duas classes.")

    unseen_val_classes = sorted(val_classes.difference(train_classes))
    if unseen_val_classes:
        raise TrainingError(
            "O conjunto de validação possui classe(s) inexistente(s) no treino: "
            + ", ".join(unseen_val_classes)
        )

    return feature_columns


# Monta um pipeline sklearn para a representação e modelo informados
def build_training_pipeline(
    settings: AppSettings,
    representation: str,
    model_name: str,
    feature_columns: list[str],
    text_column: str | None = None,
) -> Pipeline:

    resolved_text_column = text_column or resolve_training_text_column_name(settings)
    transformer = build_representation_transformer(
        settings=settings,
        representation=representation,
        feature_columns=feature_columns,
        text_column=resolved_text_column,
    )
    classifier = build_classifier(model_name, settings.project.random_state)

    return Pipeline(
        steps=[
            ("features", transformer),
            ("classifier", classifier),
        ]
    )


# Cria o transformador de entrada para cada representação suportada
def build_representation_transformer(
    settings: AppSettings,
    representation: str,
    feature_columns: list[str],
    text_column: str,
) -> ColumnTransformer:

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    if representation == "linguistic_metrics":
        return ColumnTransformer(
            transformers=[("linguistic_metrics", numeric_transformer, feature_columns)],
            remainder="drop",
        )

    if representation == "tfidf_word":
        return ColumnTransformer(
            transformers=[("tfidf_word", build_tfidf_vectorizer(settings.features.tfidf_word, "word"), text_column)],
            remainder="drop",
        )

    if representation == "tfidf_char":
        return ColumnTransformer(
            transformers=[("tfidf_char", build_tfidf_vectorizer(settings.features.tfidf_char, "char"), text_column)],
            remainder="drop",
        )

    if representation == "tfidf_word_plus_linguistic_metrics":
        return ColumnTransformer(
            transformers=[
                ("tfidf_word", build_tfidf_vectorizer(settings.features.tfidf_word, "word"), text_column),
                ("linguistic_metrics", numeric_transformer, feature_columns),
            ],
            remainder="drop",
        )

    raise TrainingError(f"Representação não suportada: {representation}")


# Constrói um vetorizador TF-IDF a partir da configuração central
def build_tfidf_vectorizer(config: TfidfVectorizerConfig, analyzer: str) -> TfidfVectorizer:

    if not config.enabled:
        raise TrainingError(f"A representação TF-IDF '{analyzer}' está desabilitada na configuração.")

    if analyzer == "word":
        return TfidfVectorizer(
            analyzer="word",
            lowercase=True,
            strip_accents=None,
            max_features=config.max_features,
            ngram_range=config.ngram_range,
            min_df=config.min_df,
        )

    if analyzer == "char":
        return TfidfVectorizer(
            analyzer="char",
            lowercase=True,
            max_features=config.max_features,
            ngram_range=config.ngram_range,
            min_df=config.min_df,
        )

    raise TrainingError(f"Analyzer TF-IDF não suportado: {analyzer}")


# Cria o classificador supervisionado informado
def build_classifier(model_name: str, random_state: int) -> BaseEstimator:

    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
        )

    if model_name == "linear_svm":
        return LinearSVC(
            random_state=random_state,
            class_weight="balanced",
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=40,
            max_depth=20,
            min_samples_leaf=1,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=1,
        )

    raise TrainingError(f"Modelo não suportado: {model_name}")


# Treina um experimento e calcula métricas no conjunto de validação
def fit_and_score_experiment(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_column: str,
    text_column: str | None = None,
    feature_columns: list[str] | None = None,
) -> tuple[dict[str, float], list[str]]:

    x_train = prepare_model_frame(train_df, target_column, text_column, feature_columns)
    y_train = train_df[target_column].astype(str)
    x_val = prepare_model_frame(val_df, target_column, text_column, feature_columns)
    y_val = val_df[target_column].astype(str)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            pipeline.fit(x_train, y_train)
            predictions = [str(value) for value in pipeline.predict(x_val)]
    except ValueError as exc:
        raise TrainingError(f"Falha de validação durante ajuste/predição do modelo: {exc}") from exc

    metrics = compute_classification_metrics(y_val.tolist(), predictions)
    return metrics, predictions


# Prepara uma cópia das entradas com texto e features em formatos seguros para sklearn
def prepare_model_frame(
    df: pd.DataFrame,
    target_column: str,
    text_column: str | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:

    frame = df.drop(columns=[target_column]).copy()

    if text_column and text_column in frame.columns:
        frame[text_column] = frame[text_column].astype("string").fillna("").astype(str)

    for column in feature_columns or []:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")

    return frame


# Calcula métricas de classificação usadas na seleção dos experimentos
def compute_classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:

    if not y_true:
        raise TrainingError("Não há rótulos verdadeiros para calcular métricas.")
    if len(y_true) != len(y_pred):
        raise TrainingError(
            f"Quantidade de predições incompatível com y_true: y_true={len(y_true)}, y_pred={len(y_pred)}."
        )

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
    }


# Seleciona o melhor experimento usando a métrica configurada
def select_best_experiment(
    experiments: list[dict[str, Any]],
    selection_metric: str,
) -> dict[str, Any]:

    if not experiments:
        raise TrainingError("Nenhum experimento disponível para seleção.")

    for experiment in experiments:
        metrics = experiment.get("validation_metrics")
        if not isinstance(metrics, dict):
            raise TrainingError(f"Experimento sem métricas de validação: {experiment.get('experiment_id')}.")
        if selection_metric not in metrics:
            available = ", ".join(sorted(metrics))
            raise TrainingError(
                f"Métrica de seleção '{selection_metric}' não encontrada. Disponíveis: {available}."
            )
        if not _is_finite_number(metrics[selection_metric]):
            raise TrainingError(
                f"Métrica '{selection_metric}' inválida no experimento {experiment.get('experiment_id')}: "
                f"{metrics[selection_metric]!r}."
            )

    ranked = sorted(
        enumerate(experiments),
        key=lambda indexed_item: (
            indexed_item[1]["validation_metrics"][selection_metric],
            indexed_item[1]["validation_metrics"].get("f1_weighted", 0.0),
            indexed_item[1]["validation_metrics"].get("accuracy", 0.0),
            -indexed_item[0],
        ),
        reverse=True,
    )
    return dict(ranked[0][1])


# Monta o JSON do melhor experimento e do contexto de treinamento
def build_best_experiment_payload(
    settings: AppSettings,
    best_experiment: dict[str, Any],
    experiments: list[dict[str, Any]],
    feature_columns: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    best_model_path: Path,
    experiment_results_path: Path,
) -> dict[str, Any]:

    target_column = settings.dataset.target_column

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": {
            "name": settings.project.name,
            "version": settings.project.version,
            "random_state": settings.project.random_state,
        },
        "selection": {
            "selection_metric": settings.training.selection_metric,
            "best_experiment_id": best_experiment["experiment_id"],
            "best_validation_score": best_experiment["validation_metrics"][settings.training.selection_metric],
        },
        "best_experiment": best_experiment,
        "training_context": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "train_class_distribution": _class_distribution(train_df, target_column),
            "val_class_distribution": _class_distribution(val_df, target_column),
            "text_column_used": resolve_training_text_column(train_df, settings),
            "feature_columns": feature_columns,
            "models_configured": list(settings.training.models),
            "representations_configured": list(settings.training.representations),
        },
        "outputs": {
            "best_model_path": relative_to_root(best_model_path, settings.project_root),
            "best_experiment_path": relative_to_root(
                settings.outputs.model_dir / "best_experiment.json",
                settings.project_root,
            ),
            "experiment_results_path": relative_to_root(experiment_results_path, settings.project_root),
        },
        "experiments": experiments,
    }


# Salva o pipeline sklearn completo via joblib
def persist_model(model: Pipeline, path: Path) -> Path:

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(model, path)
    except Exception as exc:
        raise TrainingError(f"Não foi possível salvar o modelo em {path}: {exc}") from exc
    return path


# Salva o JSON consolidado do melhor experimento
def persist_best_experiment_payload(payload: dict[str, Any], path: Path) -> Path:

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_json(payload, path)
    except Exception as exc:
        raise TrainingError(f"Não foi possível salvar o melhor experimento em {path}: {exc}") from exc
    return path


# Salva a tabela consolidada dos experimentos de validação
def persist_experiment_results(
    experiments: list[dict[str, Any]],
    path: Path,
    selection_metric: str = "f1_macro",
) -> Path:

    if not experiments:
        raise TrainingError("Não há experimentos para persistir.")

    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        row = {
            "experiment_id": experiment["experiment_id"],
            "representation": experiment["representation"],
            "model_name": experiment["model_name"],
            "selection_metric": experiment["selection_metric"],
        }
        for metric_name, metric_value in experiment["validation_metrics"].items():
            row[f"validation_{metric_name}"] = metric_value
        rows.append(row)

    sort_columns = [
        f"validation_{selection_metric}",
        "validation_f1_weighted",
        "validation_accuracy",
    ]
    existing_sort_columns = [column for column in sort_columns if column in rows[0]]

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pd.DataFrame(rows).sort_values(
            by=existing_sort_columns,
            ascending=False,
        ).to_csv(path, index=False, encoding="utf-8")
    except Exception as exc:
        raise TrainingError(f"Não foi possível salvar os resultados dos experimentos em {path}: {exc}") from exc

    return path


# Resolve a coluna textual usada pelos vetorizadores
def resolve_training_text_column(df: pd.DataFrame, settings: AppSettings) -> str:

    preferred = settings.preprocessing.clean_text_column
    fallback = settings.dataset.text_column

    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback

    raise TrainingError(f"Nenhuma coluna textual disponível. Esperado '{preferred}' ou '{fallback}'.")


# Retorna o nome preferencial da coluna textual para uso em chamadas sem DataFrame
def resolve_training_text_column_name(settings: AppSettings) -> str:

    return settings.preprocessing.clean_text_column


def _validate_target_values(df: pd.DataFrame, target_column: str, split_name: str) -> None:
    null_targets = int(df[target_column].isna().sum())
    if null_targets > 0:
        raise TrainingError(f"O conjunto '{split_name}' possui {null_targets} target(s) nulo(s).")

    blank_targets = int((df[target_column].astype(str).str.strip() == "").sum())
    if blank_targets > 0:
        raise TrainingError(f"O conjunto '{split_name}' possui {blank_targets} target(s) vazio(s).")


def _validate_text_values(df: pd.DataFrame, text_column: str, split_name: str) -> None:
    null_texts = int(df[text_column].isna().sum())
    if null_texts > 0:
        raise TrainingError(f"O conjunto '{split_name}' possui {null_texts} texto(s) nulo(s) em '{text_column}'.")

    blank_texts = int((df[text_column].astype(str).str.strip() == "").sum())
    if blank_texts > 0:
        raise TrainingError(f"O conjunto '{split_name}' possui {blank_texts} texto(s) vazio(s) em '{text_column}'.")


def _validate_numeric_feature_values(df: pd.DataFrame, feature_columns: list[str], split_name: str) -> None:
    invalid_columns: list[str] = []
    non_finite_columns: list[str] = []

    for column in feature_columns:
        try:
            values = pd.to_numeric(df[column], errors="raise")
        except (TypeError, ValueError):
            invalid_columns.append(column)
            continue

        null_count = int(values.isna().sum())
        if null_count > 0:
            non_finite_columns.append(f"{column} ({null_count} nulo(s))")
            continue

        finite_mask = values.map(lambda value: math.isfinite(float(value)))
        non_finite_count = int((~finite_mask).sum())
        if non_finite_count > 0:
            non_finite_columns.append(f"{column} ({non_finite_count} não finito(s))")

    if invalid_columns:
        raise TrainingError(
            f"O conjunto '{split_name}' possui feature(s) não numérica(s): {', '.join(invalid_columns)}."
        )

    if non_finite_columns:
        raise TrainingError(
            f"O conjunto '{split_name}' possui valor(es) inválido(s) nas feature(s): "
            + ", ".join(non_finite_columns)
            + "."
        )


def _raise_if_duplicate_keys(df: pd.DataFrame, key_columns: list[str], label: str) -> None:
    duplicated = int(df.duplicated(subset=key_columns, keep=False).sum())
    if duplicated > 0:
        keys = ", ".join(key_columns)
        raise TrainingError(f"{label} possui {duplicated} linha(s) com chave duplicada em: {keys}.")


def _class_distribution(df: pd.DataFrame, target_column: str) -> dict[str, int]:
    counts = df[target_column].astype(str).value_counts(dropna=False).sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _prediction_distribution(predictions: list[str]) -> dict[str, int]:
    series = pd.Series(predictions, dtype="string").value_counts(dropna=False).sort_index()
    return {str(label): int(count) for label, count in series.items()}


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, int | float):
        return False
    return math.isfinite(float(value))
