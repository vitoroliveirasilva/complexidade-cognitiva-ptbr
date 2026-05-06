from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..config.settings import AppSettings
from ..data.io import read_csv_dataset, write_csv_dataset
from ..data.preprocessing import clean_text
from ..features.build_features import extract_linguistic_features
from .bundle import ModelBundle, load_model_bundle


class InferenceError(RuntimeError):
    """Erro gerado quando a inferência local não pode ser concluída"""


# Resultado serializável de uma predição local
@dataclass(frozen=True)
class PredictionResult:

    predicted_class: str
    confidence: float | None
    probabilities: dict[str, float] | None
    scores: dict[str, float] | None
    linguistic_metrics: dict[str, float | int]
    warnings: tuple[str, ...]
    model_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "scores": self.scores,
            "linguistic_metrics": self.linguistic_metrics,
            "warnings": list(self.warnings),
            "model_metadata": self.model_metadata,
        }


# Carrega o bundle configurado por padrão, priorizando outputs/latest
def load_default_bundle(settings: AppSettings, model_path: str | Path | None = None) -> ModelBundle:
    path = Path(model_path).expanduser() if model_path is not None else settings.inference.default_model_path
    if not path.is_absolute():
        path = settings.project_root / path
    return load_model_bundle(path)


# Prediz a classe de um texto único com o mesmo pré-processamento e features do treino
def predict_text(
    settings: AppSettings,
    text: str,
    *,
    model_path: str | Path | None = None,
) -> PredictionResult:
    if not settings.inference.enabled:
        raise InferenceError("A inferência local está desabilitada em inference.enabled.")
    if not isinstance(text, str) or not text.strip():
        raise InferenceError("O texto para predição não pode ser vazio.")

    bundle = load_default_bundle(settings, model_path)
    frame, linguistic_metrics = build_inference_frame(settings, bundle, text)

    try:
        predicted = str(bundle.model_pipeline.predict(frame)[0])
    except Exception as exc:
        raise InferenceError(f"Falha ao predizer o texto informado: {exc}") from exc

    probabilities = _predict_probabilities(bundle, frame) if settings.inference.include_probabilities else None
    scores = None if probabilities else _predict_scores(bundle, frame)
    confidence = _resolve_confidence(predicted, probabilities, scores)
    warnings = _build_prediction_warnings(confidence, probabilities, scores)

    return PredictionResult(
        predicted_class=predicted,
        confidence=confidence,
        probabilities=probabilities,
        scores=scores,
        linguistic_metrics=linguistic_metrics if settings.inference.include_linguistic_metrics else {},
        warnings=tuple(warnings),
        model_metadata={
            "project_name": bundle.project_name,
            "project_version": bundle.project_version,
            "created_at_utc": bundle.created_at_utc,
            "representation": bundle.representation,
            "model_name": bundle.model_name,
            "classes": list(bundle.classes),
        },
    )


# Prediz um CSV de textos e salva o resultado em outro CSV
def predict_file(
    settings: AppSettings,
    input_path: str | Path,
    output_path: str | Path,
    *,
    text_column: str | None = None,
    id_column: str | None = None,
    model_path: str | Path | None = None,
) -> Path:
    source = read_csv_dataset(input_path)
    selected_text_column = text_column or settings.dataset.text_column
    selected_id_column = id_column or (settings.dataset.id_column if settings.dataset.id_column in source.columns else None)

    if selected_text_column not in source.columns:
        raise InferenceError(f"Coluna textual não encontrada no CSV: {selected_text_column}")

    rows: list[dict[str, Any]] = []
    for index, row in source.iterrows():
        raw_text = str(row[selected_text_column])
        result = predict_text(settings, raw_text, model_path=model_path)
        output_row = {
            "prediction_index": int(index),
            "predicted_class": result.predicted_class,
            "confidence": result.confidence,
            "model_name": result.model_metadata["model_name"],
            "representation": result.model_metadata["representation"],
            "predicted_at_utc": datetime.now(timezone.utc).isoformat(),
            "text_preview": raw_text[:180],
        }
        if selected_id_column and selected_id_column in source.columns:
            output_row[selected_id_column] = row[selected_id_column]
        if result.probabilities:
            for class_name, probability in result.probabilities.items():
                output_row[f"probability_{class_name}"] = probability
        for metric_name, metric_value in result.linguistic_metrics.items():
            output_row[f"metric_{metric_name}"] = metric_value
        output_row["warnings"] = "; ".join(result.warnings)
        rows.append(output_row)

    output_df = pd.DataFrame(rows)
    return write_csv_dataset(output_df, output_path)


# Monta o DataFrame esperado pelo Pipeline persistido no bundle
def build_inference_frame(
    settings: AppSettings,
    bundle: ModelBundle,
    text: str,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    cleaned = clean_text(text, normalize_whitespace=settings.preprocessing.normalize_whitespace)
    if not cleaned:
        raise InferenceError("O texto ficou vazio após a limpeza textual.")

    linguistic_metrics = extract_linguistic_features(
        cleaned,
        long_word_min_chars=settings.features.long_word_min_chars,
    )
    row: dict[str, Any] = {
        settings.dataset.id_column: "inference_1",
        bundle.text_column: text,
        bundle.clean_text_column: cleaned,
    }
    for feature_name in bundle.feature_columns:
        row[feature_name] = linguistic_metrics.get(feature_name, 0)

    frame = pd.DataFrame([row])
    return frame, linguistic_metrics


def prediction_to_console_text(result: PredictionResult) -> str:
    lines = [
        f"Classe prevista: {result.predicted_class}",
        f"Confiança: {_format_optional_float(result.confidence)}",
    ]
    if result.probabilities:
        lines.append("Probabilidades:")
        for class_name, probability in sorted(result.probabilities.items()):
            lines.append(f"  - {class_name}: {probability:.6f}")
    elif result.scores:
        lines.append("Scores normalizados:")
        for class_name, score in sorted(result.scores.items()):
            lines.append(f"  - {class_name}: {score:.6f}")
    if result.linguistic_metrics:
        lines.append("Métricas linguísticas principais:")
        for key in ("num_palavras", "num_sentencas", "media_palavras_por_sentenca", "type_token_ratio"):
            if key in result.linguistic_metrics:
                lines.append(f"  - {key}: {result.linguistic_metrics[key]}")
    if result.warnings:
        lines.append("Avisos:")
        for warning in result.warnings:
            lines.append(f"  - {warning}")
    return "\n".join(lines)


def _predict_probabilities(bundle: ModelBundle, frame: pd.DataFrame) -> dict[str, float] | None:
    predict_proba = getattr(bundle.model_pipeline, "predict_proba", None)
    if not callable(predict_proba):
        return None
    try:
        probabilities = predict_proba(frame)[0]
    except Exception:
        return None
    return {
        class_name: round(float(probability), 6)
        for class_name, probability in zip(bundle.classes, probabilities, strict=False)
    }


def _predict_scores(bundle: ModelBundle, frame: pd.DataFrame) -> dict[str, float] | None:
    decision_function = getattr(bundle.model_pipeline, "decision_function", None)
    if not callable(decision_function):
        return None
    try:
        raw_scores = decision_function(frame)
    except Exception:
        return None
    values = raw_scores[0] if hasattr(raw_scores, "ndim") and raw_scores.ndim > 1 else raw_scores
    if len(bundle.classes) == 2 and len(values) == 1:
        values = [-float(values[0]), float(values[0])]
    numeric = [float(value) for value in values]
    shifted = _softmax_like(numeric)
    return {class_name: round(score, 6) for class_name, score in zip(bundle.classes, shifted, strict=False)}


def _softmax_like(values: list[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    if total <= 0:
        return [round(1.0 / len(values), 6)] * len(values)
    return [float(value / total) for value in exps]


def _resolve_confidence(
    predicted: str,
    probabilities: dict[str, float] | None,
    scores: dict[str, float] | None,
) -> float | None:
    source = probabilities or scores
    if not source:
        return None
    if predicted in source:
        return round(float(source[predicted]), 6)
    return round(float(max(source.values())), 6)


def _build_prediction_warnings(
    confidence: float | None,
    probabilities: dict[str, float] | None,
    scores: dict[str, float] | None,
) -> list[str]:
    warnings: list[str] = []
    if probabilities is None:
        warnings.append("O modelo não expõe probabilidades calibradas; a confiança pode ser baseada em score normalizado.")
    if scores is None and probabilities is None:
        warnings.append("O modelo não expõe probabilidades nem scores de decisão.")
    if confidence is not None and confidence < 0.55:
        warnings.append("Confiança baixa: revise a predição antes de usá-la em análise acadêmica.")
    return warnings


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/d"
    return f"{value:.6f}"
