from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from ..config.settings import AppSettings
from ..utils.hashing import build_file_fingerprint
from ..utils.paths import relative_to_root


class ModelBundleError(RuntimeError):
    """Erro gerado em operações de persistência ou leitura do pacote de modelo"""


# Pacote autocontido para inferência local sem depender do dataset de treino
@dataclass(frozen=True)
class ModelBundle:

    model_pipeline: Pipeline
    classes: tuple[str, ...]
    representation: str
    model_name: str
    feature_columns: tuple[str, ...]
    config_snapshot: dict[str, Any]
    training_metrics: dict[str, Any]
    created_at_utc: str
    project_version: str
    dataset_fingerprint: dict[str, Any]
    text_column: str
    clean_text_column: str
    target_column: str
    project_name: str
    metadata: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "created_at_utc": self.created_at_utc,
            "project_name": self.project_name,
            "project_version": self.project_version,
            "representation": self.representation,
            "model_name": self.model_name,
            "classes": list(self.classes),
            "feature_columns": list(self.feature_columns),
            "text_column": self.text_column,
            "clean_text_column": self.clean_text_column,
            "target_column": self.target_column,
            "training_metrics": self.training_metrics,
            "dataset_fingerprint": self.dataset_fingerprint,
            "metadata": self.metadata,
        }


def create_model_bundle(
    *,
    settings: AppSettings,
    model_pipeline: Pipeline,
    best_experiment: dict[str, Any],
    feature_columns: list[str],
    config_snapshot: dict[str, Any],
    training_metrics: dict[str, Any],
    text_column: str,
) -> ModelBundle:
    classifier = (
        model_pipeline.named_steps.get("classifier")
        if hasattr(model_pipeline, "named_steps")
        else None
    )
    raw_classes = getattr(classifier, "classes_", ())
    classes = tuple(str(value) for value in raw_classes)
    if not classes:
        raise ModelBundleError(
            "O classificador treinado não expõe classes_ para o bundle."
        )

    try:
        fingerprint = build_file_fingerprint(settings.dataset.input_path).to_dict(
            settings.project_root
        )
    except Exception as exc:
        fingerprint = {"available": False, "error": str(exc)}

    return ModelBundle(
        model_pipeline=model_pipeline,
        classes=classes,
        representation=str(best_experiment.get("representation", "")),
        model_name=str(best_experiment.get("model_name", "")),
        feature_columns=tuple(feature_columns),
        config_snapshot=config_snapshot,
        training_metrics=training_metrics,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        project_version=settings.project.version,
        dataset_fingerprint=fingerprint,
        text_column=settings.dataset.text_column,
        clean_text_column=text_column,
        target_column=settings.dataset.target_column,
        project_name=settings.project.name,
        metadata={
            "supports_predict_proba": callable(
                getattr(model_pipeline, "predict_proba", None)
            ),
            "supports_decision_function": callable(
                getattr(model_pipeline, "decision_function", None)
            ),
            "inference_ready": True,
        },
    )


def save_model_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(bundle, output_path)
    except Exception as exc:
        raise ModelBundleError(
            f"Não foi possível salvar o ModelBundle em {output_path}: {exc}"
        ) from exc
    return output_path


def load_model_bundle(path: str | Path) -> ModelBundle:
    bundle_path = Path(path).expanduser().resolve()
    if not bundle_path.exists():
        raise ModelBundleError(f"ModelBundle não encontrado: {bundle_path}")
    if not bundle_path.is_file():
        raise ModelBundleError(
            f"O caminho do ModelBundle não aponta para um arquivo: {bundle_path}"
        )
    if bundle_path.stat().st_size <= 0:
        raise ModelBundleError(f"O arquivo do ModelBundle está vazio: {bundle_path}")

    try:
        loaded = joblib.load(bundle_path)
    except Exception as exc:
        raise ModelBundleError(
            f"Não foi possível carregar o ModelBundle em {bundle_path}: {exc}"
        ) from exc

    if not isinstance(loaded, ModelBundle):
        raise ModelBundleError("O artefato carregado não é um ModelBundle válido.")
    if not callable(getattr(loaded.model_pipeline, "predict", None)):
        raise ModelBundleError(
            "O ModelBundle não contém um pipeline com método predict."
        )
    return loaded


def bundle_to_dict(
    bundle: ModelBundle,
    project_root: Path | None = None,
    bundle_path: Path | None = None,
) -> dict[str, Any]:
    payload = bundle.to_metadata()
    if bundle_path is not None:
        payload["bundle_path"] = (
            relative_to_root(bundle_path, project_root)
            if project_root is not None
            else bundle_path.as_posix()
        )
    return payload
