from __future__ import annotations

from typing import TYPE_CHECKING

from .settings import SettingsError

if TYPE_CHECKING:
    from .settings import AppSettings

_ALLOWED_MODELS = frozenset({"logistic_regression", "linear_svm", "random_forest"})
_ALLOWED_REPRESENTATIONS = frozenset(
    {
        "linguistic_metrics",
        "tfidf_word",
        "tfidf_char",
        "tfidf_word_plus_linguistic_metrics",
    }
)
_ALLOWED_METRICS = frozenset(
    {
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "balanced_accuracy",
    }
)
_ALLOWED_CV_METHODS = frozenset({"stratified_kfold", "repeated_stratified_kfold"})
_ALLOWED_SEARCH_STRATEGIES = frozenset({"grid", "randomized"})
_ALLOWED_EXPLAINABILITY_FORMATS = frozenset({"json", "csv", "md"})


# Executa validações cruzadas entre seções do arquivo de configuração
def validate_settings(settings: AppSettings) -> None:

    _validate_training(settings)
    _validate_cross_validation(settings)
    _validate_leakage(settings)
    _validate_hyperparameter_search(settings)
    _validate_explainability(settings)
    _validate_run_tracking(settings)
    _validate_inference(settings)


def _validate_training(settings: AppSettings) -> None:
    unsupported_models = sorted(
        set(settings.training.models).difference(_ALLOWED_MODELS)
    )
    if unsupported_models:
        raise SettingsError(
            "training.models possui modelo(s) não suportado(s): "
            f"{', '.join(unsupported_models)}."
        )

    unsupported_representations = sorted(
        set(settings.training.representations).difference(_ALLOWED_REPRESENTATIONS)
    )
    if unsupported_representations:
        raise SettingsError(
            "training.representations possui representação(ões) não suportada(s): "
            f"{', '.join(unsupported_representations)}."
        )

    if settings.training.selection_metric not in _ALLOWED_METRICS:
        raise SettingsError(
            "training.selection_metric inválida: "
            f"{settings.training.selection_metric}."
        )

    if (
        "tfidf_word" in settings.training.representations
        and not settings.features.tfidf_word.enabled
    ):
        raise SettingsError(
            "training.representations usa 'tfidf_word', mas features.tfidf.word está desabilitado."
        )

    if (
        "tfidf_char" in settings.training.representations
        and not settings.features.tfidf_char.enabled
    ):
        raise SettingsError(
            "training.representations usa 'tfidf_char', mas features.tfidf.char está desabilitado."
        )

    if (
        "tfidf_word_plus_linguistic_metrics" in settings.training.representations
        and not settings.features.tfidf_word.enabled
    ):
        raise SettingsError(
            "training.representations usa 'tfidf_word_plus_linguistic_metrics', "
            "mas features.tfidf.word está desabilitado."
        )


def _validate_cross_validation(settings: AppSettings) -> None:
    cv = settings.cross_validation
    if cv.method not in _ALLOWED_CV_METHODS:
        raise SettingsError(
            "cross_validation.method inválido: "
            f"{cv.method}. Valores aceitos: {', '.join(sorted(_ALLOWED_CV_METHODS))}."
        )

    unsupported_scoring = sorted(set(cv.scoring).difference(_ALLOWED_METRICS))
    if unsupported_scoring:
        raise SettingsError(
            "cross_validation.scoring possui métrica(s) não suportada(s): "
            f"{', '.join(unsupported_scoring)}."
        )


def _validate_leakage(settings: AppSettings) -> None:
    threshold = settings.leakage_checks.near_duplicate_threshold
    if not 0.0 <= threshold <= 1.0:
        raise SettingsError(
            "leakage_checks.near_duplicate_threshold deve ficar entre 0.0 e 1.0."
        )


def _validate_hyperparameter_search(settings: AppSettings) -> None:
    search = settings.hyperparameter_search
    if search.strategy not in _ALLOWED_SEARCH_STRATEGIES:
        raise SettingsError(
            "hyperparameter_search.strategy inválida: "
            f"{search.strategy}. Valores aceitos: {', '.join(sorted(_ALLOWED_SEARCH_STRATEGIES))}."
        )

    if search.refit_metric not in _ALLOWED_METRICS:
        raise SettingsError(
            "hyperparameter_search.refit_metric inválida: " f"{search.refit_metric}."
        )

    unsupported_spaces = sorted(set(search.search_spaces).difference(_ALLOWED_MODELS))
    if unsupported_spaces:
        raise SettingsError(
            "hyperparameter_search.search_spaces possui modelo(s) não suportado(s): "
            f"{', '.join(unsupported_spaces)}."
        )


def _validate_explainability(settings: AppSettings) -> None:
    unsupported_formats = sorted(
        set(settings.explainability.output_format).difference(
            _ALLOWED_EXPLAINABILITY_FORMATS
        )
    )
    if unsupported_formats:
        raise SettingsError(
            "explainability.output_format possui formato(s) não suportado(s): "
            f"{', '.join(unsupported_formats)}."
        )


def _validate_run_tracking(settings: AppSettings) -> None:
    if "%" not in settings.run_tracking.run_id_format:
        raise SettingsError(
            "run_tracking.run_id_format deve usar diretivas de data/hora do strftime."
        )


def _validate_inference(settings: AppSettings) -> None:
    if not settings.inference.default_model_path.name:
        raise SettingsError(
            "inference.default_model_path deve apontar para um arquivo."
        )
