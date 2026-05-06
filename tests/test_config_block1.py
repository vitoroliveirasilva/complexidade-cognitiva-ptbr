from __future__ import annotations

from dataclasses import replace

import pytest

from complexidade_cognitiva_ptbr.config.settings import SettingsError
from complexidade_cognitiva_ptbr.config.validation import validate_settings


def test_block1_config_sections_have_safe_defaults(app_settings) -> None:
    assert app_settings.cross_validation.enabled is True
    assert app_settings.cross_validation.method == "stratified_kfold"
    assert "f1_macro" in app_settings.cross_validation.scoring
    assert app_settings.leakage_checks.fail_on_critical_leakage is True
    assert app_settings.hyperparameter_search.refit_metric == "f1_macro"
    assert app_settings.explainability.output_format == ("json", "csv", "md")
    assert app_settings.run_tracking.create_latest_pointer is True
    assert app_settings.inference.default_model_path.name == "best_model_bundle.joblib"
    assert app_settings.visual_reports.generate_normalized_confusion_matrix is True


def test_validate_settings_rejects_invalid_cv_metric(
    make_settings, app_settings
) -> None:
    cv = replace(app_settings.cross_validation, scoring=("accuracy", "metrica_fake"))
    settings = make_settings(cross_validation=cv)

    with pytest.raises(SettingsError, match="cross_validation.scoring"):
        validate_settings(settings)


def test_validate_settings_rejects_invalid_search_strategy(
    make_settings, app_settings
) -> None:
    search = replace(app_settings.hyperparameter_search, strategy="exaustiva_magica")
    settings = make_settings(hyperparameter_search=search)

    with pytest.raises(SettingsError, match="hyperparameter_search.strategy"):
        validate_settings(settings)
