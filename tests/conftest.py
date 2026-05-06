from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest


def _find_project_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        package_dir = candidate / "src" / "complexidade_cognitiva_ptbr"
        if package_dir.exists():
            return candidate
    return start.resolve().parent


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_MPLCONFIGDIR = PROJECT_ROOT / ".pytest_cache" / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

from complexidade_cognitiva_ptbr.config.settings import (
    AppSettings,
    CrossValidationConfig,
    DatasetConfig,
    ExplainabilityConfig,
    FeaturesConfig,
    HyperparameterSearchConfig,
    InferenceConfig,
    LeakageChecksConfig,
    LoggingConfig,
    OutputsConfig,
    PreprocessingConfig,
    ProjectConfig,
    RunTrackingConfig,
    SplitConfig,
    TfidfVectorizerConfig,
    TrainingConfig,
    VisualReportsConfig,
)
from complexidade_cognitiva_ptbr.data.preprocessing import prepare_dataset
from complexidade_cognitiva_ptbr.features.build_features import (
    build_feature_datasets,
)
from complexidade_cognitiva_ptbr.models.train import train_model


# Retorna um corpus sintético pequeno, balanceado e determinístico
@pytest.fixture()
def sample_dataset() -> pd.DataFrame:

    rows = [
        (1, "O menino viu a lua e sorriu.", "baixa"),
        (2, "A casa era pequena, mas feliz.", "baixa"),
        (3, "No quintal, a flor abriu cedo.", "baixa"),
        (4, "Ela cantou quando o sol nasceu.", "baixa"),
        (5, "O gato pulou sobre o muro.", "baixa"),
        (6, "A chuva caiu leve na tarde.", "baixa"),
        (7, "A menina leu o bilhete e guardou segredo.", "baixa"),
        (8, "O cachorro dormiu perto da porta.", "baixa"),
        (9, "O rio brilhava no fim da rua.", "baixa"),
        (10, "O pássaro voltou para a árvore.", "baixa"),
        (
            11,
            "O viajante caminhou pela estrada enquanto recordava antigas promessas.",
            "media",
        ),
        (
            12,
            "A narrativa alternava memórias e desejos, revelando conflitos silenciosos.",
            "media",
        ),
        (
            13,
            "Embora cansada, a personagem decidiu enfrentar a longa travessia.",
            "media",
        ),
        (14, "O narrador descreveu a vila com detalhes de tempo e costumes.", "media"),
        (15, "As escolhas do protagonista mudaram o ritmo da história.", "media"),
        (16, "Quando a noite chegou, todos compreenderam o valor da espera.", "media"),
        (17, "A carta explicava motivos antigos e uma decisão difícil.", "media"),
        (18, "O diálogo revelou dúvidas, afetos e pequenas contradições.", "media"),
        (19, "A família esperou em silêncio pela notícia.", "media"),
        (20, "A cena final retomou conflitos apresentados no início.", "media"),
        (
            21,
            "A tessitura simbólica do conto desloca a percepção temporal do leitor.",
            "alta",
        ),
        (
            22,
            "Conquanto a voz narrativa pareça simples, sua ambiguidade semântica intensifica a interpretação.",
            "alta",
        ),
        (
            23,
            "O encadeamento sintático prolongado exige inferências sobre memória, culpa e identidade.",
            "alta",
        ),
        (
            24,
            "As imagens metafóricas articulam tensões históricas e subjetivas em planos simultâneos.",
            "alta",
        ),
        (
            25,
            "A fragmentação discursiva impõe ao leitor reconstruir sentidos elípticos e contraditórios.",
            "alta",
        ),
        (
            26,
            "A complexidade lexical acompanha uma estrutura narrativa densamente introspectiva.",
            "alta",
        ),
        (
            27,
            "A composição intertextual convoca repertórios culturais diversos para ampliar o sentido.",
            "alta",
        ),
        (
            28,
            "A instabilidade do foco narrativo produz camadas interpretativas sobrepostas.",
            "alta",
        ),
        (
            29,
            "A alegoria estrutura uma crítica social mediada por símbolos recorrentes.",
            "alta",
        ),
        (
            30,
            "A construção polifônica tensiona vozes sociais e perspectivas narrativas divergentes.",
            "alta",
        ),
    ]
    return pd.DataFrame(rows, columns=["id", "texto", "target"])


# Cria configurações isoladas por teste usando diretórios temporários
@pytest.fixture()
def app_settings(tmp_path: Path) -> AppSettings:

    project_root = tmp_path
    return AppSettings(
        config_path=project_root / "configs" / "config.yaml",
        project_root=project_root,
        project=ProjectConfig(
            name="complexidade-cognitiva-ptbr-test",
            version="0.1.0",
            description="Configuração de teste do pipeline.",
            language="pt-BR",
            random_state=42,
        ),
        dataset=DatasetConfig(
            input_path=project_root / "data" / "raw" / "dataset.csv",
            id_column="id",
            text_column="texto",
            target_column="target",
            split=SplitConfig(
                train_size=0.70, val_size=0.15, test_size=0.15, stratify=True
            ),
        ),
        preprocessing=PreprocessingConfig(
            preserve_original_text=True,
            clean_text_column="texto_limpo",
            normalize_whitespace=True,
        ),
        features=FeaturesConfig(
            output_dir=project_root / "data" / "processed" / "features",
            long_word_min_chars=7,
            tfidf_word=TfidfVectorizerConfig(
                enabled=True,
                max_features=200,
                ngram_range=(1, 2),
                min_df=1,
            ),
            tfidf_char=TfidfVectorizerConfig(
                enabled=True,
                max_features=200,
                ngram_range=(3, 5),
                min_df=1,
            ),
        ),
        training=TrainingConfig(
            models=("logistic_regression",),
            representations=("linguistic_metrics",),
            selection_metric="f1_macro",
        ),
        cross_validation=CrossValidationConfig(
            enabled=True,
            method="stratified_kfold",
            n_splits=3,
            n_repeats=1,
            shuffle=True,
            random_state=42,
            scoring=("accuracy", "f1_macro"),
        ),
        leakage_checks=LeakageChecksConfig(
            enabled=True,
            check_id_overlap=True,
            check_exact_text_overlap=True,
            check_normalized_text_overlap=True,
            check_near_duplicates=True,
            near_duplicate_threshold=0.92,
            check_target_like_columns=True,
            fail_on_critical_leakage=True,
        ),
        hyperparameter_search=HyperparameterSearchConfig(
            enabled=True,
            strategy="grid",
            refit_metric="f1_macro",
            n_jobs=1,
            verbose=0,
            save_all_results=True,
            search_spaces={
                "logistic_regression": {
                    "C": (1.0,),
                    "max_iter": (1000,),
                    "class_weight": (None,),
                }
            },
        ),
        explainability=ExplainabilityConfig(
            enabled=True,
            top_n_terms_per_class=10,
            generate_global_tfidf_coefficients=True,
            generate_local_explanations=True,
            sample_predictions_per_class=2,
            output_format=("json", "csv", "md"),
        ),
        run_tracking=RunTrackingConfig(
            enabled=True,
            run_id_format="%Y%m%d_%H%M%S",
            create_latest_pointer=True,
            copy_config_snapshot=True,
            save_environment=True,
            save_dataset_fingerprint=True,
            save_git_commit=True,
        ),
        inference=InferenceConfig(
            enabled=True,
            default_model_path=project_root
            / "outputs"
            / "latest"
            / "models"
            / "best_model_bundle.joblib",
            include_probabilities=True,
            include_linguistic_metrics=True,
            include_explanations=True,
        ),
        visual_reports=VisualReportsConfig(
            enabled=True,
            dpi=160,
            generate_normalized_confusion_matrix=True,
            generate_experiment_comparison_chart=True,
            generate_cv_summary_chart=True,
            generate_feature_distribution_charts=True,
            generate_top_terms_chart=True,
            generate_prediction_confidence_chart=True,
        ),
        outputs=OutputsConfig(
            processed_dir=project_root / "data" / "processed",
            model_dir=project_root / "outputs" / "models",
            metrics_dir=project_root / "outputs" / "metrics",
            figures_dir=project_root / "outputs" / "figures",
            reports_dir=project_root / "outputs" / "reports",
            runs_dir=project_root / "outputs" / "runs",
            latest_dir=project_root / "outputs" / "latest",
        ),
        logging=LoggingConfig(level="INFO", format="text"),
    )


# Permite sobrescrever blocos de configuração mantendo o fixture base imutável
@pytest.fixture()
def make_settings(app_settings: AppSettings) -> Callable[..., AppSettings]:

    def factory(**overrides: object) -> AppSettings:
        return replace(app_settings, **overrides)

    return factory


# Persiste o corpus sintético no caminho bruto configurado
@pytest.fixture()
def raw_dataset_file(app_settings: AppSettings, sample_dataset: pd.DataFrame) -> Path:

    app_settings.dataset.input_path.parent.mkdir(parents=True, exist_ok=True)
    sample_dataset.to_csv(
        app_settings.dataset.input_path, index=False, encoding="utf-8"
    )
    return app_settings.dataset.input_path


# Executa a preparação dos dados uma vez por teste dependente
@pytest.fixture()
def prepared_dataset(app_settings: AppSettings, raw_dataset_file: Path):

    return prepare_dataset(app_settings)


# Gera features linguísticas para os splits preparados
@pytest.fixture()
def feature_datasets(app_settings: AppSettings, prepared_dataset):

    return build_feature_datasets(app_settings)


# Treina o modelo mínimo configurado e retorna os artefatos salvos
@pytest.fixture()
def trained_model_artifacts(app_settings: AppSettings, feature_datasets):

    return train_model(app_settings)
