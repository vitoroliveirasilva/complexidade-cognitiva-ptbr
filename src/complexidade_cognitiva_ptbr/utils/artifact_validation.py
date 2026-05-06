from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .paths import relative_to_root

if TYPE_CHECKING:
    from ..config.settings import AppSettings


class ArtifactValidationError(RuntimeError):
    """Erro gerado quando a validação de artefatos não pode ser concluída"""


# Define um artefato esperado em uma execução do pipeline
@dataclass(frozen=True)
class ArtifactExpectation:

    relative_path: str
    description: str
    required: bool = True
    min_size_bytes: int = 1


# Resultado individual da validação de um artefato
@dataclass(frozen=True)
class ArtifactCheck:

    relative_path: str
    description: str
    required: bool
    exists: bool
    is_file: bool
    size_bytes: int | None
    status: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "description": self.description,
            "required": self.required,
            "exists": self.exists,
            "is_file": self.is_file,
            "size_bytes": self.size_bytes,
            "status": self.status,
            "message": self.message,
        }


# Resumo serializável da validação final de artefatos
@dataclass(frozen=True)
class ArtifactValidationResult:

    generated_at_utc: str
    profile: str
    base_dir: Path
    passed: bool
    checks: tuple[ArtifactCheck, ...]

    @property
    def required_failures(self) -> tuple[ArtifactCheck, ...]:
        return tuple(
            check for check in self.checks if check.required and check.status != "ok"
        )

    @property
    def optional_failures(self) -> tuple[ArtifactCheck, ...]:
        return tuple(
            check
            for check in self.checks
            if not check.required and check.status != "ok"
        )

    def to_dict(self, project_root: Path | None = None) -> dict[str, Any]:
        base_dir = self.base_dir.as_posix()
        if project_root is not None:
            base_dir = relative_to_root(self.base_dir, project_root)
        return {
            "generated_at_utc": self.generated_at_utc,
            "profile": self.profile,
            "base_dir": base_dir,
            "passed": self.passed,
            "summary": {
                "total_checks": len(self.checks),
                "required_checks": sum(1 for check in self.checks if check.required),
                "optional_checks": sum(
                    1 for check in self.checks if not check.required
                ),
                "required_failures": len(self.required_failures),
                "optional_failures": len(self.optional_failures),
            },
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_json(self, project_root: Path | None = None) -> str:
        return json.dumps(self.to_dict(project_root), ensure_ascii=False, indent=2)


_CORE_EXPECTATIONS = (
    ArtifactExpectation(
        "config_snapshot.yaml", "snapshot da configuração usada na execução"
    ),
    ArtifactExpectation("environment.json", "relatório do ambiente de execução"),
    ArtifactExpectation("data_fingerprint.json", "fingerprint do dataset bruto"),
    ArtifactExpectation("run_manifest.json", "manifesto da execução versionada"),
    ArtifactExpectation(
        "metrics/leakage_report.json", "relatório JSON de vazamento de dados"
    ),
    ArtifactExpectation(
        "reports/leakage_report.md", "relatório Markdown de vazamento de dados"
    ),
    ArtifactExpectation(
        "metrics/cv_results.csv", "resultados por fold da validação cruzada"
    ),
    ArtifactExpectation(
        "metrics/cv_summary.json", "resumo agregado da validação cruzada"
    ),
    ArtifactExpectation(
        "reports/cv_report.md", "relatório Markdown da validação cruzada"
    ),
    ArtifactExpectation(
        "metrics/hyperparameter_search_results.csv",
        "resultados da busca de hiperparâmetros",
    ),
    ArtifactExpectation(
        "metrics/best_hyperparameters.json", "melhores hiperparâmetros selecionados"
    ),
    ArtifactExpectation(
        "reports/hyperparameter_search_report.md",
        "relatório da busca de hiperparâmetros",
    ),
    ArtifactExpectation(
        "models/best_model_bundle.joblib",
        "ModelBundle persistido para inferência local",
    ),
    ArtifactExpectation(
        "models/best_experiment.json", "descrição do melhor experimento"
    ),
    ArtifactExpectation(
        "models/experiment_results.csv", "comparação dos experimentos treinados"
    ),
    ArtifactExpectation(
        "metrics/final_metrics.json", "métricas finais no conjunto de teste"
    ),
    ArtifactExpectation(
        "metrics/classification_report.json", "classification report em JSON"
    ),
    ArtifactExpectation(
        "metrics/classification_report.txt", "classification report textual"
    ),
    ArtifactExpectation(
        "metrics/test_predictions.csv", "predições individuais no conjunto de teste"
    ),
    ArtifactExpectation(
        "figures/confusion_matrix_absolute.png", "matriz de confusão absoluta"
    ),
    ArtifactExpectation("reports/final_report.md", "relatório final resumido"),
    ArtifactExpectation(
        "reports/final_report_extended.md", "relatório final estendido"
    ),
)

_OPTIONAL_EXPECTATIONS = (
    ArtifactExpectation(
        "metrics/near_duplicate_pairs.csv",
        "pares de duplicidade aproximada",
        required=False,
        min_size_bytes=0,
    ),
    ArtifactExpectation(
        "figures/confusion_matrix_normalized.png",
        "matriz de confusão normalizada",
        required=False,
    ),
    ArtifactExpectation(
        "figures/experiment_comparison_f1_macro.png",
        "comparação visual de experimentos",
        required=False,
    ),
    ArtifactExpectation(
        "figures/cv_summary_f1_macro.png",
        "resumo visual de validação cruzada",
        required=False,
    ),
    ArtifactExpectation(
        "figures/prediction_confidence_distribution.png",
        "distribuição de confiança das predições",
        required=False,
    ),
    ArtifactExpectation(
        "explainability/tfidf_top_terms_by_class.csv",
        "top termos TF-IDF por classe em CSV",
        required=False,
    ),
    ArtifactExpectation(
        "explainability/tfidf_top_terms_by_class.json",
        "top termos TF-IDF por classe em JSON",
        required=False,
    ),
    ArtifactExpectation(
        "explainability/tfidf_explainability_report.md",
        "relatório de explicabilidade TF-IDF",
        required=False,
    ),
    ArtifactExpectation(
        "explainability/local_explanations_sample.csv",
        "amostras de explicação local",
        required=False,
    ),
)

_SMOKE_EXPECTATIONS = (
    ArtifactExpectation(
        "config_snapshot.yaml", "snapshot da configuração usada na execução"
    ),
    ArtifactExpectation("environment.json", "relatório do ambiente de execução"),
    ArtifactExpectation("data_fingerprint.json", "fingerprint do dataset bruto"),
    ArtifactExpectation("run_manifest.json", "manifesto da execução versionada"),
    ArtifactExpectation(
        "metrics/leakage_report.json", "relatório JSON de vazamento de dados"
    ),
    ArtifactExpectation(
        "metrics/cv_summary.json", "resumo agregado da validação cruzada"
    ),
    ArtifactExpectation(
        "models/best_model_bundle.joblib",
        "ModelBundle persistido para inferência local",
    ),
    ArtifactExpectation(
        "models/best_experiment.json", "descrição do melhor experimento"
    ),
    ArtifactExpectation(
        "metrics/final_metrics.json", "métricas finais no conjunto de teste"
    ),
    ArtifactExpectation(
        "reports/final_report_extended.md", "relatório final estendido"
    ),
)


# Resolve outputs/runs/<run_id> a partir de outputs/latest/run_id.txt
def resolve_latest_run_dir(settings: AppSettings) -> Path:

    pointer_path = settings.outputs.latest_dir / "run_id.txt"
    if not pointer_path.exists():
        raise ArtifactValidationError(
            f"Ponteiro da última execução não encontrado: {pointer_path}. "
            "Execute primeiro python scripts/run_pipeline.py."
        )

    run_id = pointer_path.read_text(encoding="utf-8").strip()
    if not run_id:
        raise ArtifactValidationError(
            f"Ponteiro da última execução está vazio: {pointer_path}"
        )

    run_dir = settings.outputs.runs_dir / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise ArtifactValidationError(
            f"Diretório da última execução não encontrado: {run_dir}"
        )
    return run_dir


# Valida artefatos esperados para banca, CI e smoke test local
def validate_artifacts(
    settings: AppSettings,
    *,
    run_dir: str | Path | None = None,
    profile: str = "complete",
    strict_optional: bool = False,
) -> ArtifactValidationResult:

    normalized_profile = profile.strip().lower()
    if normalized_profile not in {"smoke", "complete"}:
        raise ArtifactValidationError("profile deve ser 'smoke' ou 'complete'.")

    base_dir = _resolve_base_dir(settings, run_dir)
    expectations = _build_expectations(normalized_profile, strict_optional)
    checks = tuple(
        _check_expectation(base_dir, expectation) for expectation in expectations
    )
    passed = all(check.status == "ok" for check in checks if check.required)

    return ArtifactValidationResult(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        profile=normalized_profile,
        base_dir=base_dir,
        passed=passed,
        checks=checks,
    )


def _resolve_base_dir(settings: AppSettings, run_dir: str | Path | None) -> Path:
    if run_dir is not None:
        candidate = Path(run_dir).expanduser()
        if not candidate.is_absolute():
            candidate = settings.project_root / candidate
        candidate = candidate.resolve()
    else:
        candidate = resolve_latest_run_dir(settings).resolve()

    if not candidate.exists() or not candidate.is_dir():
        raise ArtifactValidationError(f"Diretório de artefatos inválido: {candidate}")
    return candidate


def _build_expectations(
    profile: str, strict_optional: bool
) -> tuple[ArtifactExpectation, ...]:
    if profile == "smoke":
        return _SMOKE_EXPECTATIONS

    if not strict_optional:
        return (*_CORE_EXPECTATIONS, *_OPTIONAL_EXPECTATIONS)

    strict_items = [
        ArtifactExpectation(
            expectation.relative_path,
            expectation.description,
            required=True,
            min_size_bytes=expectation.min_size_bytes,
        )
        for expectation in (*_CORE_EXPECTATIONS, *_OPTIONAL_EXPECTATIONS)
    ]
    return tuple(strict_items)


def _check_expectation(
    base_dir: Path, expectation: ArtifactExpectation
) -> ArtifactCheck:
    path = base_dir / expectation.relative_path
    exists = path.exists()
    is_file = path.is_file()
    size_bytes = path.stat().st_size if exists and is_file else None

    if not exists:
        status = "missing"
        message = (
            "Artefato obrigatório ausente."
            if expectation.required
            else "Artefato opcional ausente."
        )
    elif not is_file:
        status = "invalid"
        message = "O caminho existe, mas não é um arquivo."
    elif size_bytes is not None and size_bytes < expectation.min_size_bytes:
        status = "empty"
        message = f"Arquivo menor que o mínimo esperado ({expectation.min_size_bytes} byte(s))."
    else:
        status = "ok"
        message = "Artefato encontrado."

    return ArtifactCheck(
        relative_path=expectation.relative_path,
        description=expectation.description,
        required=expectation.required,
        exists=exists,
        is_file=is_file,
        size_bytes=size_bytes,
        status=status,
        message=message,
    )
