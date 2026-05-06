from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .environment import collect_environment_report, write_environment_report
from .hashing import build_file_fingerprint
from .paths import ensure_dir, relative_to_root

if TYPE_CHECKING:
    from ..config.settings import AppSettings


class RunContextError(RuntimeError):
    """Erro gerado quando o contexto versionado de execução falha"""


# Representa uma execução versionada em outputs/runs/<run_id>
@dataclass(frozen=True)
class RunContext:

    run_id: str
    run_dir: Path
    models_dir: Path
    metrics_dir: Path
    figures_dir: Path
    reports_dir: Path
    explainability_dir: Path
    latest_dir: Path
    config_snapshot_path: Path
    environment_path: Path
    data_fingerprint_path: Path
    execution_log_path: Path

    # Inicializa diretórios e metadados básicos de uma execução
    @classmethod
    def create(cls, settings: AppSettings) -> RunContext:

        if not settings.run_tracking.enabled:
            return cls.disabled(settings)

        run_id = _build_unique_run_id(settings)
        run_dir = ensure_dir(settings.outputs.runs_dir / run_id)
        context = cls(
            run_id=run_id,
            run_dir=run_dir,
            models_dir=ensure_dir(run_dir / "models"),
            metrics_dir=ensure_dir(run_dir / "metrics"),
            figures_dir=ensure_dir(run_dir / "figures"),
            reports_dir=ensure_dir(run_dir / "reports"),
            explainability_dir=ensure_dir(run_dir / "explainability"),
            latest_dir=ensure_dir(settings.outputs.latest_dir),
            config_snapshot_path=run_dir / "config_snapshot.yaml",
            environment_path=run_dir / "environment.json",
            data_fingerprint_path=run_dir / "data_fingerprint.json",
            execution_log_path=run_dir / "execution_log.txt",
        )
        context.initialize(settings)
        return context

    # Cria um contexto compatível quando run_tracking.enabled=false
    @classmethod
    def disabled(cls, settings: AppSettings) -> RunContext:

        latest_dir = ensure_dir(settings.outputs.latest_dir)
        return cls(
            run_id="disabled",
            run_dir=latest_dir,
            models_dir=ensure_dir(settings.outputs.model_dir),
            metrics_dir=ensure_dir(settings.outputs.metrics_dir),
            figures_dir=ensure_dir(settings.outputs.figures_dir),
            reports_dir=ensure_dir(settings.outputs.reports_dir),
            explainability_dir=ensure_dir(latest_dir / "explainability"),
            latest_dir=latest_dir,
            config_snapshot_path=latest_dir / "config_snapshot.yaml",
            environment_path=latest_dir / "environment.json",
            data_fingerprint_path=latest_dir / "data_fingerprint.json",
            execution_log_path=latest_dir / "execution_log.txt",
        )

    # Salva snapshot de configuração, ambiente e fingerprint do dataset
    def initialize(self, settings: AppSettings) -> None:

        self._touch_execution_log()

        if settings.run_tracking.copy_config_snapshot:
            self.copy_config_snapshot(settings.config_path)

        if settings.run_tracking.save_environment:
            self.save_environment(settings)

        if settings.run_tracking.save_dataset_fingerprint:
            self.save_dataset_fingerprint(settings)

        self.save_manifest(settings)

    # Copia o config.yaml usado na execução sem alterá-lo
    def copy_config_snapshot(self, config_path: Path) -> Path:

        if not config_path.exists():
            raise RunContextError(
                f"Arquivo de configuração não encontrado para snapshot: {config_path}"
            )

        self.config_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(config_path, self.config_snapshot_path)
        except OSError as exc:
            raise RunContextError(
                f"Não foi possível copiar snapshot de configuração para {self.config_snapshot_path}"
            ) from exc
        return self.config_snapshot_path

    # Salva informações de ambiente para reprodutibilidade
    def save_environment(self, settings: AppSettings) -> Path:

        payload = collect_environment_report(settings.project_root)
        if not settings.run_tracking.save_git_commit:
            payload["git"] = None
        return write_environment_report(payload, self.environment_path)

    # Salva fingerprint do dataset bruto oficial, sem modificar o arquivo
    def save_dataset_fingerprint(self, settings: AppSettings) -> Path:

        fingerprint = build_file_fingerprint(settings.dataset.input_path)
        payload = {
            "dataset": fingerprint.to_dict(settings.project_root),
            "note": "Fingerprint calculado somente por leitura; o dataset bruto não é alterado.",
        }
        return self._write_json(payload, self.data_fingerprint_path)

    # Salva um manifesto mínimo da execução versionada
    def save_manifest(self, settings: AppSettings) -> Path:

        payload = {
            "run_id": self.run_id,
            "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "project": {
                "name": settings.project.name,
                "version": settings.project.version,
            },
            "paths": self.to_dict(settings.project_root),
        }
        return self._write_json(payload, self.run_dir / "run_manifest.json")

    # Atualiza outputs/latest/run_id.txt apontando para a execução mais recente
    def update_latest_pointer(self) -> Path:

        self.latest_dir.mkdir(parents=True, exist_ok=True)
        pointer_path = self.latest_dir / "run_id.txt"
        pointer_path.write_text(f"{self.run_id}\n", encoding="utf-8")
        return pointer_path

    # Representa os principais caminhos do contexto
    def to_dict(self, project_root: Path | None = None) -> dict[str, str]:

        def fmt(path: Path) -> str:
            if project_root is None:
                return path.as_posix()
            return relative_to_root(path, project_root)

        return {
            "run_id": self.run_id,
            "run_dir": fmt(self.run_dir),
            "models_dir": fmt(self.models_dir),
            "metrics_dir": fmt(self.metrics_dir),
            "figures_dir": fmt(self.figures_dir),
            "reports_dir": fmt(self.reports_dir),
            "explainability_dir": fmt(self.explainability_dir),
            "latest_dir": fmt(self.latest_dir),
            "config_snapshot_path": fmt(self.config_snapshot_path),
            "environment_path": fmt(self.environment_path),
            "data_fingerprint_path": fmt(self.data_fingerprint_path),
            "execution_log_path": fmt(self.execution_log_path),
        }

    def _touch_execution_log(self) -> None:
        self.execution_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.execution_log_path.exists():
            self.execution_log_path.write_text("", encoding="utf-8")

    def _write_json(self, payload: dict[str, Any], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path


def _build_unique_run_id(settings: AppSettings) -> str:
    base_id = datetime.now(tz=timezone.utc).strftime(
        settings.run_tracking.run_id_format
    )
    candidate = base_id
    suffix = 1
    while (settings.outputs.runs_dir / candidate).exists():
        suffix += 1
        candidate = f"{base_id}_{suffix:02d}"
    return candidate
