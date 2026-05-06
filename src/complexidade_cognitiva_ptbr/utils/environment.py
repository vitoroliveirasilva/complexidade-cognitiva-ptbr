from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_IMPORTANT_PACKAGES = (
    "pandas",
    "numpy",
    "scikit-learn",
    "nltk",
    "matplotlib",
    "PyYAML",
    "joblib",
)


class EnvironmentReportError(RuntimeError):
    """Erro gerado quando o relatório de ambiente não pode ser salvo"""


# Coleta informações mínimas para reproduzir uma execução do pipeline
def collect_environment_report(project_root: str | Path) -> dict[str, Any]:

    root = Path(project_root).expanduser().resolve()
    return {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "packages": _collect_package_versions(),
        "git": _collect_git_metadata(root),
    }


# Salva o relatório de ambiente em JSON indentado e UTF-8
def write_environment_report(payload: dict[str, Any], output_path: str | Path) -> Path:

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as exc:
        raise EnvironmentReportError(
            f"Não foi possível salvar relatório de ambiente em {path}"
        ) from exc

    return path


def _collect_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in _IMPORTANT_PACKAGES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _collect_git_metadata(project_root: Path) -> dict[str, Any]:
    return {
        "commit": _run_git_command(project_root, "rev-parse", "HEAD"),
        "branch": _run_git_command(project_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "is_dirty": _is_git_dirty(project_root),
    }


def _is_git_dirty(project_root: Path) -> bool | None:
    output = _run_git_command(project_root, "status", "--porcelain")
    if output is None:
        return None
    return bool(output.strip())


def _run_git_command(project_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None

    return completed.stdout.strip() or None
