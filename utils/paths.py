from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.settings import AppSettings

_PROJECT_MARKERS = ("pyproject.toml", "configs", "src")


class PathResolutionError(RuntimeError):
    """Erro gerado quando um caminho esperado é inválido para o pipeline"""


# Localiza a raiz do projeto com base em marcadores estruturais
def find_project_root(start: str | Path | None = None) -> Path:

    current = _resolve_start_path(start)
    candidates = (current, *current.parents)

    for candidate in candidates:
        if _has_project_markers(candidate):
            return candidate

    return current


# Cria um diretório quando ele ainda não existe e retorna o caminho resolvido
def ensure_dir(path: str | Path) -> Path:

    directory = Path(path).expanduser().resolve()

    if directory.exists() and not directory.is_dir():
        raise PathResolutionError(
            f"O caminho existe, mas não é um diretório: {directory}"
        )

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PathResolutionError(
            f"Não foi possível criar o diretório {directory}: {exc}"
        ) from exc

    return directory


# Garante a existência dos diretórios gerenciados pela configuração
def ensure_project_directories(settings: AppSettings) -> tuple[Path, ...]:

    created_or_existing: list[Path] = []
    seen: set[Path] = set()

    for directory in settings.managed_directories:
        resolved_directory = ensure_dir(directory)
        if resolved_directory not in seen:
            created_or_existing.append(resolved_directory)
            seen.add(resolved_directory)

    return tuple(created_or_existing)


# Retorna uma representação relativa à raiz quando possível
def relative_to_root(path: str | Path, project_root: str | Path) -> str:

    resolved_path = Path(path).expanduser().resolve()
    resolved_root = Path(project_root).expanduser().resolve()

    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def _resolve_start_path(start: str | Path | None) -> Path:
    current = Path(start or Path.cwd()).expanduser().resolve()
    if current.exists() and current.is_file():
        return current.parent
    return current


def _has_project_markers(candidate: Path) -> bool:
    return all((candidate / marker).exists() for marker in _PROJECT_MARKERS)
