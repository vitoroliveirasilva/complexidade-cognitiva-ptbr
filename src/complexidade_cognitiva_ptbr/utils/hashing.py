from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_CHUNK_SIZE = 1024 * 1024


class FingerprintError(RuntimeError):
    """Erro gerado quando não é possível calcular fingerprint de arquivo"""


# Resumo rastreável de um arquivo de entrada
@dataclass(frozen=True)
class FileFingerprint:

    path: Path
    exists: bool
    size_bytes: int | None
    sha256: str | None
    modified_at_utc: str | None
    generated_at_utc: str

    def to_dict(self, project_root: Path | None = None) -> dict[str, Any]:
        if project_root is None:
            path = self.path.as_posix()
        else:
            path = _relative_to_root(self.path, project_root)

        return {
            "path": path,
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "modified_at_utc": self.modified_at_utc,
            "generated_at_utc": self.generated_at_utc,
        }


# Calcula metadados e SHA-256 de um arquivo sem modificar seu conteúdo
def build_file_fingerprint(path: str | Path) -> FileFingerprint:

    resolved_path = Path(path).expanduser().resolve()
    generated_at = _utc_now()

    if not resolved_path.exists():
        return FileFingerprint(
            path=resolved_path,
            exists=False,
            size_bytes=None,
            sha256=None,
            modified_at_utc=None,
            generated_at_utc=generated_at,
        )

    if not resolved_path.is_file():
        raise FingerprintError(f"O caminho não aponta para um arquivo: {resolved_path}")

    try:
        stat = resolved_path.stat()
        digest = hash_file(resolved_path)
    except OSError as exc:
        raise FingerprintError(
            f"Não foi possível acessar o arquivo para fingerprint: {resolved_path}"
        ) from exc

    return FileFingerprint(
        path=resolved_path,
        exists=True,
        size_bytes=int(stat.st_size),
        sha256=digest,
        modified_at_utc=datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        generated_at_utc=generated_at,
    )


# Retorna o SHA-256 hexadecimal de um arquivo
def hash_file(path: str | Path, chunk_size: int = _DEFAULT_CHUNK_SIZE) -> str:

    if chunk_size <= 0:
        raise FingerprintError("chunk_size deve ser maior que zero.")

    resolved_path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()

    try:
        with resolved_path.open("rb") as file:
            for chunk in iter(lambda: file.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FingerprintError(
            f"Não foi possível ler o arquivo: {resolved_path}"
        ) from exc

    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _relative_to_root(path: Path, project_root: Path) -> str:
    resolved_path = path.expanduser().resolve()
    resolved_root = project_root.expanduser().resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError:
        return resolved_path.as_posix()
