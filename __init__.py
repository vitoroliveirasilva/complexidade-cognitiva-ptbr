from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

_PACKAGE_NAME = "complexidade-cognitiva-ptbr"
_FALLBACK_VERSION = "0.1.0"


# Obtém a versão instalada do pacote, com fallback para execução local
def _resolve_version() -> str:

    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return _FALLBACK_VERSION


__version__ = _resolve_version()
