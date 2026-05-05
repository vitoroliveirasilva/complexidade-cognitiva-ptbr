from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, TextIO

_ALLOWED_LOG_FORMATS = {"json", "text"}
_PROJECT_HANDLER_MARKER = "_complexidade_cognitiva_ptbr_handler"


class LoggingConfigurationError(ValueError):
    """Erro gerado quando a configuração de logging é inválida"""


# Formatter simples para logs estruturados em JSON
class JsonFormatter(logging.Formatter):

    # Formata um registro de log em JSON serializável
    def format(self, record: logging.LogRecord) -> str:

        payload: dict[str, Any] = {
            "timestamp": _format_utc_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


# Configura logging para execução dos scripts e do pipeline
def configure_logging(level: str = "INFO", output_format: str = "text") -> None:

    numeric_level = _resolve_logging_level(level)
    normalized_format = _resolve_output_format(output_format)

    root_logger = logging.getLogger()
    _remove_project_handlers(root_logger)
    root_logger.setLevel(numeric_level)

    handler = _build_stream_handler(
        stream=sys.stdout,
        numeric_level=numeric_level,
        output_format=normalized_format,
    )
    setattr(handler, _PROJECT_HANDLER_MARKER, True)
    root_logger.addHandler(handler)

    logging.captureWarnings(True)


# Retorna um logger nomeado para os módulos do projeto
def get_logger(name: str) -> logging.Logger:

    if not isinstance(name, str) or not name.strip():
        raise LoggingConfigurationError("O nome do logger deve ser um texto não vazio.")

    return logging.getLogger(name.strip())


def _build_stream_handler(
    *,
    stream: TextIO,
    numeric_level: int,
    output_format: str,
) -> logging.StreamHandler[TextIO]:
    handler: logging.StreamHandler[TextIO] = logging.StreamHandler(stream)
    handler.setLevel(numeric_level)

    if output_format == "json":
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    return handler


# Remove apenas handlers criados por este módulo
def _remove_project_handlers(logger: logging.Logger) -> None:

    for handler in list(logger.handlers):
        if getattr(handler, _PROJECT_HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()


def _resolve_logging_level(level: str) -> int:
    if not isinstance(level, str) or not level.strip():
        raise LoggingConfigurationError("logging.level deve ser um texto não vazio.")

    normalized_level = level.strip().upper()
    numeric_level = logging.getLevelName(normalized_level)

    if not isinstance(numeric_level, int):
        allowed_levels = sorted(logging._nameToLevel)
        raise LoggingConfigurationError(
            f"logging.level inválido: {normalized_level}. Valores aceitos: {allowed_levels}."
        )

    return numeric_level


def _resolve_output_format(output_format: str) -> str:
    if not isinstance(output_format, str) or not output_format.strip():
        raise LoggingConfigurationError("logging.format deve ser um texto não vazio.")

    normalized_format = output_format.strip().lower()
    if normalized_format not in _ALLOWED_LOG_FORMATS:
        allowed = ", ".join(sorted(_ALLOWED_LOG_FORMATS))
        raise LoggingConfigurationError(
            f"logging.format inválido: {normalized_format}. Valores aceitos: {allowed}."
        )

    return normalized_format


def _format_utc_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
