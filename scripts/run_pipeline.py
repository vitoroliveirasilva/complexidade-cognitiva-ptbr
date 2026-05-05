from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130


# Constrói o parser de argumentos da CLI
def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Executa as etapas disponíveis do pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Caminho do arquivo YAML de configuração.",
    )
    parser.add_argument(
        "--bootstrap-only",
        action="store_true",
        help="Valida apenas configuração, logging e diretórios, sem exigir dataset bruto.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Exibe traceback completo em caso de erro.",
    )
    return parser


# Ponto de entrada da CLI
def main(argv: list[str] | None = None) -> int:

    args = build_parser().parse_args(argv)

    try:
        result = run_pipeline(
            config_path=args.config,
            bootstrap_only=args.bootstrap_only,
        )
        sys.stdout.write(f"{result.to_json()}\n")
        return EXIT_SUCCESS
    except KeyboardInterrupt:
        sys.stderr.write("Execução interrompida pelo usuário.\n")
        return EXIT_INTERRUPTED
    except Exception as exc:
        if args.debug:
            raise
        sys.stderr.write(f"Erro ao executar o pipeline: {exc}\n")
        return EXIT_ERROR


# Executa o pipeline completo ou apenas o bootstrap estrutural
def run_pipeline(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    bootstrap_only: bool = False,
) -> Any:

    pipeline_cls = _load_pipeline_class()
    pipeline = pipeline_cls.from_config(config_path)

    if bootstrap_only:
        return pipeline.bootstrap(stage="run_pipeline_bootstrap")

    return pipeline.run()


# Carrega a classe do pipeline priorizando o pacote local em `src`
def _load_pipeline_class() -> type[Any]:

    _ensure_local_src_on_path()
    from complexidade_cognitiva_ptbr.pipeline import CognitiveComplexityPipeline

    return CognitiveComplexityPipeline


# Adiciona `src` ao `sys.path` quando o script roda direto do repositório
def _ensure_local_src_on_path() -> None:

    src_dir = Path(__file__).resolve().parents[1] / "src"
    if not src_dir.is_dir():
        return

    resolved_src = src_dir.resolve()
    normalized_paths = {Path(entry).resolve() for entry in sys.path if entry}
    if resolved_src not in normalized_paths:
        sys.path.insert(0, resolved_src.as_posix())


if __name__ == "__main__":
    raise SystemExit(main())
