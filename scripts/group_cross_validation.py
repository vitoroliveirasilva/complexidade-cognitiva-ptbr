from __future__ import annotations

import argparse
import json
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
        description="Executa avaliação complementar por GroupKFold usando obra/autor/fonte como agrupamento.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Caminho do arquivo YAML de configuração.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="CSV de entrada. Se omitido, usa dataset.input_path do YAML.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Arquivo JSON de saída. Se omitido, usa group_cross_validation.output_json ou outputs.metrics_dir.",
    )
    parser.add_argument(
        "--md",
        type=Path,
        default=None,
        help="Relatório Markdown de saída. Se omitido, usa group_cross_validation.output_md ou outputs.reports_dir.",
    )
    parser.add_argument(
        "--group-column",
        default=None,
        help="Coluna preferencial de agrupamento, por exemplo obra ou autor.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=None,
        help="Número de folds. Se omitido, usa group_cross_validation.n_splits.",
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
        result = run_group_cross_validation_cli(
            config_path=args.config,
            input_path=args.input,
            output_json=args.json,
            output_md=args.md,
            group_column=args.group_column,
            n_splits=args.n_splits,
        )
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2))
        sys.stdout.write("\n")
        return EXIT_SUCCESS
    except KeyboardInterrupt:
        sys.stderr.write("Execução interrompida pelo usuário.\n")
        return EXIT_INTERRUPTED
    except Exception as exc:
        if args.debug:
            raise
        sys.stderr.write(f"Erro ao executar avaliação por agrupamento: {exc}\n")
        return EXIT_ERROR


# Executa a avaliação complementar por agrupamento
def run_group_cross_validation_cli(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    input_path: str | Path | None = None,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
    group_column: str | None = None,
    n_splits: int | None = None,
) -> dict[str, Any]:
    _ensure_local_src_on_path()
    from complexidade_cognitiva_ptbr.evaluation.group_cross_validation import (
        run_group_cross_validation_from_config,
    )

    return run_group_cross_validation_from_config(
        config_path=config_path,
        input_path=input_path,
        output_json=output_json,
        output_md=output_md,
        preferred_group_column=group_column,
        n_splits=n_splits,
    )


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
