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
        description="Classifica localmente um texto usando o último ModelBundle treinado.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--text", required=True, help="Texto em português a ser classificado.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Caminho do arquivo YAML de configuração.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Caminho opcional para um best_model_bundle.joblib específico.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime a saída completa em JSON.",
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
        result = run_predict_text(
            text=args.text,
            config_path=args.config,
            model_path=args.model_path,
        )
        if args.json:
            sys.stdout.write(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            sys.stdout.write("\n")
        else:
            from complexidade_cognitiva_ptbr.models.inference import prediction_to_console_text

            sys.stdout.write(f"{prediction_to_console_text(result)}\n")
        return EXIT_SUCCESS
    except KeyboardInterrupt:
        sys.stderr.write("Execução interrompida pelo usuário.\n")
        return EXIT_INTERRUPTED
    except Exception as exc:
        if args.debug:
            raise
        sys.stderr.write(f"Erro ao predizer texto: {exc}\n")
        return EXIT_ERROR


# Executa a inferência local de texto único
def run_predict_text(
    *,
    text: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    model_path: str | Path | None = None,
) -> Any:
    _ensure_local_src_on_path()
    from complexidade_cognitiva_ptbr.config.settings import load_settings
    from complexidade_cognitiva_ptbr.models.inference import predict_text

    settings = load_settings(config_path)
    return predict_text(settings, text, model_path=model_path)


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
