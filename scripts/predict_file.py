from __future__ import annotations

import argparse
import sys
from pathlib import Path
DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
DEFAULT_OUTPUT_PATH = Path("outputs/latest/predictions/predictions.csv")
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130


# Constrói o parser de argumentos da CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classifica localmente um CSV de textos usando o último ModelBundle treinado.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="CSV de entrada com textos.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="CSV de saída das predições.")
    parser.add_argument("--text-column", default=None, help="Nome da coluna textual do CSV.")
    parser.add_argument("--id-column", default=None, help="Nome opcional da coluna de identificador.")
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
    parser.add_argument("--debug", action="store_true", help="Exibe traceback completo em caso de erro.")
    return parser


# Ponto de entrada da CLI
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_path = run_predict_file(
            input_path=args.input,
            output_path=args.output,
            config_path=args.config,
            text_column=args.text_column,
            id_column=args.id_column,
            model_path=args.model_path,
        )
        sys.stdout.write(f"Predições salvas em: {output_path}\n")
        return EXIT_SUCCESS
    except KeyboardInterrupt:
        sys.stderr.write("Execução interrompida pelo usuário.\n")
        return EXIT_INTERRUPTED
    except Exception as exc:
        if args.debug:
            raise
        sys.stderr.write(f"Erro ao predizer arquivo: {exc}\n")
        return EXIT_ERROR


# Executa a inferência local em lote
def run_predict_file(
    *,
    input_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    text_column: str | None = None,
    id_column: str | None = None,
    model_path: str | Path | None = None,
) -> Path:
    _ensure_local_src_on_path()
    from complexidade_cognitiva_ptbr.config.settings import load_settings
    from complexidade_cognitiva_ptbr.models.inference import predict_file

    settings = load_settings(config_path)
    return predict_file(
        settings,
        input_path=input_path,
        output_path=output_path,
        text_column=text_column,
        id_column=id_column,
        model_path=model_path,
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
