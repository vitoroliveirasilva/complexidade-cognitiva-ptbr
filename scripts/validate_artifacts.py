from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INVALID_ARTIFACTS = 2
EXIT_INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Valida os artefatos finais gerados por uma execução versionada do pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Caminho do arquivo YAML de configuração.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Diretório outputs/runs/<run_id> a validar. Se omitido, usa outputs/latest/run_id.txt.",
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "complete"),
        default="complete",
        help="Conjunto de artefatos esperados.",
    )
    parser.add_argument(
        "--strict-optional",
        action="store_true",
        help="No perfil complete, trata artefatos normalmente opcionais como obrigatórios.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Exibe traceback completo em caso de erro.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_validate_artifacts(
            config_path=args.config,
            run_dir=args.run_dir,
            profile=args.profile,
            strict_optional=args.strict_optional,
        )
        sys.stdout.write(f"{result.to_json(_project_root(args.config))}\n")
        return EXIT_SUCCESS if result.passed else EXIT_INVALID_ARTIFACTS
    except KeyboardInterrupt:
        sys.stderr.write("Validação interrompida pelo usuário.\n")
        return EXIT_INTERRUPTED
    except Exception as exc:
        if args.debug:
            raise
        sys.stderr.write(f"Erro ao validar artefatos: {exc}\n")
        return EXIT_ERROR


def run_validate_artifacts(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    run_dir: str | Path | None = None,
    profile: str = "complete",
    strict_optional: bool = False,
) -> Any:
    _ensure_local_src_on_path()
    from complexidade_cognitiva_ptbr.config.settings import load_settings
    from complexidade_cognitiva_ptbr.utils.artifact_validation import validate_artifacts

    settings = load_settings(config_path)
    return validate_artifacts(
        settings,
        run_dir=run_dir,
        profile=profile,
        strict_optional=strict_optional,
    )


def _project_root(config_path: str | Path) -> Path | None:
    try:
        _ensure_local_src_on_path()
        from complexidade_cognitiva_ptbr.config.settings import load_settings

        return load_settings(config_path).project_root
    except Exception:
        return None


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
