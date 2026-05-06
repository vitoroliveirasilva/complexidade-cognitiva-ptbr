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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera explicabilidade global/local do melhor modelo treinado.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Caminho opcional para configs/config.yaml.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Diretório com best_model.joblib e best_experiment.json. Padrão: outputs/latest/models quando existir; caso contrário outputs/models.",
    )
    parser.add_argument(
        "--explainability-dir",
        type=Path,
        default=None,
        help="Diretório de saída da explicabilidade. Padrão: outputs/latest/explainability.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Diretório para gráficos. Padrão: outputs/figures.",
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
        result = run_explain_model(
            config_path=args.config,
            model_dir=args.model_dir,
            explainability_dir=args.explainability_dir,
            figures_dir=args.figures_dir,
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
        sys.stderr.write(f"Erro ao gerar explicabilidade: {exc}\n")
        return EXIT_ERROR


def run_explain_model(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    model_dir: str | Path | None = None,
    explainability_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
) -> dict[str, Any]:
    _ensure_local_src_on_path()

    from complexidade_cognitiva_ptbr.config.settings import load_settings
    from complexidade_cognitiva_ptbr.evaluation.explainability import generate_explainability_artifacts
    from complexidade_cognitiva_ptbr.evaluation.reporting import (
        load_best_experiment,
        load_best_model,
        load_test_frame,
    )
    from complexidade_cognitiva_ptbr.models.train import prepare_model_frame

    settings = load_settings(config_path)
    default_latest_model_dir = settings.outputs.latest_dir / "models"
    resolved_model_dir = Path(model_dir).expanduser() if model_dir is not None else None
    if resolved_model_dir is None:
        resolved_model_dir = default_latest_model_dir if (default_latest_model_dir / "best_model.joblib").exists() else settings.outputs.model_dir
    if not resolved_model_dir.is_absolute():
        resolved_model_dir = settings.project_root / resolved_model_dir

    resolved_explainability_dir = Path(explainability_dir).expanduser() if explainability_dir else settings.outputs.latest_dir / "explainability"
    resolved_figures_dir = Path(figures_dir).expanduser() if figures_dir else settings.outputs.figures_dir
    if not resolved_explainability_dir.is_absolute():
        resolved_explainability_dir = settings.project_root / resolved_explainability_dir
    if not resolved_figures_dir.is_absolute():
        resolved_figures_dir = settings.project_root / resolved_figures_dir

    model = load_best_model(settings, model_dir=resolved_model_dir)
    best_experiment = load_best_experiment(settings, model_dir=resolved_model_dir)
    test_df = load_test_frame(settings)
    text_column = settings.preprocessing.clean_text_column if settings.preprocessing.clean_text_column in test_df.columns else settings.dataset.text_column
    feature_columns = [
        column
        for column in test_df.columns
        if column
        not in {
            settings.dataset.id_column,
            settings.dataset.target_column,
            settings.dataset.text_column,
            settings.preprocessing.clean_text_column,
        }
    ]
    x_test = prepare_model_frame(test_df, settings.dataset.target_column, text_column, feature_columns)
    y_pred = [str(value) for value in model.predict(x_test)]

    result = generate_explainability_artifacts(
        settings=settings,
        model=model,
        best_experiment=best_experiment,
        test_df=test_df,
        y_pred=y_pred,
        explainability_dir=resolved_explainability_dir,
        figures_dir=resolved_figures_dir,
    )
    return result.to_dict(settings.project_root)


# Adiciona `src` ao `sys.path` quando o script roda direto do repositório.
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
