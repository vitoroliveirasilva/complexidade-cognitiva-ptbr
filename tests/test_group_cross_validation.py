from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, SRC.as_posix())

from complexidade_cognitiva_ptbr.evaluation.group_cross_validation import (
    run_group_cross_validation,
    select_group_column,
)


def _grouped_dataset() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    labels = ["baixa", "media", "alta"]
    for group_idx in range(6):
        for label in labels:
            rows.append(
                {
                    "id": f"{group_idx}_{label}",
                    "texto": (
                        f"Texto literário do grupo {group_idx} com sinais da classe {label}. "
                        "A cena apresenta memória, descrição e organização narrativa suficiente para teste."
                    ),
                    "target": label,
                    "obra": f"obra_{group_idx}",
                    "autor": f"autor_{group_idx // 2}",
                }
            )
    return pd.DataFrame(rows)


def test_select_group_column_prefers_obra() -> None:
    df = _grouped_dataset()

    selection = select_group_column(
        df,
        preferred_group_column="obra",
        fallback_group_columns=["autor"],
    )

    assert selection.column == "obra"
    assert selection.strategy == "preferred_column"
    assert selection.groups.nunique() == 6


def test_group_cross_validation_has_no_group_overlap(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    json_path = tmp_path / "group_cv.json"
    md_path = tmp_path / "group_cv.md"
    _grouped_dataset().to_csv(dataset_path, index=False, encoding="utf-8")

    result = run_group_cross_validation(
        input_path=dataset_path,
        output_json=json_path,
        output_md=md_path,
        preferred_group_column="obra",
        n_splits=3,
        max_features=100,
    )

    assert result["status"] == "completed"
    assert result["group_column"] == "obra"
    assert result["effective_n_splits"] == 3
    assert all(fold["group_overlap_count"] == 0 for fold in result["folds"])
    assert json_path.exists()
    assert md_path.exists()

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["aggregated_confusion_matrix"]


def test_group_cross_validation_fallbacks_to_author(tmp_path: Path) -> None:
    df = _grouped_dataset().drop(columns=["obra"])
    dataset_path = tmp_path / "dataset.csv"
    json_path = tmp_path / "group_cv.json"
    md_path = tmp_path / "group_cv.md"
    df.to_csv(dataset_path, index=False, encoding="utf-8")

    result = run_group_cross_validation(
        input_path=dataset_path,
        output_json=json_path,
        output_md=md_path,
        preferred_group_column="obra",
        fallback_group_columns=["autor"],
        n_splits=3,
        max_features=100,
    )

    assert result["group_column"] == "autor"
    assert result["group_strategy"] == "fallback_column"
    assert all(fold["group_overlap_count"] == 0 for fold in result["folds"])


def test_group_cross_validation_uses_safe_row_fallback_when_no_group_column(
    tmp_path: Path,
) -> None:
    df = _grouped_dataset().drop(columns=["obra", "autor"])
    dataset_path = tmp_path / "dataset.csv"
    json_path = tmp_path / "group_cv.json"
    md_path = tmp_path / "group_cv.md"
    df.to_csv(dataset_path, index=False, encoding="utf-8")

    result = run_group_cross_validation(
        input_path=dataset_path,
        output_json=json_path,
        output_md=md_path,
        preferred_group_column="obra",
        fallback_group_columns=["autor", "fonte"],
        fallback_strategy="row_id",
        n_splits=3,
        max_features=100,
    )

    assert result["group_column"] == "__row_group__"
    assert result["group_strategy"] == "synthetic_row_id"
    assert result["effective_n_splits"] == 3
    assert all(fold["group_overlap_count"] == 0 for fold in result["folds"])
    assert any("fallback" in warning.casefold() for warning in result["warnings"])
