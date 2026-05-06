from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from complexidade_cognitiva_ptbr.data.leakage import (
    LeakageError,
    character_ngram_jaccard,
    detect_leakage,
    normalize_text_for_leakage,
    run_leakage_checks,
)


def _split_frames() -> dict[str, pd.DataFrame]:
    return {
        "train": pd.DataFrame(
            {
                "id": [1, 2],
                "texto": ["Texto simples, com sol.", "Outra narrativa breve."],
                "texto_limpo": ["Texto simples, com sol.", "Outra narrativa breve."],
                "target": ["baixa", "media"],
            }
        ),
        "val": pd.DataFrame(
            {
                "id": [3, 4],
                "texto": ["texto simples com sol", "Cena simbólica complexa."],
                "texto_limpo": ["texto simples com sol", "Cena simbólica complexa."],
                "target": ["baixa", "alta"],
            }
        ),
        "test": pd.DataFrame(
            {
                "id": [5, 6],
                "texto": ["Leitura curta.", "Discurso metafórico."],
                "texto_limpo": ["Leitura curta.", "Discurso metafórico."],
                "target": ["baixa", "alta"],
            }
        ),
    }


def test_normalize_text_for_leakage_ignores_case_punctuation_and_spaces() -> None:
    assert normalize_text_for_leakage("  Texto,   SIMPLES!!! ") == "texto simples"


def test_character_ngram_jaccard_detects_close_texts() -> None:
    similarity = character_ngram_jaccard(
        "Texto literário simples", "texto literario simples"
    )

    assert 0.7 <= similarity <= 1.0


def test_detect_leakage_reports_normalized_text_overlap(app_settings) -> None:
    report, near_pairs = detect_leakage(_split_frames(), app_settings)

    checks = {finding["check"] for finding in report["findings"]}
    assert "normalized_text_overlap" in checks
    assert report["summary"]["critical_findings"] >= 1
    assert isinstance(near_pairs, pd.DataFrame)


def test_detect_leakage_reports_id_overlap(app_settings) -> None:
    splits = _split_frames()
    splits["test"].loc[0, "id"] = 1

    report, _ = detect_leakage(splits, app_settings)

    id_findings = [f for f in report["findings"] if f["check"] == "id_overlap"]
    assert id_findings
    assert id_findings[0]["severity"] == "critical"


def test_detect_leakage_reports_target_like_columns(app_settings) -> None:
    splits = _split_frames()
    splits["train"]["target_encoded"] = [0, 1]

    report, _ = detect_leakage(splits, app_settings)

    assert any(f["check"] == "target_like_columns" for f in report["findings"])


def test_run_leakage_checks_persists_report_and_can_fail(
    make_settings, app_settings, prepared_dataset
) -> None:
    processed_dir = app_settings.outputs.processed_dir
    train = pd.read_csv(processed_dir / "train.csv")
    val = pd.read_csv(processed_dir / "val.csv")
    val.loc[val.index[0], "id"] = train.loc[train.index[0], "id"]
    val.to_csv(processed_dir / "val.csv", index=False, encoding="utf-8")

    with pytest.raises(LeakageError, match="Vazamento crítico"):
        run_leakage_checks(app_settings)

    leakage = replace(app_settings.leakage_checks, fail_on_critical_leakage=False)
    settings = make_settings(leakage_checks=leakage)
    result = run_leakage_checks(settings)

    assert result.report_json_path.exists()
    assert result.report_markdown_path.exists()
    assert result.near_duplicate_pairs_path.exists()
    assert result.report["summary"]["critical_findings"] >= 1
