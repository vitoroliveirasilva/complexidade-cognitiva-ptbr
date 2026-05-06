from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from ..config.settings import AppSettings
from ..data.io import read_csv_dataset, write_json
from ..utils.paths import relative_to_root

_NORMALIZE_SPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^0-9a-záàâãéêíóôõúç]+", flags=re.IGNORECASE)
_TARGET_LIKE_NAMES = frozenset(
    {
        "classe",
        "class",
        "label",
        "labels",
        "target_encoded",
        "target_code",
        "nivel",
        "nível",
        "complexidade",
        "complexidade_cognitiva",
        "categoria",
    }
)
_REQUIRED_SPLITS = ("train", "val", "test")
_SPLIT_PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))
_MAX_NEAR_DUPLICATE_COMPARISONS = 300_000
_MAX_NEAR_DUPLICATE_RECORDS = 1_000


class LeakageError(RuntimeError):
    """Erro gerado quando o relatório encontra vazamento crítico"""


# Resultado serializável das verificações de vazamento
@dataclass(frozen=True)
class LeakageCheckResult:

    report_json_path: Path
    report_markdown_path: Path
    near_duplicate_pairs_path: Path
    report: dict[str, Any]

    def to_dict(self, project_root: Path | None = None) -> dict[str, Any]:
        if project_root is None:
            return {
                "report_json_path": self.report_json_path.as_posix(),
                "report_markdown_path": self.report_markdown_path.as_posix(),
                "near_duplicate_pairs_path": self.near_duplicate_pairs_path.as_posix(),
                "report": self.report,
            }

        return {
            "report_json_path": relative_to_root(self.report_json_path, project_root),
            "report_markdown_path": relative_to_root(
                self.report_markdown_path, project_root
            ),
            "near_duplicate_pairs_path": relative_to_root(
                self.near_duplicate_pairs_path, project_root
            ),
            "report": self.report,
        }


# Executa as verificações de vazamento entre splits preparados e persiste relatórios
def run_leakage_checks(
    settings: AppSettings,
    *,
    metrics_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> LeakageCheckResult:
    metrics_dir = metrics_dir or settings.outputs.metrics_dir
    reports_dir = reports_dir or settings.outputs.reports_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not settings.leakage_checks.enabled:
        report = _build_disabled_report(settings)
        result = _persist_report(report, pd.DataFrame(), metrics_dir, reports_dir)
        return result

    splits = load_prepared_splits(settings)
    report, near_pairs = detect_leakage(splits, settings)
    result = _persist_report(report, near_pairs, metrics_dir, reports_dir)

    if report["summary"]["critical_findings"] > 0 and settings.leakage_checks.fail_on_critical_leakage:
        raise LeakageError(
            "Vazamento crítico detectado entre splits. Consulte "
            f"{result.report_json_path} e {result.report_markdown_path}."
        )

    return result


# Carrega train/val/test gerados por prepare_dataset.py
def load_prepared_splits(settings: AppSettings) -> dict[str, pd.DataFrame]:
    processed_dir = settings.outputs.processed_dir
    paths = {
        "train": processed_dir / "train.csv",
        "val": processed_dir / "val.csv",
        "test": processed_dir / "test.csv",
    }

    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise LeakageError(
            "Split(s) preparado(s) ausente(s) para verificação de vazamento: "
            + ", ".join(missing)
            + ". Execute primeiro scripts/prepare_dataset.py."
        )

    return {name: read_csv_dataset(path) for name, path in paths.items()}


# Aplica as verificações configuradas e retorna relatório + pares aproximados
def detect_leakage(
    splits: dict[str, pd.DataFrame],
    settings: AppSettings,
) -> tuple[dict[str, Any], pd.DataFrame]:
    _validate_split_mapping(splits)

    findings: list[dict[str, Any]] = []
    near_duplicate_rows: list[dict[str, Any]] = []
    cfg = settings.leakage_checks

    if cfg.check_target_like_columns:
        findings.extend(_find_target_like_columns(splits, settings))

    if cfg.check_id_overlap:
        findings.extend(_find_value_overlap(splits, settings.dataset.id_column, "id_overlap"))

    text_columns = [settings.dataset.text_column]
    clean_col = settings.preprocessing.clean_text_column
    if clean_col != settings.dataset.text_column:
        text_columns.append(clean_col)

    if cfg.check_exact_text_overlap:
        for column in text_columns:
            findings.extend(_find_value_overlap(splits, column, "exact_text_overlap"))

    if cfg.check_normalized_text_overlap:
        for column in text_columns:
            findings.extend(_find_normalized_text_overlap(splits, column))

    if cfg.check_near_duplicates:
        near_findings, near_duplicate_rows = _find_near_duplicates(
            splits,
            text_column=_preferred_existing_text_column(splits, settings),
            threshold=cfg.near_duplicate_threshold,
            id_column=settings.dataset.id_column,
            target_column=settings.dataset.target_column,
        )
        findings.extend(near_findings)

    severity_counts = _severity_counts(findings)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": {
            "name": settings.project.name,
            "version": settings.project.version,
            "random_state": settings.project.random_state,
        },
        "config": {
            "check_id_overlap": cfg.check_id_overlap,
            "check_exact_text_overlap": cfg.check_exact_text_overlap,
            "check_normalized_text_overlap": cfg.check_normalized_text_overlap,
            "check_near_duplicates": cfg.check_near_duplicates,
            "near_duplicate_threshold": cfg.near_duplicate_threshold,
            "check_target_like_columns": cfg.check_target_like_columns,
            "fail_on_critical_leakage": cfg.fail_on_critical_leakage,
        },
        "splits": _split_summary(splits, settings),
        "summary": {
            "total_findings": len(findings),
            "info_findings": severity_counts.get("info", 0),
            "warning_findings": severity_counts.get("warning", 0),
            "critical_findings": severity_counts.get("critical", 0),
            "status": "failed" if severity_counts.get("critical", 0) else "passed",
        },
        "findings": findings,
        "near_duplicate_pairs_preview": near_duplicate_rows[:20],
    }
    return report, pd.DataFrame(near_duplicate_rows)


# Normaliza texto para detectar duplicidades disfarçadas por caixa/pontuação/espaços
def normalize_text_for_leakage(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _NORMALIZE_SPACE_RE.sub(" ", text)
    return text.strip()


# Calcula similaridade Jaccard leve por n-gramas de caracteres
def character_ngram_jaccard(left: object, right: object, n: int = 5) -> float:
    left_ngrams = _character_ngrams(normalize_text_for_leakage(left), n=n)
    right_ngrams = _character_ngrams(normalize_text_for_leakage(right), n=n)
    if not left_ngrams and not right_ngrams:
        return 1.0
    if not left_ngrams or not right_ngrams:
        return 0.0
    intersection = len(left_ngrams.intersection(right_ngrams))
    union = len(left_ngrams.union(right_ngrams))
    return round(intersection / union, 6) if union else 0.0


def _validate_split_mapping(splits: dict[str, pd.DataFrame]) -> None:
    missing = [name for name in _REQUIRED_SPLITS if name not in splits]
    if missing:
        raise LeakageError("Split(s) ausente(s): " + ", ".join(missing) + ".")
    for name in _REQUIRED_SPLITS:
        if not isinstance(splits[name], pd.DataFrame):
            raise LeakageError(f"O split '{name}' deve ser um pandas.DataFrame.")
        if splits[name].empty:
            raise LeakageError(f"O split '{name}' está vazio.")


def _find_target_like_columns(
    splits: dict[str, pd.DataFrame], settings: AppSettings
) -> list[dict[str, Any]]:
    allowed = {settings.dataset.id_column, settings.dataset.text_column, settings.dataset.target_column}
    allowed.add(settings.preprocessing.clean_text_column)
    findings: list[dict[str, Any]] = []

    for split_name, df in splits.items():
        suspicious = []
        for column in df.columns:
            normalized_name = normalize_text_for_leakage(column).replace(" ", "_")
            if column in allowed:
                continue
            if normalized_name in _TARGET_LIKE_NAMES or (
                "target" in normalized_name and normalized_name != settings.dataset.target_column
            ):
                suspicious.append(str(column))

        if suspicious:
            findings.append(
                _finding(
                    severity="warning",
                    check="target_like_columns",
                    message=(
                        f"O split '{split_name}' possui coluna(s) suspeita(s) que podem derivar do rótulo: "
                        + ", ".join(suspicious[:10])
                        + "."
                    ),
                    details={"split": split_name, "columns": suspicious},
                )
            )
    return findings


def _find_value_overlap(
    splits: dict[str, pd.DataFrame], column: str, check_name: str
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for left_name, right_name in _SPLIT_PAIRS:
        left_df = splits[left_name]
        right_df = splits[right_name]
        if column not in left_df.columns or column not in right_df.columns:
            continue

        left_values = _series_as_clean_strings(left_df[column])
        right_values = _series_as_clean_strings(right_df[column])
        overlap = sorted(set(left_values).intersection(set(right_values)))
        overlap = [value for value in overlap if value]
        if not overlap:
            continue

        severity = "critical" if check_name in {"id_overlap", "exact_text_overlap"} else "warning"
        findings.append(
            _finding(
                severity=severity,
                check=check_name,
                message=(
                    f"Sobreposição em '{column}' entre '{left_name}' e '{right_name}': "
                    f"{len(overlap)} valor(es)."
                ),
                details={
                    "left_split": left_name,
                    "right_split": right_name,
                    "column": column,
                    "overlap_count": len(overlap),
                    "examples": overlap[:10],
                },
            )
        )
    return findings


def _find_normalized_text_overlap(
    splits: dict[str, pd.DataFrame], column: str
) -> list[dict[str, Any]]:
    normalized_splits: dict[str, pd.Series] = {}
    for split_name, df in splits.items():
        if column in df.columns:
            normalized_splits[split_name] = df[column].map(normalize_text_for_leakage)

    findings: list[dict[str, Any]] = []
    for left_name, right_name in _SPLIT_PAIRS:
        if left_name not in normalized_splits or right_name not in normalized_splits:
            continue
        overlap = sorted(
            set(normalized_splits[left_name]).intersection(set(normalized_splits[right_name]))
        )
        overlap = [value for value in overlap if value]
        if overlap:
            findings.append(
                _finding(
                    severity="critical",
                    check="normalized_text_overlap",
                    message=(
                        f"Sobreposição textual normalizada em '{column}' entre '{left_name}' e '{right_name}': "
                        f"{len(overlap)} valor(es)."
                    ),
                    details={
                        "left_split": left_name,
                        "right_split": right_name,
                        "column": column,
                        "overlap_count": len(overlap),
                        "examples": overlap[:10],
                    },
                )
            )
    return findings


def _find_near_duplicates(
    splits: dict[str, pd.DataFrame],
    *,
    text_column: str,
    threshold: float,
    id_column: str,
    target_column: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    skipped_pairs: list[dict[str, Any]] = []

    prepared: dict[str, list[dict[str, Any]]] = {}
    for split_name, df in splits.items():
        if text_column not in df.columns:
            continue
        split_rows = []
        for index, row in df.iterrows():
            text = row[text_column]
            normalized = normalize_text_for_leakage(text)
            if not normalized:
                continue
            ngrams = _character_ngrams(normalized, n=5)
            split_rows.append(
                {
                    "split": split_name,
                    "row_index": int(index),
                    "id": str(row.get(id_column, "")),
                    "target": str(row.get(target_column, "")),
                    "text": str(text),
                    "normalized_text": normalized,
                    "normalized_len": len(normalized),
                    "ngrams": ngrams,
                }
            )
        prepared[split_name] = split_rows

    for left_name, right_name in _SPLIT_PAIRS:
        left_rows = prepared.get(left_name, [])
        right_rows = prepared.get(right_name, [])
        if not left_rows or not right_rows:
            continue

        if len(left_rows) * len(right_rows) > _MAX_NEAR_DUPLICATE_COMPARISONS:
            vectorized_rows = _find_near_duplicates_vectorized(
                left_rows, right_rows, left_name, right_name, threshold
            )
            rows.extend(vectorized_rows)
            continue

        right_index = _build_ngram_candidate_index(right_rows)
        comparisons = 0
        pair_limit_reached = False

        for left in left_rows:
            candidate_indices = _candidate_indices_for_left_row(left, right_index, right_rows)
            for candidate_index in candidate_indices:
                if comparisons >= _MAX_NEAR_DUPLICATE_COMPARISONS:
                    pair_limit_reached = True
                    break
                comparisons += 1
                right = right_rows[candidate_index]
                similarity = _jaccard_sets(left["ngrams"], right["ngrams"])
                if similarity >= threshold:
                    rows.append(
                        {
                            "left_split": left_name,
                            "right_split": right_name,
                            "left_id": left["id"],
                            "right_id": right["id"],
                            "left_target": left["target"],
                            "right_target": right["target"],
                            "similarity": round(float(similarity), 6),
                            "left_text_preview": left["text"][:180],
                            "right_text_preview": right["text"][:180],
                        }
                    )
                    if len(rows) >= _MAX_NEAR_DUPLICATE_RECORDS:
                        pair_limit_reached = True
                        break
            if pair_limit_reached:
                skipped_pairs.append(
                    {
                        "left_split": left_name,
                        "right_split": right_name,
                        "reason": "comparison_or_record_limit_reached",
                        "comparisons": comparisons,
                        "max_comparisons": _MAX_NEAR_DUPLICATE_COMPARISONS,
                        "max_records": _MAX_NEAR_DUPLICATE_RECORDS,
                    }
                )
                break

    if rows:
        findings.append(
            _finding(
                severity="warning",
                check="near_duplicate_text_overlap",
                message=(
                    f"Foram encontrados {len(rows)} par(es) de textos quase duplicados "
                    f"entre splits usando threshold={threshold}."
                ),
                details={
                    "text_column": text_column,
                    "threshold": threshold,
                    "pair_count": len(rows),
                    "examples": rows[:10],
                },
            )
        )

    if skipped_pairs:
        findings.append(
            _finding(
                severity="warning",
                check="near_duplicate_scan_limited",
                message=(
                    "A busca de duplicidades aproximadas atingiu limite de segurança "
                    "para evitar custo quadrático excessivo."
                ),
                details={"limited_pairs": skipped_pairs},
            )
        )

    return findings, rows



def _find_near_duplicates_vectorized(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    left_name: str,
    right_name: str,
    threshold: float,
) -> list[dict[str, Any]]:
    left_texts = [str(row.get("normalized_text", "")) for row in left_rows]
    right_texts = [str(row.get("normalized_text", "")) for row in right_rows]
    if not left_texts or not right_texts:
        return []

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(5, 5),
        lowercase=False,
        min_df=1,
    )
    matrix = vectorizer.fit_transform([*left_texts, *right_texts])
    left_matrix = matrix[: len(left_texts)]
    right_matrix = matrix[len(left_texts) :]

    neighbors = NearestNeighbors(
        radius=max(0.0, 1.0 - threshold),
        metric="cosine",
        algorithm="brute",
        n_jobs=1,
    )
    neighbors.fit(right_matrix)
    distances, indices = neighbors.radius_neighbors(left_matrix, sort_results=True)

    rows: list[dict[str, Any]] = []
    for left_index, (distance_values, index_values) in enumerate(zip(distances, indices, strict=False)):
        left = left_rows[left_index]
        for distance, right_index in zip(distance_values, index_values, strict=False):
            similarity = 1.0 - float(distance)
            if similarity < threshold:
                continue
            right = right_rows[int(right_index)]
            rows.append(
                {
                    "left_split": left_name,
                    "right_split": right_name,
                    "left_id": left["id"],
                    "right_id": right["id"],
                    "left_target": left["target"],
                    "right_target": right["target"],
                    "similarity": round(float(similarity), 6),
                    "left_text_preview": left["text"][:180],
                    "right_text_preview": right["text"][:180],
                }
            )
            if len(rows) >= _MAX_NEAR_DUPLICATE_RECORDS:
                return rows
    return rows

def _build_ngram_candidate_index(rows: list[dict[str, Any]]) -> dict[str, set[int]]:
    index: dict[str, set[int]] = {}
    for row_index, row in enumerate(rows):
        for ngram in row["ngrams"]:
            index.setdefault(ngram, set()).add(row_index)
    return index


def _candidate_indices_for_left_row(
    left: dict[str, Any],
    right_index: dict[str, set[int]],
    right_rows: list[dict[str, Any]],
) -> list[int]:
    candidates: set[int] = set()
    for ngram in left["ngrams"]:
        candidates.update(right_index.get(ngram, set()))

    left_len = max(int(left.get("normalized_len", 0)), 1)
    filtered = []
    for candidate in candidates:
        right_len = max(int(right_rows[candidate].get("normalized_len", 0)), 1)
        length_ratio = min(left_len, right_len) / max(left_len, right_len)
        if length_ratio >= 0.70:
            filtered.append(candidate)

    return sorted(filtered)

def _preferred_existing_text_column(
    splits: dict[str, pd.DataFrame], settings: AppSettings
) -> str:
    clean_col = settings.preprocessing.clean_text_column
    if all(clean_col in df.columns for df in splits.values()):
        return clean_col
    return settings.dataset.text_column


def _persist_report(
    report: dict[str, Any],
    near_pairs: pd.DataFrame,
    metrics_dir: Path,
    reports_dir: Path,
) -> LeakageCheckResult:
    report_json_path = write_json(report, metrics_dir / "leakage_report.json")
    near_pairs_path = metrics_dir / "near_duplicate_pairs.csv"
    near_pairs.to_csv(near_pairs_path, index=False, encoding="utf-8")
    report_md_path = reports_dir / "leakage_report.md"
    report_md_path.write_text(_build_markdown_report(report), encoding="utf-8")
    return LeakageCheckResult(report_json_path, report_md_path, near_pairs_path, report)


def _build_markdown_report(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Relatório de vazamento de dados",
        "",
        f"Gerado em UTC: `{report.get('generated_at_utc', '')}`",
        "",
        "## Resumo",
        "",
        f"- Status: `{summary.get('status', 'unknown')}`",
        f"- Achados críticos: {summary.get('critical_findings', 0)}",
        f"- Alertas: {summary.get('warning_findings', 0)}",
        f"- Informativos: {summary.get('info_findings', 0)}",
        "",
        "## Achados",
        "",
    ]
    findings = report.get("findings", [])
    if not findings:
        lines.append("Nenhum vazamento crítico ou alerta foi detectado.")
    for idx, finding in enumerate(findings, start=1):
        lines.extend(
            [
                f"### {idx}. `{finding.get('severity')}` - `{finding.get('check')}`",
                "",
                str(finding.get("message", "")),
                "",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _build_disabled_report(settings: AppSettings) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": {"name": settings.project.name, "version": settings.project.version},
        "summary": {
            "total_findings": 0,
            "info_findings": 1,
            "warning_findings": 0,
            "critical_findings": 0,
            "status": "disabled",
        },
        "findings": [
            _finding(
                severity="info",
                check="leakage_checks_disabled",
                message="Verificações de vazamento desabilitadas por configuração.",
                details={},
            )
        ],
    }


def _split_summary(
    splits: dict[str, pd.DataFrame], settings: AppSettings
) -> dict[str, Any]:
    target_column = settings.dataset.target_column
    summary = {}
    for split_name, df in splits.items():
        distribution = {}
        if target_column in df.columns:
            distribution = {
                str(key): int(value)
                for key, value in df[target_column]
                .astype(str)
                .value_counts(dropna=False)
                .sort_index()
                .items()
            }
        summary[split_name] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "class_distribution": distribution,
        }
    return summary


def _finding(
    *, severity: str, check: str, message: str, details: dict[str, Any]
) -> dict[str, Any]:
    return {
        "severity": severity,
        "check": check,
        "message": message,
        "details": details,
    }


def _series_as_clean_strings(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").astype(str).str.strip()


def _character_ngrams(text: str, n: int) -> set[str]:
    if not text:
        return set()
    compact = f"  {text}  "
    if len(compact) <= n:
        return {compact}
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def _jaccard_sets(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = left.union(right)
    if not union:
        return 0.0
    return len(left.intersection(right)) / len(union)


def _severity_counts(findings: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"info": 0, "warning": 0, "critical": 0}
    for finding in findings:
        severity = str(finding.get("severity", "info"))
        counts[severity] = counts.get(severity, 0) + 1
    return counts


__all__ = [
    "LeakageCheckResult",
    "LeakageError",
    "character_ngram_jaccard",
    "detect_leakage",
    "load_prepared_splits",
    "normalize_text_for_leakage",
    "run_leakage_checks",
]
