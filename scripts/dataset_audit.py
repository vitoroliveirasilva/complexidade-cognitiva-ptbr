from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)?|\d+", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")
EDITORIAL_PATTERNS = [
    "project gutenberg",
    "produced by",
    "online distributed proofreading",
    "license",
    "ebook",
    "start of the project",
    "end of the project",
    "transcriber",
]


def strip_accents(text: str) -> str:
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", str(text))
        if unicodedata.category(ch) != "Mn"
    )


def normalize_text(text: str) -> str:
    text = strip_accents(str(text).lower())
    text = re.sub(r"\W+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return WORD_RE.findall(str(text))


def sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE_RE.split(str(text)) if p.strip()]
    return parts or ([str(text).strip()] if str(text).strip() else [])


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def length_bin(n: int) -> str:
    if n <= 40:
        return "<=40"
    if n <= 80:
        return "41-80"
    if n <= 120:
        return "81-120"
    if n <= 160:
        return "121-160"
    if n <= 220:
        return "161-220"
    return ">220"


def sentence_repetition(df: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for text in df["texto"].astype(str):
        for sentence in sentences(text):
            norm = normalize_text(sentence)
            if len(norm.split()) >= 4:
                counter[norm] += 1
    return [
        {"sentence_normalized": sentence, "count": count}
        for sentence, count in counter.most_common(limit)
        if count > 1
    ]


def metrics_by_class(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        toks = words(row["texto"])
        sents = sentences(row["texto"])
        rows.append(
            {
                "target": row["target"],
                "words": len(toks),
                "sentences": len(sents),
                "avg_words_sentence": len(toks) / max(1, len(sents)),
                "ttr": len(set(t.lower() for t in toks)) / max(1, len(toks)),
                "length_bin": length_bin(len(toks)),
            }
        )
    mdf = pd.DataFrame(rows)
    result: dict[str, dict[str, Any]] = {}
    for label, group in mdf.groupby("target"):
        result[str(label)] = {
            "rows": int(len(group)),
            "words_mean": round(float(group["words"].mean()), 4),
            "words_median": round(float(group["words"].median()), 4),
            "words_min": int(group["words"].min()),
            "words_max": int(group["words"].max()),
            "avg_words_sentence_mean": round(
                float(group["avg_words_sentence"].mean()), 4
            ),
            "type_token_ratio_mean": round(float(group["ttr"].mean()), 4),
        }
    return result


def crosstab_length(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    tmp = df.copy()
    tmp["_words"] = tmp["texto"].astype(str).map(lambda x: len(words(x)))
    tmp["_bin"] = tmp["_words"].map(length_bin)
    table = pd.crosstab(tmp["_bin"], tmp["target"])
    order = ["<=40", "41-80", "81-120", "121-160", "161-220", ">220"]
    labels = sorted(tmp["target"].astype(str).unique())
    out: dict[str, dict[str, int]] = {}
    for b in order:
        out[b] = {
            label: (
                int(table.loc[b, label])
                if b in table.index and label in table.columns
                else 0
            )
            for label in labels
        }
    return out


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Relatório de qualidade do dataset",
        "",
        f"- Gerado em UTC: `{report['generated_at_utc']}`",
        f"- Linhas: `{report['rows']}`",
        f"- Colunas: `{', '.join(report['columns'])}`",
        f"- Duplicatas exatas: `{report['exact_duplicate_texts']}`",
        f"- Duplicatas normalizadas: `{report['normalized_duplicate_texts']}`",
        f"- IDs duplicados: `{report['id_duplicates']}`",
        "",
        "## Distribuição por classe",
        "",
    ]
    for label, count in report["class_distribution"].items():
        lines.append(f"- `{label}`: {count}")
    lines += [
        "",
        "## Métricas por classe",
        "",
        "| classe | linhas | palavras média | mediana | min | max | média palavras/sentença | TTR médio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, data in report["metrics_by_class"].items():
        lines.append(
            f"| {label} | {data['rows']} | {data['words_mean']} | {data['words_median']} | "
            f"{data['words_min']} | {data['words_max']} | {data['avg_words_sentence_mean']} | {data['type_token_ratio_mean']} |"
        )
    lines += ["", "## Alertas", ""]
    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    else:
        lines.append("- Nenhum alerta crítico identificado.")
    lines += ["", "## Sentenças repetidas mais frequentes", ""]
    if report["repeated_sentences_preview"]:
        for item in report["repeated_sentences_preview"]:
            lines.append(
                f"- `{item['sentence_normalized']}`: {item['count']} ocorrências"
            )
    else:
        lines.append("- Nenhuma sentença repetida relevante encontrada.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--md", default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    input_path = Path(
        args.input
        or config.get("paths", {}).get("dataset_candidate_path", "data/raw/dataset.csv")
    )
    json_path = Path(
        args.json
        or config.get("paths", {}).get(
            "audit_json_path", "outputs/metrics/dataset_quality_report.json"
        )
    )
    md_path = Path(
        args.md
        or config.get("paths", {}).get(
            "audit_md_path", "outputs/reports/dataset_quality_report.md"
        )
    )
    audit_config = config.get("audit", {})

    df = pd.read_csv(input_path)
    required = {"id", "texto", "target"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Colunas obrigatórias ausentes: {missing}")

    norm_texts = df["texto"].astype(str).map(normalize_text)
    exact_dups = int(df["texto"].astype(str).duplicated().sum())
    norm_dups = int(norm_texts.duplicated().sum())
    id_dups = int(df["id"].astype(str).duplicated().sum())
    by_class = metrics_by_class(df)
    class_dist = {
        str(k): int(v) for k, v in df["target"].value_counts().sort_index().items()
    }

    warnings: list[str] = []
    if len(df) < int(audit_config.get("critical_min_total_rows", 10000)):
        warnings.append(f"Total de linhas abaixo da meta recomendada: {len(df)}.")
    min_rows = int(audit_config.get("critical_min_rows_per_class", 3000))
    for label, count in class_dist.items():
        if count < min_rows:
            warnings.append(
                f"Classe {label} abaixo da meta recomendada de {min_rows} registros: {count}."
            )
    means = [data["words_mean"] for data in by_class.values()]
    if means and min(means) > 0:
        gap_ratio = (max(means) - min(means)) / max(1.0, min(means))
        if gap_ratio > float(
            audit_config.get("critical_max_word_mean_gap_ratio", 0.10)
        ):
            warnings.append(
                f"Média de palavras por classe tem diferença relativa alta: {gap_ratio:.3f}."
            )
    if exact_dups or norm_dups or id_dups:
        warnings.append("Há duplicatas em id/texto/texto normalizado.")
    noise_count = int(
        df["texto"]
        .astype(str)
        .map(lambda x: any(p in normalize_text(x) for p in EDITORIAL_PATTERNS))
        .sum()
    )
    if noise_count:
        warnings.append(
            f"Há {noise_count} trechos com possível ruído editorial/licença."
        )
    if "autor" in df.columns:
        author_share = df["autor"].value_counts(normalize=True).max()
        if author_share > float(audit_config.get("warning_max_author_share", 0.45)):
            warnings.append(f"Um autor concentra {author_share:.1%} do dataset.")
    if "obra" in df.columns:
        work_share = df["obra"].value_counts(normalize=True).max()
        if work_share > float(audit_config.get("warning_max_work_share", 0.20)):
            warnings.append(f"Uma obra concentra {work_share:.1%} do dataset.")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "class_distribution": class_dist,
        "id_duplicates": id_dups,
        "exact_duplicate_texts": exact_dups,
        "normalized_duplicate_texts": norm_dups,
        "metrics_by_class": by_class,
        "length_bin_by_class": crosstab_length(df),
        "author_distribution": (
            df["autor"].value_counts().head(20).to_dict()
            if "autor" in df.columns
            else {}
        ),
        "work_distribution": (
            df["obra"].value_counts().head(20).to_dict() if "obra" in df.columns else {}
        ),
        "repeated_sentences_preview": sentence_repetition(
            df, int(audit_config.get("repeated_sentence_preview_limit", 30))
        ),
        "warnings": warnings,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
