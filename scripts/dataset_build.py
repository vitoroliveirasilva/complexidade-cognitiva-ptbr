from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)?|\d+", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")

PORTUGUESE_STOPWORDS = {
    "a",
    "o",
    "as",
    "os",
    "um",
    "uma",
    "uns",
    "umas",
    "de",
    "do",
    "da",
    "dos",
    "das",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "por",
    "para",
    "com",
    "sem",
    "que",
    "se",
    "e",
    "ou",
    "mas",
    "como",
    "ao",
    "aos",
    "à",
    "às",
    "era",
    "foi",
    "ser",
    "ter",
    "havia",
    "não",
    "mais",
    "menos",
    "muito",
    "também",
    "lhe",
    "seu",
    "sua",
    "seus",
    "suas",
    "ele",
    "ela",
}
SUBORDINATION = {
    "porque",
    "embora",
    "quando",
    "enquanto",
    "conforme",
    "caso",
    "se",
    "como",
    "pois",
    "portanto",
    "todavia",
    "entretanto",
    "contudo",
    "ainda",
    "apesar",
    "segundo",
    "logo",
    "desde",
    "até",
    "quando",
    "onde",
    "cujo",
    "cuja",
    "quais",
    "qual",
    "quanto",
    "que",
}
DISCOURSE = {
    "assim",
    "portanto",
    "logo",
    "porém",
    "contudo",
    "entretanto",
    "todavia",
    "além",
    "desse",
    "modo",
    "por isso",
    "depois",
    "antes",
    "durante",
    "enfim",
    "então",
    "finalmente",
    "também",
}
ABSTRACT_MARKERS = {
    "memória",
    "lembrança",
    "tempo",
    "destino",
    "consciência",
    "culpa",
    "desejo",
    "moral",
    "sentido",
    "interpretação",
    "linguagem",
    "identidade",
    "solidão",
    "angústia",
    "imaginação",
    "julgamento",
    "hipótese",
    "abstração",
    "pensamento",
    "verdade",
    "aparência",
    "realidade",
}
AMBIGUITY_MARKERS = {
    "ambíguo",
    "ambígua",
    "incerto",
    "incerta",
    "silêncio",
    "sugerido",
    "implícito",
    "implícita",
    "metáfora",
    "símbolo",
    "simbólico",
    "perspectiva",
    "enunciação",
    "fragmento",
    "lacuna",
    "ironia",
    "paradoxo",
    "contradição",
    "deslocamento",
    "reconstruir",
    "inferência",
}
TEMPORAL_MARKERS = {
    "ontem",
    "hoje",
    "amanhã",
    "passado",
    "presente",
    "futuro",
    "outrora",
    "agora",
    "antes",
    "depois",
    "tempo",
    "memória",
    "lembrança",
    "recordação",
    "anos",
    "dias",
    "noite",
    "tarde",
}
EDITORIAL_PATTERNS = [
    "project gutenberg",
    "produced by",
    "online distributed proofreading",
    "start of the project",
    "end of the project",
    "ebook",
    "license",
    "copyright",
    "transcriber",
    "nota de editor",
    "índice",
    "indice",
    "sumario",
    "sumário",
    "bibliotheca",
    "typographia",
    "livreiro-editor",
]


@dataclass(frozen=True)
class SourceMeta:
    slug: str
    title: str
    author: str
    work: str
    source: str
    url: str
    genre: str


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_sources(path: Path) -> dict[str, SourceMeta]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        row["slug"]: SourceMeta(
            slug=row["slug"],
            title=row.get("title", row["slug"]),
            author=row.get("author", ""),
            work=row.get("work", row.get("title", row["slug"])),
            source=row.get("source", ""),
            url=row.get("url", ""),
            genre=row.get("genre", ""),
        )
        for row in rows
    }


def strip_boilerplate(text: str) -> str:
    start_patterns = [
        r"\*\*\* START OF (?:THE )?PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*",
    ]
    end_patterns = [
        r"\*\*\* END OF (?:THE )?PROJECT GUTENBERG EBOOK.*",
        r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK.*",
    ]
    for pattern in start_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = text[match.end() :]
            break
    for pattern in end_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = text[: match.start()]
            break
    return text


def normalize_spaces(text: str) -> str:
    text = text.replace("\ufeff", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_editorial_lines(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            lines.append("")
            continue
        normalized = strip_accents(clean.lower())
        if any(pattern in normalized for pattern in EDITORIAL_PATTERNS):
            continue
        if re.fullmatch(r"[\W\d_]{1,20}", clean):
            continue
        lines.append(clean)
    return "\n".join(lines)


def strip_accents(text: str) -> str:
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def normalized_words(text: str) -> list[str]:
    return [
        strip_accents(w.lower())
        for w in words(text)
        if re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", w)
    ]


def sentence_split(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE_RE.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(ch.isalpha() for ch in text)
    return alpha / max(1, len(text))


def stopword_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if token in PORTUGUESE_STOPWORDS) / len(tokens)


def paragraphs_from_text(text: str) -> list[str]:
    raw = re.split(r"\n\s*\n", text)
    paragraphs: list[str] = []
    buffer: list[str] = []
    for p in raw:
        p = re.sub(r"\s+", " ", p.strip())
        if not p:
            continue
        if len(words(p)) < 25:
            buffer.append(p)
            continue
        if buffer:
            p = " ".join(buffer + [p])
            buffer = []
        paragraphs.append(p)
    if buffer:
        joined = " ".join(buffer)
        if len(words(joined)) >= 25:
            paragraphs.append(joined)
    return paragraphs


# Mantém pontuação junto ao token original usando split simples por espaço para reconstrução aceitável
def build_windows(
    paragraphs: list[str],
    min_words: int,
    target_words: int,
    max_words: int,
    overlap_words: int,
) -> list[str]:
    all_tokens: list[str] = []
    for p in paragraphs:
        all_tokens.extend(p.split())
    windows: list[str] = []
    step = max(20, target_words - overlap_words)
    i = 0
    while i < len(all_tokens):
        chunk = all_tokens[i : i + target_words]
        if len(chunk) < min_words:
            break
        if len(chunk) > max_words:
            chunk = chunk[:max_words]
        text = " ".join(chunk).strip()
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        windows.append(text)
        i += step
    return windows


def has_editorial_noise(text: str) -> bool:
    norm = strip_accents(text.lower())
    return any(pattern in norm for pattern in EDITORIAL_PATTERNS)


def length_bin(n_words: int, bins: list[list[int]]) -> str:
    for lo, hi in bins:
        if lo <= n_words <= hi:
            return f"{lo}-{hi}"
    return "out_of_range"


def count_any(tokens: list[str], lexicon: set[str]) -> int:
    return sum(
        1 for token in tokens if token in {strip_accents(x.lower()) for x in lexicon}
    )


def features_for(text: str) -> dict[str, float]:
    toks = normalized_words(text)
    sents = sentence_split(text)
    n_words = len(toks)
    unique = len(set(toks))
    sent_lengths = [len(normalized_words(s)) for s in sents]
    avg_sent = sum(sent_lengths) / max(1, len(sent_lengths))
    long_ratio = sum(1 for t in toks if len(t) >= 8) / max(1, n_words)
    ttr = unique / max(1, n_words)
    sub = count_any(toks, SUBORDINATION) / max(1, n_words)
    disc = count_any(toks, DISCOURSE) / max(1, n_words)
    abstract = count_any(toks, ABSTRACT_MARKERS) / max(1, n_words)
    ambiguity = count_any(toks, AMBIGUITY_MARKERS) / max(1, n_words)
    temporal = count_any(toks, TEMPORAL_MARKERS) / max(1, n_words)
    punct = sum(1 for ch in text if ch in ",;:!?—-()") / max(1, len(text))
    dialogue = 1.0 if re.search(r"(^|\s)[—-]\s*[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ]", text) else 0.0
    return {
        "num_palavras": float(n_words),
        "num_sentencas": float(len(sents)),
        "media_palavras_por_sentenca": float(avg_sent),
        "type_token_ratio": float(ttr),
        "razao_palavras_longas": float(long_ratio),
        "frequencia_subordinacao": float(sub),
        "frequencia_marcadores_discursivos": float(disc),
        "densidade_pontuacao": float(punct),
        "frequencia_abstracao": float(abstract),
        "frequencia_ambiguidade": float(ambiguity),
        "frequencia_temporalidade": float(temporal),
        "dialogo_detectado": float(dialogue),
    }


def complexity_score(feat: dict[str, float], weights: dict[str, float]) -> float:
    return (
        weights.get("avg_sentence_length", 0.18)
        * min(feat["media_palavras_por_sentenca"] / 35.0, 1.5)
        + weights.get("long_word_ratio", 0.12)
        * min(feat["razao_palavras_longas"] / 0.35, 1.5)
        + weights.get("lexical_diversity", 0.10)
        * min(feat["type_token_ratio"] / 0.85, 1.3)
        + weights.get("subordination_ratio", 0.13)
        * min(feat["frequencia_subordinacao"] / 0.12, 1.5)
        + weights.get("discourse_marker_ratio", 0.10)
        * min(feat["frequencia_marcadores_discursivos"] / 0.08, 1.5)
        + weights.get("punctuation_density", 0.07)
        * min(feat["densidade_pontuacao"] / 0.05, 1.5)
        + weights.get("abstract_marker_ratio", 0.14)
        * min(feat["frequencia_abstracao"] / 0.07, 1.8)
        + weights.get("ambiguity_marker_ratio", 0.08)
        * min(feat["frequencia_ambiguidade"] / 0.04, 1.8)
        + weights.get("temporal_marker_ratio", 0.04)
        * min(feat["frequencia_temporalidade"] / 0.06, 1.5)
        + weights.get("dialogue_penalty", -0.04) * feat["dialogo_detectado"]
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[int(pos)]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def stable_hash(text: str) -> str:
    norm = strip_accents(text.lower())
    norm = re.sub(r"\W+", " ", norm).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def clean_text_for_csv(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dataset_card(df: pd.DataFrame, config: dict[str, Any]) -> str:
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Dataset Card - complexidade-cognitiva-ptbr",
        "",
        f"- Gerado em UTC: `{now}`",
        f"- Total de registros: `{len(df)}`",
        "- Fonte: obras literárias em português de domínio público, baixadas principalmente do Project Gutenberg.",
        "- Uso: treinamento e avaliação local do pipeline acadêmico de classificação de complexidade cognitiva.",
        "",
        "## Colunas",
        "",
        "| coluna | descrição |",
        "| --- | --- |",
        "| id | identificador estável do trecho |",
        "| texto | trecho textual usado pelo modelo |",
        "| target | classe operacional: baixa, media ou alta |",
        "| fonte | acervo de origem |",
        "| autor | autor da obra |",
        "| obra | título da obra |",
        "| capitulo | marcador aproximado da janela textual |",
        "| tipo_trecho | tipo operacional do trecho |",
        "| criterio_rotulo | resumo do critério usado para atribuição do rótulo |",
        "| origem_url | URL da fonte textual |",
        "| janela_inicio | índice aproximado inicial da janela na obra |",
        "| janela_fim | índice aproximado final da janela na obra |",
        "",
        "## Distribuição por classe",
        "",
    ]
    counts = df["target"].value_counts().sort_index()
    for label, count in counts.items():
        lines.append(f"- `{label}`: {count}")
    lines.extend(
        [
            "",
            "## Observações metodológicas",
            "",
            "Os rótulos são operacionais e heurísticos. Eles não substituem avaliação humana especializada.",
            "A rotulagem usa escores linguísticos e tercis dentro de faixas de tamanho para reduzir o risco de o modelo aprender apenas comprimento textual.",
            "Antes de usar o dataset como versão oficial, execute a auditoria e revise os alertas gerados.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument(
        "--no-write-main",
        action="store_true",
        help="Gera apenas dataset_candidate.csv, sem copiar para data/raw/dataset.csv.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    random.seed(int(config["project"].get("random_state", 42)))
    sources = read_sources(Path(config["paths"]["sources_csv"]))
    text_dir = Path(config["paths"]["raw_text_dir"])

    min_words = int(config["segmentation"]["min_words"])
    target_words = int(config["segmentation"]["target_words"])
    max_words = int(config["segmentation"]["max_words"])
    overlap_words = int(config["segmentation"]["overlap_words"])
    min_sentences = int(config["segmentation"]["min_sentences"])
    max_samples_per_work = int(config["segmentation"]["max_samples_per_work"])
    max_samples_per_work_per_class = int(
        config["segmentation"]["max_samples_per_work_per_class"]
    )
    bins = config["labeling"]["length_bins"]
    labels = config["labeling"]["labels"]
    weights = config["labeling"]["score_weights"]

    candidates: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for slug, meta in sources.items():
        path = text_dir / f"{slug}.txt"
        if not path.exists():
            print(f"AVISO: texto não encontrado para {slug}: {path}")
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        text = strip_boilerplate(raw)
        text = remove_editorial_lines(text)
        text = normalize_spaces(text)
        paragraphs = paragraphs_from_text(text)
        windows = build_windows(
            paragraphs, min_words, target_words, max_words, overlap_words
        )
        per_work_count = 0
        for idx, window in enumerate(windows):
            window = clean_text_for_csv(window)
            toks = normalized_words(window)
            if not (min_words <= len(toks) <= max_words):
                continue
            if len(sentence_split(window)) < min_sentences:
                continue
            if alpha_ratio(window) < float(config["text_cleaning"]["min_alpha_ratio"]):
                continue
            if stopword_ratio(toks) < float(
                config["text_cleaning"]["min_portuguese_stopword_ratio"]
            ):
                continue
            if config["segmentation"].get(
                "drop_windows_with_editorial_noise", True
            ) and has_editorial_noise(window):
                continue
            h = stable_hash(window)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            feat = features_for(window)
            n_words = int(feat["num_palavras"])
            b = length_bin(n_words, bins)
            if b == "out_of_range":
                continue
            candidates.append(
                {
                    "id_base": f"{slug}_{idx:05d}_{h}",
                    "texto": window,
                    "fonte": meta.source,
                    "autor": meta.author,
                    "obra": meta.work,
                    "capitulo": f"janela_{idx:05d}",
                    "tipo_trecho": "janela_textual_dominio_publico",
                    "origem_url": meta.url,
                    "janela_inicio": idx,
                    "janela_fim": idx + target_words,
                    "length_bin": b,
                    "score": complexity_score(feat, weights),
                    **feat,
                }
            )
            per_work_count += 1
            if per_work_count >= max_samples_per_work * 2:
                break

    if not candidates:
        raise SystemExit(
            "Nenhum candidato gerado. Rode primeiro dataset_download_sources.py."
        )

    # Rotula por tercis dentro de cada faixa de tamanho
    by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_bin[row["length_bin"]].append(row)

    labeled: list[dict[str, Any]] = []
    for b, rows in by_bin.items():
        scores = [float(r["score"]) for r in rows]
        q1 = percentile(scores, 1 / 3)
        q2 = percentile(scores, 2 / 3)
        for row in rows:
            score = float(row["score"])
            if score <= q1:
                target = labels["low"]
            elif score <= q2:
                target = labels["medium"]
            else:
                target = labels["high"]
            row["target"] = target
            row["criterio_rotulo"] = (
                f"Rotulagem heurística por escore linguístico dentro da faixa {b}; "
                f"tercis q1={q1:.4f}, q2={q2:.4f}, score={score:.4f}."
            )
            labeled.append(row)

    # Limita excesso por obra/classe para evitar domínio de uma única obra
    rng = random.Random(int(config["project"].get("random_state", 42)))
    rng.shuffle(labeled)
    counts_work_class: Counter[tuple[str, str]] = Counter()
    counts_work_total: Counter[str] = Counter()
    capped: list[dict[str, Any]] = []
    for row in labeled:
        key = (row["obra"], row["target"])
        if counts_work_class[key] >= max_samples_per_work_per_class:
            continue
        if counts_work_total[row["obra"]] >= max_samples_per_work:
            continue
        counts_work_class[key] += 1
        counts_work_total[row["obra"]] += 1
        capped.append(row)

    # Balanceia classes e bins
    target_per_class = int(config["labeling"]["target_per_class"])
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in capped:
        groups[(row["target"], row["length_bin"])].append(row)
    for rows in groups.values():
        rng.shuffle(rows)

    selected: list[dict[str, Any]] = []
    class_counts_available = {
        label: sum(1 for r in capped if r["target"] == label)
        for label in labels.values()
    }
    n_per_class = min(target_per_class, min(class_counts_available.values()))
    
    # Distribui alvo por bins de forma aproximada
    bins_labels = [f"{lo}-{hi}" for lo, hi in bins]
    per_bin_target = max(1, n_per_class // len(bins_labels))
    for label in [labels["low"], labels["medium"], labels["high"]]:
        chosen_for_label: list[dict[str, Any]] = []
        for b in bins_labels:
            chosen_for_label.extend(groups[(label, b)][:per_bin_target])
        if len(chosen_for_label) < n_per_class:
            remaining = [
                r for r in capped if r["target"] == label and r not in chosen_for_label
            ]
            rng.shuffle(remaining)
            chosen_for_label.extend(remaining[: n_per_class - len(chosen_for_label)])
        selected.extend(chosen_for_label[:n_per_class])

    rng.shuffle(selected)
    output_rows: list[dict[str, Any]] = []
    for i, row in enumerate(selected, start=1):
        output_rows.append(
            {
                "id": f"{i:06d}",
                "texto": row["texto"],
                "target": row["target"],
                "fonte": row["fonte"],
                "autor": row["autor"],
                "obra": row["obra"],
                "capitulo": row["capitulo"],
                "tipo_trecho": row["tipo_trecho"],
                "criterio_rotulo": row["criterio_rotulo"],
                "origem_url": row["origem_url"],
                "janela_inicio": row["janela_inicio"],
                "janela_fim": row["janela_fim"],
            }
        )

    df = pd.DataFrame(output_rows)
    candidate_path = Path(config["paths"]["dataset_candidate_path"])
    main_path = Path(config["paths"]["dataset_main_path"])
    card_path = Path(config["paths"]["dataset_card_path"])
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(candidate_path, index=False, encoding="utf-8")
    if not args.no_write_main:
        df[
            [
                "id",
                "texto",
                "target",
                "fonte",
                "autor",
                "obra",
                "capitulo",
                "tipo_trecho",
                "criterio_rotulo",
                "origem_url",
                "janela_inicio",
                "janela_fim",
            ]
        ].to_csv(main_path, index=False, encoding="utf-8")
    card_path.write_text(dataset_card(df, config), encoding="utf-8")

    print(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_path": str(candidate_path),
                "main_path": str(main_path) if not args.no_write_main else None,
                "dataset_card_path": str(card_path),
                "rows": len(df),
                "class_distribution": df["target"].value_counts().to_dict(),
                "works": int(df["obra"].nunique()),
                "authors": int(df["autor"].nunique()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
