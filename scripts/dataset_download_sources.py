from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Configuração não encontrada: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def download_text(url: str, timeout: int) -> str:
    headers = {
        "User-Agent": "complexidade-cognitiva-ptbr-dataset/0.1 academic research"
    }
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()

    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def rows_from_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def unique_non_empty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = (value or "").strip()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def gutenberg_fallback_urls(source: dict[str, str]) -> list[str]:
    # Monta URLs alternativas para reduzir falhas temporárias do Project Gutenberg
    source_id = (source.get("source_id") or "").strip()
    candidates = [
        source.get("url", ""),
        source.get("text_url", ""),
    ]

    if source_id:
        candidates.extend(
            [
                f"https://www.gutenberg.org/cache/epub/{source_id}/pg{source_id}.txt",
                f"https://www.gutenberg.org/files/{source_id}/{source_id}-0.txt",
                f"https://www.gutenberg.org/files/{source_id}/{source_id}.txt",
                f"https://www.gutenberg.org/ebooks/{source_id}.txt.utf-8",
            ]
        )

    return unique_non_empty(candidates)


def download_with_retries(
    source: dict[str, str],
    *,
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> tuple[str, str, list[dict[str, str]]]:
    attempts: list[dict[str, str]] = []
    last_error = ""

    for url in gutenberg_fallback_urls(source):
        for attempt in range(1, retries + 1):
            try:
                text = download_text(url, timeout=timeout)
                attempts.append(
                    {
                        "url": url,
                        "attempt": str(attempt),
                        "status": "ok",
                        "error": "",
                    }
                )
                return text, url, attempts
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = str(exc)
                attempts.append(
                    {
                        "url": url,
                        "attempt": str(attempt),
                        "status": "error",
                        "error": last_error,
                    }
                )
                if attempt < retries:
                    time.sleep(retry_sleep)

    raise RuntimeError(last_error or "Falha desconhecida no download.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.6,
        help="Pausa entre obras para não agredir a fonte.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permite finalizar com código 0 mesmo se alguma obra falhar.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Não baixa novamente obras cujo .txt já existe localmente.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    sources_path = Path(config["paths"]["sources_csv"])
    out_dir = Path(config["paths"]["raw_text_dir"])
    manifest_path = Path(config["paths"]["manifest_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    sources = rows_from_csv(sources_path)
    results: list[dict[str, Any]] = []
    ok_count = 0

    for source in sources:
        slug = source["slug"].strip()
        output_path = out_dir / f"{slug}.txt"
        item: dict[str, Any] = {
            "slug": slug,
            "title": source.get("title", ""),
            "author": source.get("author", ""),
            "source_id": source.get("source_id", ""),
            "configured_url": source.get("url", ""),
            "output_path": str(output_path),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        if (
            args.skip_existing
            and output_path.exists()
            and output_path.stat().st_size > 0
        ):
            chars = len(output_path.read_text(encoding="utf-8", errors="replace"))
            item.update(
                {
                    "status": "skipped_existing",
                    "chars": chars,
                    "final_url": "",
                    "error": "",
                    "attempts": [],
                }
            )
            ok_count += 1
            print(
                f"OK {slug}: arquivo já existe ({chars:,} caracteres) -> {output_path}"
            )
            results.append(item)
            continue

        try:
            text, final_url, attempts = download_with_retries(
                source,
                timeout=args.timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
            output_path.write_text(text, encoding="utf-8")
            item.update(
                {
                    "status": "ok",
                    "chars": len(text),
                    "final_url": final_url,
                    "error": "",
                    "attempts": attempts,
                }
            )
            ok_count += 1
            print(f"OK {slug}: {len(text):,} caracteres -> {output_path}")
        except Exception as exc:
            item.update(
                {
                    "status": "error",
                    "chars": 0,
                    "final_url": "",
                    "error": str(exc),
                    "attempts": item.get("attempts", []),
                }
            )
            print(f"ERRO {slug}: {exc}", file=sys.stderr)

        results.append(item)
        time.sleep(args.sleep)

    error_count = len(sources) - ok_count
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources_total": len(sources),
        "downloads_ok": ok_count,
        "downloads_error": error_count,
        "items": results,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Manifesto salvo em {manifest_path}")
    print(f"Resumo: {ok_count}/{len(sources)} obras disponíveis.")

    if ok_count == 0:
        print(
            "Nenhum texto foi baixado. Verifique internet, conexão ou URLs.",
            file=sys.stderr,
        )
        return 2

    if error_count and not args.allow_partial:
        print(
            "Algumas obras falharam. Rode novamente ou use --allow-partial "
            "apenas se aceitar construir dataset incompleto.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
