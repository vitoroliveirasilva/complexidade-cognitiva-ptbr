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
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def rows_from_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.4,
        help="Pausa entre downloads para não agredir a fonte.",
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
        url = source["url"].strip()
        output_path = out_dir / f"{slug}.txt"
        item = {
            "slug": slug,
            "title": source.get("title", ""),
            "author": source.get("author", ""),
            "url": url,
            "output_path": str(output_path),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            text = download_text(url, timeout=args.timeout)
            output_path.write_text(text, encoding="utf-8")
            item.update({"status": "ok", "chars": len(text), "error": ""})
            ok_count += 1
            print(f"OK {slug}: {len(text):,} caracteres -> {output_path}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            item.update({"status": "error", "chars": 0, "error": str(exc)})
            print(f"ERRO {slug}: {exc}", file=sys.stderr)
        results.append(item)
        time.sleep(args.sleep)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources_total": len(sources),
        "downloads_ok": ok_count,
        "downloads_error": len(sources) - ok_count,
        "items": results,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Manifesto salvo em {manifest_path}")

    if ok_count == 0:
        print(
            "Nenhum texto foi baixado. Verifique internet, conexão ou URLs.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
