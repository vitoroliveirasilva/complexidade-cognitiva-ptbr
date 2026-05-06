from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="data/raw/dataset_candidate.csv")
    parser.add_argument("--output", default="data/raw/dataset.csv")
    parser.add_argument("--backup-existing", action="store_true")
    args = parser.parse_args()

    candidate = Path(args.candidate)
    output = Path(args.output)
    if not candidate.exists():
        raise SystemExit(f"Candidato não encontrado: {candidate}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.backup_existing and output.exists():
        backup = output.with_name(
            f"{output.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{output.suffix}"
        )
        shutil.copy2(output, backup)
        print(f"Backup criado: {backup}")
    shutil.copy2(candidate, output)
    print(f"Dataset promovido: {candidate} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
