# Comandos rápidos - Dataset

## 1. Baixar fontes

```powershell
python scripts/dataset_download_sources.py
```

## 2. Gerar dataset

```powershell
python scripts/dataset_build.py
```

## 3. Auditar dataset

```powershell
python scripts/dataset_audit.py
```

## 4. Rodar pipeline com o dataset

```powershell
python scripts/run_pipeline.py
python scripts/validate_artifacts.py --profile complete
python scripts/predict_text.py --text "O texto combina memória, ambiguidade e símbolos que exigem reconstrução interpretativa."
python -m pytest
ruff check .
```

## 5. Conferir relatórios principais

```powershell
Get-Content outputs/metrics/dataset_quality_report.md
Get-Content outputs/latest/run_id.txt
```

## 6. Se quiser gerar apenas candidato sem sobrescrever data/raw/dataset.csv

```powershell
python scripts/dataset_build.py --no-write-main
```

Depois de auditar:

```powershell
python scripts/dataset_promote.py --candidate data/raw/dataset_candidate.csv --output data/raw/dataset.csv --backup-existing
```
