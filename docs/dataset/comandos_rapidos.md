# Comandos rápidos - Dataset

Este guia resume os comandos principais para gerar, auditar e usar o dataset.

## 1. Baixar as fontes

```powershell
python scripts/dataset_download_sources.py
```

## 2. Gerar o dataset

```powershell
python scripts/dataset_build.py
```

Esse comando gera:

```text
data/raw/dataset_candidate.csv
data/raw/dataset.csv
docs/dataset_card.md
```

## 3. Gerar apenas o candidato

Para criar o dataset candidato sem sobrescrever `data/raw/dataset.csv`:

```powershell
python scripts/dataset_build.py --no-write-main
```

Depois de auditar e aprovar o candidato:

```powershell
python scripts/dataset_promote.py --candidate data/raw/dataset_candidate.csv --output data/raw/dataset.csv --backup-existing
```

## 4. Auditar o dataset

```powershell
python scripts/dataset_audit.py
```

Relatórios gerados:

```text
outputs/metrics/dataset_quality_report.json
outputs/reports/dataset_quality_report.md
```

Para visualizar no PowerShell:

```powershell
Get-Content outputs/reports/dataset_quality_report.md
```

## 5. Rodar o pipeline completo

Depois de gerar e auditar o dataset:

```powershell
python scripts/run_pipeline.py
python scripts/validate_artifacts.py --profile complete
python scripts/predict_text.py --text "O texto combina memória, ambiguidade e símbolos que exigem reconstrução interpretativa."
python -m pytest
ruff check .
```

## 6. Conferir a última execução

```powershell
Get-Content outputs/latest/run_id.txt
```

Depois use o `run_id` para consultar artefatos em:

```text
outputs/runs/<run_id>/
```

## 7. Fluxo completo recomendado

```powershell
python scripts/dataset_download_sources.py
python scripts/dataset_build.py
python scripts/dataset_audit.py
python scripts/run_pipeline.py
python scripts/validate_artifacts.py --profile complete
python scripts/predict_text.py --text "O narrador reorganiza lembranças, símbolos e ambiguidades para construir uma interpretação instável dos acontecimentos."
python -m pytest
ruff check .
```

## Observação

Como o dataset final é gerado localmente, ele não é versionado.

## 8. Rodar avaliação complementar por agrupamento

```powershell
python scripts/group_cross_validation.py --config configs/config.yaml
```

Essa avaliação usa `GroupKFold` e tenta agrupar primeiro por `obra`, depois por colunas alternativas como `autor` ou `fonte`.

Relatórios gerados:

```text
outputs/metrics/group_cross_validation_results.json
outputs/reports/group_cross_validation_report.md
```
