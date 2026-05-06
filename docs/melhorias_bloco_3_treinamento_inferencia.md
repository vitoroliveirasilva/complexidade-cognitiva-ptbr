# Melhorias Bloco 3 - Treinamento avançado e inferência

## Objetivo

Este bloco adiciona busca de hiperparâmetros, seleção mais robusta do melhor experimento, persistência de um pacote autocontido de modelo e inferência local por texto ou arquivo CSV.

O `dataset.csv` permanece tratado como artefato bruto somente de leitura.

## Busca de hiperparâmetros

- Módulo `src/complexidade_cognitiva_ptbr/models/search.py`.
- Suporte a `GridSearchCV` e `RandomizedSearchCV`.
- Leitura dos espaços de busca em `configs/config.yaml`.
- Busca feita apenas sobre o conjunto de treino, com validação cruzada interna.
- Seleção por `hyperparameter_search.refit_metric`.
- Persistência de:
  - `hyperparameter_search_results.csv`;
  - `best_hyperparameters.json`;
  - `hyperparameter_search_report.md`.

## ModelBundle

- Módulo `src/complexidade_cognitiva_ptbr/models/bundle.py`.
- O bundle inclui:
  - pipeline sklearn treinado;
  - classes;
  - representação usada;
  - modelo usado;
  - features linguísticas;
  - snapshot essencial de configuração;
  - métricas de validação;
  - fingerprint do dataset;
  - metadados de inferência.

Artefatos gerados:

```text
outputs/runs/<run_id>/models/best_model_bundle.joblib
outputs/latest/models/best_model_bundle.joblib
```

## Inferência local

- Módulo `src/complexidade_cognitiva_ptbr/models/inference.py`.
- Script `scripts/predict_text.py`.
- Script `scripts/predict_file.py`.

A inferência reconstrói a entrada esperada pelo pipeline com:

- texto original;
- texto limpo;
- métricas linguísticas;
- colunas necessárias para `ColumnTransformer`.

## Como validar

```bash
python scripts/run_pipeline.py --bootstrap-only
python -m pytest -q
python -m compileall -q src tests scripts
```

Depois de treinar o projeto, validar inferência com:

```bash
python scripts/predict_text.py --text "Texto literário de exemplo para classificação."
python scripts/predict_file.py --input data/raw/exemplo_textos.csv --output outputs/latest/predictions/exemplo_predicoes.csv
```

## Observação metodológica

A busca de hiperparâmetros não usa o conjunto de teste. O teste final continua reservado para a avaliação final, após a escolha do melhor experimento.
