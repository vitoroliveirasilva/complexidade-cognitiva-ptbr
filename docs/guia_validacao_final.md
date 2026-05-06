# Guia de validação final do pipeline

Este guia resume como validar o projeto depois da implementação dos cinco blocos de melhoria.

## 1. Preparar ambiente

```bash
python -m venv .venv
.venv\Scripts\activate # Windows
# source .venv/bin/activate  # Linux/macOS
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 2. Validar configuração e estrutura

```bash
python scripts/run_pipeline.py --bootstrap-only
```

Esse comando valida configuração, logging, diretórios gerenciados e criação do contexto versionado, sem treinar modelo.

## 3. Rodar pipeline completo

```bash
python scripts/run_pipeline.py
```

O pipeline executa:

1. preparação do dataset;
2. verificação de vazamento;
3. extração de features;
4. validação cruzada;
5. busca de hiperparâmetros, quando habilitada;
6. treino e seleção do melhor modelo;
7. avaliação final;
8. explicabilidade;
9. relatórios visuais;
10. atualização de `outputs/latest/run_id.txt`.

## 4. Validar artefatos gerados

```bash
python scripts/validate_artifacts.py --profile complete
```

O script valida a última execução registrada em `outputs/latest/run_id.txt`.

## 5. Testar inferência local

```bash
python scripts/predict_text.py --text "Texto literário de exemplo para classificação."
```

Para arquivo CSV:

```bash
python scripts/predict_file.py \
  --input data/raw/exemplo_textos.csv \
  --output outputs/latest/predictions/exemplo_predicoes.csv
```

## 6. Rodar testes e lint

```bash
python -m pytest
ruff check .
```

Para cobertura local:

```bash
python -m pytest \
  --cov=complexidade_cognitiva_ptbr \
  --cov-report=term-missing
```

## 7. Smoke test sem dataset oficial

Use este fluxo quando não quiser expor ou depender do `data/raw/dataset.csv` oficial:

```bash
rm -rf outputs/ci_smoke
python scripts/run_pipeline.py --config configs/ci_smoke_config.yaml
python scripts/validate_artifacts.py --config configs/ci_smoke_config.yaml --profile smoke
python scripts/predict_text.py \
  --config configs/ci_smoke_config.yaml \
  --text "Texto literário de exemplo para classificação."
```

Esse é o fluxo usado no GitHub Actions.
