# Bloco 5 - Qualidade final

Este bloco fecha a evolução do pipeline de TCC com foco em qualidade de software, validação objetiva de artefatos, CI e documentação operacional.

## Itens entregues

- Workflow `.github/workflows/ci.yml` com lint, testes, cobertura e smoke test.
- Configuração `configs/ci_smoke_config.yaml` para executar o pipeline sem depender do `data/raw/dataset.csv` oficial.
- Fixture `tests/fixtures/smoke_dataset.csv` com corpus pequeno e balanceado para CI.
- Script `scripts/validate_artifacts.py` para validar artefatos de uma execução versionada.
- Módulo `utils/artifact_validation.py` com validação reutilizável por código e CLI.
- Testes de validação de artefatos.
- Configuração de cobertura em `pyproject.toml`.
- Ajustes de `.gitignore` para diretórios versionados/temporários gerados pelo pipeline.
- Documentação final de execução, validação e CI.

## Comandos finais recomendados

```bash
python scripts/run_pipeline.py --bootstrap-only
python scripts/run_pipeline.py
python scripts/validate_artifacts.py --profile complete
python scripts/predict_text.py --text "Texto literário de exemplo para classificação."
python scripts/predict_file.py --input data/raw/exemplo_textos.csv --output outputs/latest/predictions/exemplo_predicoes.csv
python -m pytest
ruff check .
```

## Comandos de smoke test sem dataset oficial

```bash
python scripts/run_pipeline.py --config configs/ci_smoke_config.yaml
python scripts/validate_artifacts.py --config configs/ci_smoke_config.yaml --profile smoke
python scripts/predict_text.py --config configs/ci_smoke_config.yaml --text "Texto literário de exemplo para classificação."
```

## Validação de artefatos

Por padrão, o script resolve a última execução pelo arquivo `outputs/latest/run_id.txt` e valida o diretório correspondente em `outputs/runs/<run_id>`.

Perfis disponíveis:

- `smoke`: valida o conjunto mínimo esperado no CI.
- `complete`: valida o conjunto final recomendado para o TCC, tratando gráficos e explicabilidade específicos como opcionais quando o modelo não suportar algum recurso.

Para tratar também os artefatos opcionais como obrigatórios:

```bash
python scripts/validate_artifacts.py --profile complete --strict-optional
```

## Critério de aceite final

A implementação completa deve ser considerada fechada quando os comandos finais rodarem sem falha e a execução gerar artefatos versionados em `outputs/runs/<run_id>/`, incluindo métricas, relatórios, gráficos, bundle de modelo e relatórios de segurança experimental.
