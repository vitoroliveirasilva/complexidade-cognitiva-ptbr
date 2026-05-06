# Melhorias do pipeline - Bloco 1: base estrutural

Este bloco prepara o projeto para as próximas etapas de robustez experimental sem alterar o `data/raw/dataset.csv`.

## O que foi adicionado

- Expansão do `configs/config.yaml` com seções para validação cruzada, detecção de vazamento, busca de hiperparâmetros, explicabilidade, versionamento de execuções, inferência local e relatórios visuais.
- Validação cruzada de configuração em `src/complexidade_cognitiva_ptbr/config/validation.py`.
- Fingerprint SHA-256 do dataset bruto em `src/complexidade_cognitiva_ptbr/utils/hashing.py`.
- Relatório de ambiente em `src/complexidade_cognitiva_ptbr/utils/environment.py`.
- Contexto de execução versionada em `src/complexidade_cognitiva_ptbr/utils/run_context.py`.
- Integração do `RunContext` ao bootstrap do pipeline, criando `outputs/runs/<run_id>/` com metadados iniciais.
- Testes iniciais para configuração, hashing e versionamento.

## Artefatos criados no bootstrap

Ao executar o pipeline ou o bootstrap, são criados:

```text
outputs/runs/<run_id>/config_snapshot.yaml
outputs/runs/<run_id>/environment.json
outputs/runs/<run_id>/data_fingerprint.json
outputs/runs/<run_id>/execution_log.txt
outputs/runs/<run_id>/run_manifest.json
outputs/runs/<run_id>/models/
outputs/runs/<run_id>/metrics/
outputs/runs/<run_id>/figures/
outputs/runs/<run_id>/reports/
outputs/runs/<run_id>/explainability/
```

## Como validar este bloco

```bash
python scripts/run_pipeline.py --bootstrap-only
python -m pytest
```

Se o Ruff estiver instalado no ambiente:

```bash
ruff check .
```

## Observação metodológica

O fingerprint é calculado apenas por leitura. Ele registra caminho, existência, tamanho, data de modificação e SHA-256 do dataset bruto, mas não modifica nem regrava o arquivo oficial de entrada.
