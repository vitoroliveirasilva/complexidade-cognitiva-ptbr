# Melhorias - Bloco 2: Segurança experimental

Este bloco adiciona verificações explícitas para reduzir risco de vazamento de dados e validação cruzada estratificada para tornar os resultados experimentais mais defensáveis.

## Detecção de vazamento

Módulo:

```text
src/complexidade_cognitiva_ptbr/data/leakage.py
```

O módulo verifica, entre os splits `train`, `val` e `test`:

- sobreposição de IDs;
- sobreposição textual exata;
- sobreposição textual normalizada, ignorando caixa, pontuação, acentos e espaços;
- duplicidade textual aproximada por similaridade de n-gramas de caracteres;
- colunas suspeitas que possam derivar diretamente do rótulo.

Artefatos gerados:

```text
outputs/runs/<run_id>/metrics/leakage_report.json
outputs/runs/<run_id>/metrics/near_duplicate_pairs.csv
outputs/runs/<run_id>/reports/leakage_report.md
```

Quando `leakage_checks.fail_on_critical_leakage` está ativo, achados críticos interrompem a execução depois de salvar o relatório.

## Validação cruzada

Módulo:

```text
src/complexidade_cognitiva_ptbr/evaluation/cross_validation.py
```

A validação cruzada é executada sobre `train + val`, preservando o conjunto `test` isolado para avaliação final. Cada fold monta um novo `Pipeline` scikit-learn, garantindo que TF-IDF, normalização e classificador sejam ajustados apenas no subconjunto de treino do fold.

Artefatos gerados:

```text
outputs/runs/<run_id>/metrics/cv_results.csv
outputs/runs/<run_id>/metrics/cv_summary.json
outputs/runs/<run_id>/reports/cv_report.md
outputs/runs/<run_id>/figures/cv_metrics_comparison.png
```

## Integração no pipeline

A execução completa agora segue esta ordem:

```text
prepare_dataset

detect_leakage

build_features

cross_validate

train_model

evaluate_model
```

---

## Comandos de validação

```bash
python scripts/run_pipeline.py --bootstrap-only
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m compileall -q src tests scripts
```

Também é possível validar os módulos diretamente após preparar dados e features:

```python
from complexidade_cognitiva_ptbr.config.settings import load_settings
from complexidade_cognitiva_ptbr.data.leakage import run_leakage_checks
from complexidade_cognitiva_ptbr.evaluation.cross_validation import run_cross_validation

settings = load_settings("configs/config.yaml")
run_leakage_checks(settings)
run_cross_validation(settings)
```

## Observações

- O `dataset.csv` bruto não é alterado.
- O teste final continua isolado.
- A busca aproximada de duplicidade possui limites de segurança para evitar custo quadrático excessivo em datasets maiores.
- Os testes usam corpus sintético pequeno e não dependem do dataset oficial.
