# Melhorias do Bloco 4 - Explicabilidade e relatórios

Este bloco adiciona explicabilidade por coeficientes de TF-IDF, explicações locais, gráficos avançados e relatório final estendido ao pipeline do TCC.

## Principais entregas

- `src/complexidade_cognitiva_ptbr/evaluation/explainability.py`
  - extrai termos TF-IDF com maior peso por classe para modelos lineares;
  - suporta Regressão Logística e Linear SVM quando combinadas com TF-IDF;
  - registra estado `unsupported` sem quebrar o pipeline quando o melhor modelo não é explicável por coeficientes;
  - gera explicações locais amostrais com contribuição `tfidf * coeficiente`.

- `src/complexidade_cognitiva_ptbr/evaluation/visualization.py`
  - gera matriz de confusão absoluta;
  - gera matriz de confusão normalizada;
  - gera comparação de experimentos;
  - gera resumo visual de validação cruzada;
  - gera distribuição de features linguísticas por classe;
  - gera distribuição de confiança das predições.

- `src/complexidade_cognitiva_ptbr/evaluation/reporting.py`
  - integra os relatórios visuais na avaliação final;
  - executa explicabilidade durante a avaliação final;
  - gera `final_report_extended.md` com métricas finais, validação cruzada, leakage, hiperparâmetros, explicabilidade e gráficos.

- `scripts/explain_model.py`
  - permite gerar explicabilidade de forma isolada, sem reexecutar treino.

## Artefatos esperados

Após avaliação completa, a execução versionada pode gerar:

```text
outputs/runs/<run_id>/explainability/tfidf_top_terms_by_class.csv
outputs/runs/<run_id>/explainability/tfidf_top_terms_by_class.json
outputs/runs/<run_id>/explainability/tfidf_explainability_report.md
outputs/runs/<run_id>/explainability/local_explanations_sample.csv
outputs/runs/<run_id>/figures/confusion_matrix_absolute.png
outputs/runs/<run_id>/figures/confusion_matrix_normalized.png
outputs/runs/<run_id>/figures/experiment_comparison_f1_macro.png
outputs/runs/<run_id>/figures/cv_summary_f1_macro.png
outputs/runs/<run_id>/figures/feature_distribution_<feature>.png
outputs/runs/<run_id>/figures/top_tfidf_terms_<classe>.png
outputs/runs/<run_id>/figures/prediction_confidence_distribution.png
outputs/runs/<run_id>/reports/final_report.md
outputs/runs/<run_id>/reports/final_report_extended.md
```

## Como validar

```bash
python scripts/run_pipeline.py --bootstrap-only
python scripts/explain_model.py --help
python -m compileall -q src tests scripts
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

O limite de threads nos testes reduz variações de tempo em ambientes pequenos, especialmente em rotinas de `scikit-learn`, BLAS e geração de gráficos.

## Observação metodológica

A explicabilidade global é gerada apenas quando o modelo selecionado expõe coeficientes lineares e vocabulário TF-IDF. Quando o melhor modelo for baseado apenas em métricas linguísticas ou for um modelo não linear, o pipeline salva relatórios vazios com status controlado, sem interromper a execução.
