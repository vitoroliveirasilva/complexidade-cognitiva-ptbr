# Documentação acadêmica do projeto

Esta pasta reúne a documentação acadêmica e os artefatos oficiais gerados pelo pipeline do projeto `complexidade-cognitiva-ptbr`.

A estrutura foi organizada para apoiar a escrita do TCC, especialmente as seções de metodologia, resultados, discussão, limitações e trabalhos futuros.

## Arquivos documentais

| Arquivo | Finalidade |
| --- | --- |
| `dataset_card.md` | Descreve composição, uso recomendado, validações e limitações do dataset. |
| `metodologia_experimental.md` | Documenta o protocolo experimental adotado no pipeline. |
| `analise_resultados.md` | Interpreta os resultados obtidos na execução oficial. |
| `explicabilidade.md` | Explica o papel das métricas linguísticas e das representações TF-IDF. |
| `limitacoes_e_validade.md` | Discute validade interna, validade externa e limitações metodológicas. |
| `discussao_resultados_tcc.md` | Texto-base para adaptação no capítulo de Resultados e Discussão. |
| `guia_uso_no_tcc.md` | Indica como aproveitar os artefatos no texto acadêmico. |

## Artefatos oficiais em `docs/resultados/`

Os arquivos em `docs/resultados/` são uma cópia controlada dos artefatos gerados automaticamente pelo pipeline.

| Arquivo | Origem no pipeline | Finalidade |
| --- | --- | --- |
| `preparation_report.json` | `data/processed/preparation_report.json` | Relatório de validação, limpeza e divisão do dataset. |
| `features_metadata.json` | `data/processed/features/features_metadata.json` | Metadados das métricas linguísticas extraídas. |
| `best_experiment.json` | `outputs/models/best_experiment.json` | Registro do melhor experimento selecionado em validação. |
| `experiment_results.csv` | `outputs/models/experiment_results.csv` | Tabela comparativa dos experimentos treinados. |
| `final_metrics.json` | `outputs/metrics/final_metrics.json` | Métricas finais no conjunto de teste. |
| `classification_report.json` | `outputs/metrics/classification_report.json` | Relatório de classificação em formato JSON. |
| `classification_report.txt` | `outputs/metrics/classification_report.txt` | Relatório de classificação textual. |
| `test_predictions.csv` | `outputs/metrics/test_predictions.csv` | Predições individuais no conjunto de teste. |
| `confusion_matrix.png` | `outputs/figures/confusion_matrix.png` | Matriz de confusão da avaliação final. |
| `final_report.md` | `outputs/reports/final_report.md` | Relatório final gerado pelo pipeline. |

## Observação importante

Esta pasta não inclui tabelas derivadas, gráficos extras ou arquivos criados manualmente. A ideia é manter `docs/resultados/` fiel aos artefatos oficiais produzidos por `python scripts/run_pipeline.py`.

Os arquivos Markdown da raiz de `docs/` são documentos acadêmicos auxiliares escritos a partir desses artefatos.
