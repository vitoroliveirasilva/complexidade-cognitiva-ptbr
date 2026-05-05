# Relatório final de avaliação

## Identificação

- Projeto: `complexidade-cognitiva-ptbr`
- Versão: `0.1.0`
- Gerado em UTC: `2026-05-05T02:45:19.931278+00:00`
- Seed configurada: `42`

## Experimento selecionado

- Experimento: `tfidf_word__logistic_regression`
- Representação: `tfidf_word`
- Modelo: `logistic_regression`
- Métrica de seleção: `f1_macro`
- Pontuação em validação: `1.0`

### Métricas em validação

| Métrica | Valor |
| --- | ---: |
| accuracy | 1.0 |
| precision_macro | 1.0 |
| recall_macro | 1.0 |
| f1_macro | 1.0 |
| f1_weighted | 1.0 |
| balanced_accuracy | 1.0 |

## Avaliação no conjunto de teste

- Total de registros avaliados: `1350`
- Classes avaliadas: `alta, baixa, media`

### Distribuição de classes no teste

| Classe | Quantidade |
| --- | ---: |
| alta | 450 |
| baixa | 450 |
| media | 450 |

### Métricas finais

| Métrica | Valor |
| --- | ---: |
| accuracy | 1.0 |
| precision_macro | 1.0 |
| recall_macro | 1.0 |
| f1_macro | 1.0 |
| f1_weighted | 1.0 |
| balanced_accuracy | 1.0 |

## Artefatos gerados

- `final_metrics`: `outputs/metrics/final_metrics.json`
- `classification_report_json`: `outputs/metrics/classification_report.json`
- `classification_report_txt`: `outputs/metrics/classification_report.txt`
- `test_predictions`: `outputs/metrics/test_predictions.csv`
- `confusion_matrix`: `outputs/figures/confusion_matrix.png`

## Observações para discussão acadêmica

A avaliação final utiliza o melhor experimento escolhido no conjunto de validação e aplica esse modelo ao conjunto de teste, preservando a separação entre treino, validação e teste.

As métricas macro ajudam a observar o comportamento médio entre classes, enquanto o F1 ponderado considera a distribuição real de exemplos por classe.

A matriz de confusão permite analisar visualmente quais classes foram mais confundidas pelo modelo, apoiando a discussão dos resultados no TCC.

As métricas linguísticas interpretáveis permanecem disponíveis nos arquivos de features e podem ser usadas para explicar parte do comportamento dos modelos treinados.

## Referência do relatório de treino

- Arquivo do melhor experimento: `outputs/models/best_experiment.json`
- Quantidade de features linguísticas consideradas no treino: `17`
