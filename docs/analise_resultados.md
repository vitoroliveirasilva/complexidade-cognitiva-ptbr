# Análise dos Resultados

## Síntese da execução

O pipeline foi executado integralmente, contemplando preparação dos dados, extração de features, treinamento, seleção do melhor experimento e avaliação final no conjunto de teste.

A execução validou um dataset com 9000 exemplos, distribuídos igualmente entre três classes de complexidade cognitiva. A divisão estratificada produziu 1350 registros para o conjunto de teste, com distribuição balanceada entre `alta`, `baixa` e `media`.

## Qualidade e integridade dos dados

A etapa de preparação indicou que o dataset estava estruturalmente adequado:

- valores nulos nas colunas obrigatórias: 0;
- campos obrigatórios vazios: 0;
- duplicidade textual exata: 0;
- duplicidade textual normalizada: 0;
- textos vazios após limpeza: 0;
- avisos relevantes: 0.

Esses pontos indicam que a base estava consistente para a validação inicial do pipeline.

## Extração de atributos linguísticos

Foram extraídas 17 métricas linguísticas interpretáveis para cada texto. Os arquivos de features foram gerados para treino, validação e teste sem valores ausentes.

As métricas contemplam aspectos como extensão textual, estrutura sentencial, diversidade lexical, pontuação, presença de números, conectivos, marcadores de subordinação e densidade lexical aproximada.

## Comparação dos experimentos

Foram avaliados 12 experimentos, combinando quatro representações textuais e três modelos supervisionados.

| experiment_id | representation | model_name | validation_accuracy | validation_f1_macro | validation_f1_weighted | validation_balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| tfidf_word__logistic_regression | tfidf_word | logistic_regression | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| tfidf_word__linear_svm | tfidf_word | linear_svm | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| tfidf_word__random_forest | tfidf_word | random_forest | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| tfidf_char__logistic_regression | tfidf_char | logistic_regression | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| tfidf_char__linear_svm | tfidf_char | linear_svm | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| tfidf_char__random_forest | tfidf_char | random_forest | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| tfidf_word_plus_linguistic_metrics__logistic_regression | tfidf_word_plus_linguistic_metrics | logistic_regression | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| tfidf_word_plus_linguistic_metrics__linear_svm | tfidf_word_plus_linguistic_metrics | linear_svm | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| tfidf_word_plus_linguistic_metrics__random_forest | tfidf_word_plus_linguistic_metrics | random_forest | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| linguistic_metrics__linear_svm | linguistic_metrics | linear_svm | 0.996296 | 0.996296 | 0.996296 | 0.996296 |
| linguistic_metrics__logistic_regression | linguistic_metrics | logistic_regression | 0.994815 | 0.994815 | 0.994815 | 0.994815 |
| linguistic_metrics__random_forest | linguistic_metrics | random_forest | 0.992593 | 0.992592 | 0.992592 | 0.992593 |

Os experimentos com TF-IDF, tanto por palavras quanto por caracteres, obtiveram desempenho perfeito no conjunto de validação. A combinação entre TF-IDF de palavras e métricas linguísticas também atingiu valores máximos nas métricas avaliadas.

Os experimentos baseados apenas em métricas linguísticas interpretáveis também apresentaram desempenho muito alto, variando aproximadamente entre 0,9926 e 0,9963 em F1 macro. Esse resultado indica que as métricas extraídas possuem forte poder discriminativo no dataset atual.

## Melhor experimento selecionado

| Campo | Valor |
| --- | --- |
| Experimento | `tfidf_word__logistic_regression` |
| Representação | `tfidf_word` |
| Modelo | `logistic_regression` |
| Métrica de seleção | `f1_macro` |
| F1 macro em validação | `1.000000` |

## Avaliação no conjunto de teste

O conjunto de teste possui 1350 registros:

| Classe | Quantidade |
| --- | ---: |
| `alta` | 450 |
| `baixa` | 450 |
| `media` | 450 |

Métricas finais:

| Métrica | Valor |
| --- | ---: |
| Accuracy | 1.000000 |
| Precision macro | 1.000000 |
| Recall macro | 1.000000 |
| F1 macro | 1.000000 |
| F1 weighted | 1.000000 |
| Balanced accuracy | 1.000000 |

## Interpretação da matriz de confusão

A matriz de confusão apresentou 450 acertos para cada classe e nenhum erro entre classes.

| Classe real | Predita como `alta` | Predita como `baixa` | Predita como `media` |
| --- | ---: | ---: | ---: |
| `alta` | 450 | 0 | 0 |
| `baixa` | 0 | 450 | 0 |
| `media` | 0 | 0 | 450 |

## Discussão acadêmica

Os resultados demonstram que o pipeline implementado é capaz de executar todo o ciclo experimental de forma consistente. O desempenho perfeito obtido por representações TF-IDF sugere que o dataset atual contém sinais lexicais muito claros entre as classes.

Esse comportamento não invalida o pipeline. Ele confirma que a implementação está operacional e que as etapas conseguem capturar padrões presentes nos dados. Entretanto, a interpretação deve ser cautelosa: os resultados indicam validade operacional no dataset avaliado, mas não comprovam generalização ampla para textos literários reais e heterogêneos.

## Pontos fortes

- pipeline completo e reprodutível;
- arquitetura modular organizada em `src layout`;
- validações de dados antes do treinamento;
- comparação entre múltiplas representações e modelos;
- preservação de métricas interpretáveis;
- geração automática de artefatos acadêmicos;
- documentação final separada dos artefatos gerados automaticamente.
