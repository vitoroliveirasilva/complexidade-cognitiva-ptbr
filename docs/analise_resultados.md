# Análise dos Resultados

## Síntese da execução

O pipeline foi executado integralmente, contemplando preparação dos dados, extração de features, treinamento, seleção do melhor experimento e avaliação final no conjunto de teste.

A execução validou um dataset com 9000 exemplos, distribuídos igualmente entre três classes de complexidade cognitiva. A divisão estratificada produziu 1350 registros para o conjunto de teste, com distribuição balanceada entre as classes.

## Qualidade e integridade dos dados

A etapa de preparação indicou que o dataset estava estruturalmente adequado para os experimentos:

- não foram encontrados valores nulos nas colunas obrigatórias;
- não foram identificados campos obrigatórios vazios;
- não houve duplicidade exata de textos;
- a distribuição de classes permaneceu balanceada;
- a limpeza textual não removeu nenhum texto por completo;
- a divisão estratificada foi aplicada com sucesso.

Esses pontos indicam que a base estava consistente para a validação inicial do pipeline.

## Extração de atributos linguísticos

Foram extraídas 17 métricas linguísticas interpretáveis para cada texto. Os arquivos de features foram gerados para treino, validação e teste sem valores ausentes.

As métricas contemplam aspectos superficiais e interpretáveis dos textos, como extensão, estrutura sentencial, diversidade lexical, pontuação, presença de números, conectivos, marcadores de subordinação e densidade lexical aproximada.

## Comparação dos experimentos

Foram avaliados 12 experimentos, combinando quatro representações textuais e três modelos supervisionados.

Tabela consolidada dos experimentos em validação:

| experiment_id | representation | model_name | validation_accuracy | validation_f1_macro | validation_f1_weighted | validation_balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| tfidf_char__linear_svm | tfidf_char | linear_svm | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| tfidf_char__logistic_regression | tfidf_char | logistic_regression | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| tfidf_char__random_forest | tfidf_char | random_forest | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| tfidf_word__linear_svm | tfidf_word | linear_svm | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| tfidf_word__logistic_regression | tfidf_word | logistic_regression | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| tfidf_word__random_forest | tfidf_word | random_forest | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| tfidf_word_plus_linguistic_metrics__linear_svm | tfidf_word_plus_linguistic_metrics | linear_svm | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| tfidf_word_plus_linguistic_metrics__logistic_regression | tfidf_word_plus_linguistic_metrics | logistic_regression | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| tfidf_word_plus_linguistic_metrics__random_forest | tfidf_word_plus_linguistic_metrics | random_forest | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| linguistic_metrics__linear_svm | linguistic_metrics | linear_svm | 0.996296 | 0.996296 | 0.996296 | 0.996296 |
| linguistic_metrics__logistic_regression | linguistic_metrics | logistic_regression | 0.994815 | 0.994815 | 0.994815 | 0.994815 |
| linguistic_metrics__random_forest | linguistic_metrics | random_forest | 0.992593 | 0.992592 | 0.992592 | 0.992593 |

Os experimentos com TF-IDF, tanto por palavras quanto por caracteres, obtiveram desempenho perfeito no conjunto de validação. A combinação entre TF-IDF de palavras e métricas linguísticas também atingiu valores máximos nas métricas avaliadas.

Os experimentos baseados apenas em métricas linguísticas interpretáveis também apresentaram desempenho muito alto, variando aproximadamente entre 0,9926 e 0,9963 em F1 macro. Esse resultado indica que as métricas linguísticas extraídas possuem forte poder discriminativo no dataset atual.

## Melhor experimento selecionado

O melhor experimento selecionado foi:

| Campo | Valor |
| --- | --- |
| Experimento | `tfidf_word__logistic_regression` |
| Representação | `tfidf_word` |
| Modelo | `logistic_regression` |
| Métrica de seleção | `f1_macro` |
| F1 macro em validação | 1.000000 |

A Regressão Logística com TF-IDF por palavras foi selecionada com F1 macro igual a 1,0 no conjunto de validação.

## Avaliação no conjunto de teste

O conjunto de teste possui 1350 registros, distribuídos da seguinte forma:

| Classe | Quantidade |
| --- | ---: |
| `alta` | 450 |
| `baixa` | 450 |
| `media` | 450 |

Métricas finais no conjunto de teste:

| Métrica | Valor |
| --- | ---: |
| Accuracy | 1.000000 |
| Precision macro | 1.000000 |
| Recall macro | 1.000000 |
| F1 macro | 1.000000 |
| F1 weighted | 1.000000 |
| Balanced accuracy | 1.000000 |

A distribuição das predições foi idêntica à distribuição real do conjunto de teste:

| Classe predita | Quantidade |
| --- | ---: |
| `alta` | 450 |
| `baixa` | 450 |
| `media` | 450 |

## Interpretação da matriz de confusão

A matriz de confusão apresentou 450 acertos para cada classe e nenhum erro entre classes. Isso significa que, no conjunto de teste desta execução, todos os exemplos foram classificados corretamente pelo modelo selecionado.

A diagonal principal da matriz concentra todos os registros avaliados:

| Classe real | Predita como `alta` | Predita como `baixa` | Predita como `media` |
| --- | ---: | ---: | ---: |
| `alta` | 450 | 0 | 0 |
| `baixa` | 0 | 450 | 0 |
| `media` | 0 | 0 | 450 |

## Discussão acadêmica dos resultados

Os resultados demonstram que o pipeline implementado é capaz de executar todo o ciclo experimental de forma consistente: preparação dos dados, extração de atributos, treinamento, comparação entre experimentos, seleção objetiva do melhor modelo e avaliação final.

O desempenho perfeito obtido por representações TF-IDF sugere que o dataset atual contém sinais lexicais muito claros entre as classes. Isso pode ocorrer em bases sintéticas, curadas ou construídas com padrões linguísticos bem separados por classe.

Esse comportamento não invalida o pipeline. Pelo contrário, confirma que a implementação está operacional e que as etapas do processo conseguem capturar padrões presentes nos dados. No entanto, para fins acadêmicos, a interpretação deve ser cuidadosa: os resultados indicam validade operacional do método no dataset avaliado, mas não comprovam generalização ampla para textos literários reais, heterogêneos e não controlados.

## Limitações

As principais limitações observadas são:

1. **Dataset sintético ou controlado:** a separação entre classes pode estar mais evidente do que em corpora literários reais.
2. **Métricas perfeitas:** resultados iguais a 1,0 exigem cautela interpretativa, pois podem indicar facilidade excessiva da tarefa experimental.
3. **Ausência de validação externa:** ainda não houve teste em corpus independente, produzido ou rotulado por outra fonte.
4. **Rótulos dependentes da estratégia de construção:** se os textos foram gerados ou curados com critérios muito explícitos, o modelo pode aprender padrões de geração além da complexidade cognitiva pretendida.
5. **Métricas linguísticas aproximadas:** as features são interpretáveis, mas não substituem análise linguística profunda com anotação sintática, semântica ou discursiva especializada.

## Pontos fortes

Apesar das limitações, a execução apresenta pontos relevantes para o TCC:

- pipeline completo e reprodutível;
- arquitetura modular organizada em `src layout`;
- validações de dados antes do treinamento;
- geração automática de relatórios e artefatos;
- comparação entre múltiplas representações e modelos;
- preservação de métricas interpretáveis;
- testes automatizados aprovados;
- resultados documentados para discussão acadêmica.
