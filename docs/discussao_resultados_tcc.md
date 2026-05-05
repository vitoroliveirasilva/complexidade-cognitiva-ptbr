# Discussão dos Resultados para o TCC

## 1. Síntese da execução experimental

A execução completa do pipeline contemplou as etapas de preparação do dataset, extração de métricas linguísticas, treinamento de modelos supervisionados, seleção do melhor experimento e avaliação final no conjunto de teste.

O dataset possui 9000 exemplos distribuídos de forma balanceada entre três classes de complexidade cognitiva: baixa, média e alta. Após a validação inicial, não foram identificados valores nulos, campos obrigatórios vazios ou duplicidades textuais. A divisão foi estratificada, resultando em 6300 exemplos para treino, 1350 para validação e 1350 para teste.

## 2. Comparação dos experimentos

Foram avaliadas combinações entre representações textuais e modelos supervisionados. As representações incluíram métricas linguísticas interpretáveis, TF-IDF de palavras, TF-IDF de caracteres e uma representação combinada entre TF-IDF de palavras e métricas linguísticas.

O melhor experimento selecionado foi `tfidf_word__logistic_regression`, composto pela representação `tfidf_word` e pelo modelo `logistic_regression`. A seleção utilizou a métrica `f1_macro`, adequada para problemas multiclasse por considerar o desempenho médio entre classes.

## 3. Desempenho final no conjunto de teste

No conjunto de teste, o modelo selecionado obteve:

| Métrica | Valor |
| --- | ---: |
| Accuracy | 1.000000 |
| Precision macro | 1.000000 |
| Recall macro | 1.000000 |
| F1 macro | 1.000000 |
| F1 weighted | 1.000000 |
| Balanced accuracy | 1.000000 |

A matriz de confusão mostrou que todos os exemplos foram classificados corretamente, com 450 acertos em cada uma das três classes.

## 4. Interpretação das métricas perfeitas

Os resultados demonstram a viabilidade técnica do pipeline proposto e sua capacidade de aprender padrões associados às classes no conjunto experimental utilizado. Contudo, por se tratar de um dataset sintético ou curado e altamente separável, a generalização para textos literários reais deve ser investigada em estudos futuros.

## 5. Papel das métricas linguísticas

As métricas linguísticas interpretáveis tiveram papel importante na análise, ainda que o melhor desempenho tenha sido obtido com TF-IDF. Elas permitem observar indícios objetivos de complexidade textual, como extensão, quantidade de palavras, tamanho de sentença, diversidade lexical e proporção de palavras longas.

## 6. Papel do TF-IDF

O TF-IDF de palavras teve desempenho máximo com Regressão Logística. Esse resultado sugere que os padrões lexicais foram altamente discriminativos. Em textos literários, escolhas vocabulares podem refletir diferenças de complexidade, mas também podem introduzir dependência do corpus utilizado.

Por isso, o TF-IDF deve ser interpretado como representação eficiente para o conjunto experimental, mas não como explicação completa do fenômeno cognitivo estudado.

## 7. Contribuição para o TCC

Os resultados contribuem para o TCC em três dimensões:

1. **Dimensão técnica:** o pipeline foi implementado e executado integralmente.
2. **Dimensão experimental:** diferentes modelos e representações foram comparados.
3. **Dimensão interpretável:** métricas linguísticas permitiram discutir características textuais relacionadas às classes.

## 8. Limitações e trabalhos futuros

As principais limitações estão relacionadas à natureza sintética/curada do dataset, à ausência de corpus externo e à falta de validação profissional independente dos rótulos.

Trabalhos futuros devem incluir textos literários reais anotados por especialistas, análise dos coeficientes do modelo TF-IDF selecionado, validação cruzada, embeddings semânticos e avaliação por gênero literário.

## 9. Conclusão da discussão

A execução experimental confirmou que a arquitetura proposta é funcional, reprodutível e capaz de gerar artefatos úteis para análise acadêmica. O desempenho perfeito reforça a consistência interna do experimento, mas deve ser apresentado junto às limitações metodológicas para evitar conclusões excessivas.

Assim, o projeto deve ser compreendido como etapa inicial sólida para investigação automatizada da complexidade cognitiva textual, com potencial de expansão para corpora reais e estratégias interpretáveis mais avançadas.
