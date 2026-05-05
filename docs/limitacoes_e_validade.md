# Limitações e Validade Experimental

## Visão geral

A execução final do pipeline apresentou desempenho perfeito no conjunto de teste, com todas as métricas iguais a 1.0. A matriz de confusão indicou ausência de erros: todas as 450 amostras de cada classe foram classificadas corretamente.

Esse resultado demonstra que o pipeline está funcional e que o dataset atual possui sinais suficientes para separação das classes. Entretanto, métricas perfeitas precisam ser interpretadas com cautela.

## Validade interna

A validade interna é fortalecida por:

- separação explícita entre treino, validação e teste;
- uso de `random_state = 42`;
- estratificação dos subconjuntos;
- validação automática de colunas obrigatórias, nulos, vazios e duplicidades;
- geração de relatórios intermediários;
- testes automatizados executados com sucesso;
- versionamento do código e dos artefatos acadêmicos principais.

O conjunto de teste teve 1350 registros, distribuídos igualmente entre as classes.

## Validade externa

O pipeline foi validado funcionalmente e apresentou desempenho perfeito no dataset experimental utilizado, mas sua generalização para corpora literários reais depende de validação posterior com textos anotados por critérios externos ou especialistas.

## Validade de construto

O conceito investigado é complexidade cognitiva em textos literários. As métricas implementadas aproximam aspectos relacionados à complexidade textual, como extensão, densidade lexical, conectivos e subordinação.

Contudo, complexidade cognitiva também envolve fatores que não são totalmente capturados por métricas superficiais ou TF-IDF, como:

- ambiguidade interpretativa;
- intertextualidade;
- exigência inferencial;
- complexidade simbólica;
- conhecimento prévio do leitor;
- estrutura narrativa não linear;
- carga semântica e pragmática.

Assim, o modelo estima níveis de complexidade a partir de sinais linguísticos observáveis, mas não substitui avaliação humana especializada.

## Risco de separação artificial

O principal risco é que as classes do dataset estejam separadas de forma muito evidente. Nesse cenário, o modelo pode aprender padrões artificiais do corpus em vez de aprender uma noção mais ampla de complexidade cognitiva.

A acurácia perfeita, portanto, é positiva para validar a implementação, mas deve ser discutida como possível evidência de baixa dificuldade experimental.

## Limitações metodológicas atuais

1. Ausência de corpus literário real anotado por especialistas.
2. Ausência de validação externa em outro conjunto de dados.
3. Ausência de avaliação humana comparativa.
4. Classes balanceadas artificialmente.
5. Ausência de análise por gênero literário, autor, época ou estilo.
6. Ausência de explicação lexical por coeficientes do modelo TF-IDF selecionado.

## Estratégias de mitigação

- validar o pipeline com textos literários reais;
- usar anotação humana ou rubrica pedagógica para os rótulos;
- aplicar validação cruzada;
- avaliar generalização em corpus externo;
- gerar explicação por coeficientes do modelo;
- incluir embeddings semânticos;
- comparar com modelos de linguagem pré-treinados para português;
- documentar critérios de rotulagem com maior formalidade.

## Conclusão

A etapa atual é válida como demonstração experimental controlada de um pipeline de PLN e aprendizado de máquina para classificação automática de níveis de complexidade textual. A principal ressalva é que o resultado perfeito deve ser apresentado como evidência de funcionamento no conjunto experimental, não como prova definitiva de generalização.
