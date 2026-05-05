# Metodologia Experimental

## Visão geral

Este documento descreve a metodologia experimental adotada no projeto `complexidade-cognitiva-ptbr`, cujo objetivo é desenvolver e avaliar um pipeline de PLN e Aprendizado de Máquina para classificar textos literários em níveis de complexidade cognitiva.

A metodologia foi organizada para garantir rastreabilidade, reprodutibilidade e separação adequada entre preparação dos dados, extração de atributos, treinamento, seleção de modelo e avaliação final.

## Objetivo experimental

Avaliar se diferentes representações textuais permitem classificar automaticamente textos em três níveis de complexidade cognitiva:

- `baixa`;
- `media`;
- `alta`.

A abordagem compara métricas linguísticas interpretáveis, representações TF-IDF e combinações entre elas.

## Arquitetura do pipeline

O pipeline é composto por quatro etapas principais:

1. **Preparação dos dados**: leitura, validação, limpeza textual e divisão do dataset.
2. **Extração de features**: geração de métricas linguísticas interpretáveis para cada texto.
3. **Treinamento e seleção**: comparação de combinações entre representações e modelos supervisionados.
4. **Avaliação final**: aplicação do melhor modelo ao conjunto de teste e geração dos artefatos acadêmicos.

A execução completa pode ser realizada por:

```bash
python scripts/run_pipeline.py
```

A validação estrutural isolada pode ser realizada por:

```bash
python scripts/run_pipeline.py --bootstrap-only
```

## Configuração experimental

A configuração central fica em `configs/config.yaml`. Ela concentra os parâmetros do experimento, incluindo:

- caminho do dataset bruto;
- nomes das colunas obrigatórias;
- proporções de treino, validação e teste;
- uso de estratificação;
- coluna de texto limpo;
- parâmetros das métricas linguísticas;
- parâmetros de TF-IDF;
- modelos avaliados;
- representações avaliadas;
- métrica de seleção;
- diretórios de saída;
- configuração de logging.

## Preparação dos dados

O dataset bruto foi lido a partir de:

```text
data/raw/dataset.csv
```

A etapa de preparação validou:

- existência do arquivo;
- formato CSV;
- presença das colunas `id`, `texto` e `target`;
- ausência de valores nulos;
- ausência de campos obrigatórios vazios;
- unicidade do identificador;
- existência de múltiplas classes;
- integridade dos textos após limpeza.

A limpeza textual foi controlada, preservando o texto original e criando a coluna `texto_limpo`. A divisão dos dados usou `random_state = 42` e estratificação.

## Divisão dos dados

| Subconjunto | Proporção | Quantidade | Classe `alta` | Classe `baixa` | Classe `media` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Treino | 70% | 6300 | 2100 | 2100 | 2100 |
| Validação | 15% | 1350 | 450 | 450 | 450 |
| Teste | 15% | 1350 | 450 | 450 | 450 |

A estratificação manteve o balanceamento entre as classes em todos os subconjuntos.

## Métricas linguísticas interpretáveis

Foram extraídas 17 métricas linguísticas aproximadas:

1. número de caracteres;
2. número de palavras;
3. número de sentenças;
4. média de palavras por sentença;
5. média de caracteres por palavra;
6. comprimento da maior sentença;
7. variância do tamanho das sentenças;
8. quantidade de palavras únicas;
9. type-token ratio;
10. diversidade lexical;
11. razão de palavras longas;
12. razão de palavras repetidas;
13. razão de pontuação;
14. razão de números;
15. frequência de conectivos;
16. frequência de marcadores de subordinação;
17. densidade lexical aproximada.

Essas métricas foram escolhidas por oferecerem sinais interpretáveis relacionados a extensão, densidade, diversidade lexical, complexidade sintática aproximada e estrutura superficial do texto.

## Representações avaliadas

| Representação | Descrição |
| --- | --- |
| `linguistic_metrics` | Usa apenas as métricas linguísticas interpretáveis |
| `tfidf_word` | Usa TF-IDF baseado em palavras |
| `tfidf_char` | Usa TF-IDF baseado em caracteres |
| `tfidf_word_plus_linguistic_metrics` | Combina TF-IDF de palavras com métricas linguísticas |

## Modelos avaliados

| Modelo | Identificador |
| --- | --- |
| Regressão Logística | `logistic_regression` |
| Linear SVM | `linear_svm` |
| Random Forest | `random_forest` |

## Métricas de avaliação

Foram calculadas as seguintes métricas:

- accuracy;
- precision macro;
- recall macro;
- F1 macro;
- F1 weighted;
- balanced accuracy.

A métrica de seleção configurada foi `f1_macro`, adequada para comparar desempenho médio entre classes sem favorecer diretamente a classe mais frequente.

## Critério de seleção do melhor experimento

O melhor experimento foi selecionado no conjunto de validação. O experimento escolhido foi:

| Campo | Valor |
| --- | --- |
| Experimento | `tfidf_word__logistic_regression` |
| Representação | `tfidf_word` |
| Modelo | `logistic_regression` |
| Métrica de seleção | `f1_macro` |
| Pontuação em validação | 1.000000 |

## Avaliação final

Após a seleção, o melhor modelo foi aplicado apenas ao conjunto de teste. O conjunto de teste possui 1350 registros, com 450 exemplos da classe `alta`, 450 da classe `baixa` e 450 da classe `media`.

Os resultados finais foram salvos em:

- `outputs/metrics/final_metrics.json`;
- `outputs/metrics/classification_report.json`;
- `outputs/metrics/classification_report.txt`;
- `outputs/metrics/test_predictions.csv`;
- `outputs/figures/confusion_matrix.png`;
- `outputs/reports/final_report.md`.

## Reprodutibilidade

Para reproduzir o experimento, executar:

```bash
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
python -m pytest -q
```

No Linux ou macOS, a ativação do ambiente virtual deve ser adaptada para:

```bash
source venv/bin/activate
```

## Observação metodológica

Como o dataset utilizado nesta execução é sintético, curado ou controlado, os resultados devem ser interpretados como validação do pipeline e da estratégia experimental. Para inferências mais amplas sobre textos literários reais, recomenda-se validação adicional com corpus externo, curadoria humana dos rótulos e avaliação cruzada em bases independentes.
