# Metodologia Experimental

## Visão geral

Este documento descreve a metodologia experimental adotada no projeto `complexidade-cognitiva-ptbr`. O objetivo é documentar o protocolo usado para preparar dados, extrair atributos, treinar modelos, selecionar o melhor experimento e avaliar o desempenho final.

## Objetivo experimental

Avaliar se representações textuais e métricas linguísticas permitem classificar automaticamente textos em três níveis de complexidade cognitiva:

- `baixa`;
- `media`;
- `alta`.

## Arquitetura do pipeline

A execução experimental foi organizada em quatro etapas:

1. **Preparação dos dados:** leitura do CSV, validação estrutural, limpeza textual e divisão em treino, validação e teste.
2. **Extração de features:** cálculo de métricas linguísticas interpretáveis para cada texto.
3. **Treinamento e seleção:** comparação de modelos e representações usando o conjunto de validação.
4. **Avaliação final:** aplicação do melhor modelo ao conjunto de teste e geração de métricas e relatórios.

A execução completa é realizada com:

```bash
python scripts/run_pipeline.py
```

A validação estrutural isolada é realizada com:

```bash
python scripts/run_pipeline.py --bootstrap-only
```

## Configuração experimental

A configuração central do experimento fica em `configs/config.yaml`. Ela controla:

- caminho do dataset bruto;
- nomes das colunas obrigatórias;
- proporções de treino, validação e teste;
- uso de estratificação;
- opções de limpeza textual;
- parâmetros de métricas linguísticas;
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

A execução oficial processou 9000 registros. A etapa de preparação validou a estrutura mínima do arquivo, a ausência de nulos, a ausência de campos obrigatórios vazios, a unicidade de identificadores e a integridade dos textos após limpeza.

A limpeza textual preservou o texto original e criou a coluna `texto_limpo`.

## Divisão dos dados

| Subconjunto | Proporção | Quantidade | `alta` | `baixa` | `media` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Treino | 70% | 6300 | 2100 | 2100 | 2100 |
| Validação | 15% | 1350 | 450 | 450 | 450 |
| Teste | 15% | 1350 | 450 | 450 | 450 |

A divisão usou `random_state = 42` e estratificação por classe.

## Métricas linguísticas interpretáveis

Foram extraídas 17 métricas linguísticas:

1. `num_caracteres`: Quantidade total de caracteres do texto limpo.
2. `num_palavras`: Quantidade total de tokens lexicais identificados no texto.
3. `num_sentencas`: Quantidade aproximada de sentenças identificadas por pontuação final.
4. `media_palavras_por_sentenca`: Média de palavras por sentença.
5. `media_caracteres_por_palavra`: Média de caracteres por palavra.
6. `maior_sentenca_palavras`: Comprimento da maior sentença em número de palavras.
7. `variancia_tamanho_sentencas`: Variância do número de palavras por sentença.
8. `palavras_unicas`: Quantidade de palavras únicas em caixa baixa.
9. `type_token_ratio`: Razão entre palavras únicas e total de palavras.
10. `diversidade_lexical`: Medida de diversidade lexical equivalente ao type-token ratio nesta versão.
11. `razao_palavras_longas`: Proporção de palavras com tamanho maior ou igual ao limite configurado.
12. `razao_palavras_repetidas`: Proporção de tokens que aparecem mais de uma vez no texto.
13. `razao_pontuacao`: Proporção de caracteres de pontuação em relação ao total de caracteres.
14. `razao_numeros`: Proporção de tokens numéricos em relação ao total de palavras/tokens.
15. `frequencia_conectivos`: Proporção de conectivos simples em relação ao total de palavras.
16. `frequencia_marcadores_subordinacao`: Proporção de marcadores de subordinação em relação ao total de palavras.
17. `densidade_lexical_aproximada`: Proporção aproximada de palavras de conteúdo em relação ao total de palavras.

Essas métricas aproximam aspectos de extensão, densidade lexical, estrutura sentencial, repetição, pontuação, conectivos e subordinação.

## Representações avaliadas

| Representação | Descrição |
| --- | --- |
| `linguistic_metrics` | Usa apenas métricas linguísticas interpretáveis. |
| `tfidf_word` | Usa TF-IDF baseado em palavras. |
| `tfidf_char` | Usa TF-IDF baseado em caracteres. |
| `tfidf_word_plus_linguistic_metrics` | Combina TF-IDF de palavras com métricas linguísticas. |

## Modelos avaliados

| Modelo | Identificador |
| --- | --- |
| Regressão Logística | `logistic_regression` |
| Linear SVM | `linear_svm` |
| Random Forest | `random_forest` |

## Métrica de seleção

A métrica de seleção foi `f1_macro`, adequada para problemas multiclasse porque calcula o desempenho médio entre classes. Como o dataset é balanceado, essa métrica também ajuda a verificar se o desempenho foi consistente entre as três classes.

## Experimento selecionado

| Campo | Valor |
| --- | --- |
| Experimento | `tfidf_word__logistic_regression` |
| Representação | `tfidf_word` |
| Modelo | `logistic_regression` |
| Métrica de seleção | `f1_macro` |
| Pontuação em validação | `1.000000` |

## Avaliação final

Após a seleção, o melhor modelo foi aplicado ao conjunto de teste, composto por 1350 registros balanceados.

Os artefatos finais foram salvos em `docs/resultados/` como cópia controlada dos arquivos gerados automaticamente pelo pipeline.

## Reprodutibilidade

Para reproduzir a execução:

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

Como a base usada nesta etapa é sintética, curada ou controlada, os resultados devem ser interpretados como validação operacional do pipeline. A generalização para textos literários reais exige validação posterior com corpus externo e critérios de rotulagem independentes.
