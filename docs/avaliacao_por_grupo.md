# Avaliação complementar por agrupamento

A avaliação complementar por agrupamento foi adicionada para estimar a capacidade do modelo de generalizar para grupos textuais não vistos durante o treinamento de cada fold.

Essa etapa não substitui a avaliação tradicional do projeto. Ela funciona como uma camada metodológica adicional, especialmente importante porque o dataset é formado por múltiplos trechos extraídos de obras literárias. Sem controle por grupo, trechos da mesma obra ou do mesmo autor podem aparecer em treino e validação, o que pode elevar artificialmente as métricas caso o modelo aprenda padrões estilísticos específicos.

## Estratégia de agrupamento

A estratégia preferencial é usar a coluna `obra`, quando disponível e com ao menos dois grupos distintos. Se `obra` não estiver disponível, a rotina procura colunas alternativas na seguinte ordem configurável:

```text
obra
titulo
título
title
work
autor
author
fonte
source
```

O comportamento padrão é:

1. usar `obra`, se existir e for válida;
2. usar uma coluna alternativa, como `autor` ou `fonte`, se `obra` não for utilizável;
3. aplicar fallback sintético por linha, se nenhuma coluna de agrupamento existir.

O fallback sintético por linha permite que a rotina execute de forma segura em datasets pequenos ou fixtures de CI, mas não deve ser interpretado como avaliação de generalização por obra ou autor. Nesse caso, o relatório gerado registra explicitamente o modo degradado.

## Diferença para a avaliação tradicional

A validação tradicional estratificada divide os registros tentando preservar a distribuição das classes em cada fold. Ela é adequada para medir desempenho geral sob partições balanceadas.

A avaliação por agrupamento impõe uma restrição adicional: todos os exemplos de um mesmo grupo ficam exclusivamente no treino ou exclusivamente na validação dentro de cada fold. Assim, quando o agrupamento é feito por `obra`, o modelo é validado em obras não vistas naquela dobra.

Essa avaliação tende a ser mais exigente. Se as métricas por agrupamento forem menores do que as métricas tradicionais, isso pode indicar dependência de padrões de obra, autor, vocabulário ou estilo recorrente.

## Configuração

A configuração fica em `configs/config.yaml`:

```yaml
group_cross_validation:
  enabled: true
  preferred_group_column: "obra"
  fallback_group_columns:
    - "obra"
    - "titulo"
    - "título"
    - "title"
    - "work"
    - "autor"
    - "author"
    - "fonte"
    - "source"
  fallback_strategy: "row_id"
  n_splits: 5
  estimator: "logistic_regression"
  C: 1.0
  max_iter: 1000
  class_weight: "balanced"
  max_features: 5000
  ngram_range: [1, 2]
  min_df: 1
  fail_on_error: false
  output_json: "outputs/metrics/group_cross_validation_results.json"
  output_md: "outputs/reports/group_cross_validation_report.md"
```

A configuração de CI usa valores menores e caminhos em `outputs/ci_smoke`, mantendo a execução leve e independente do dataset oficial.

## Como executar

Para executar apenas a avaliação por agrupamento:

```powershell
python scripts/group_cross_validation.py --config configs/config.yaml
```

Para forçar uma coluna de agrupamento específica:

```powershell
python scripts/group_cross_validation.py --config configs/config.yaml --group-column autor
```

Para usar um CSV específico:

```powershell
python scripts/group_cross_validation.py --config configs/config.yaml --input data/raw/dataset.csv
```

A etapa também é chamada ao final de `scripts/run_pipeline.py` quando `group_cross_validation.enabled` está definido como `true`.

## Artefatos gerados

A execução gera:

```text
outputs/metrics/group_cross_validation_results.json
outputs/reports/group_cross_validation_report.md
```

O JSON contém:

- coluna de agrupamento escolhida;
- estratégia aplicada;
- número de grupos;
- número de folds solicitado e efetivo;
- métricas por fold;
- médias e desvios padrão;
- matriz de confusão agregada;
- alertas metodológicos.

O relatório Markdown resume os resultados em formato adequado para inspeção acadêmica e versionamento leve, sem incluir artefatos pesados.

## Métricas calculadas

As métricas calculadas por fold são:

- `accuracy`;
- `precision_macro`;
- `recall_macro`;
- `f1_macro`.

Também é gerada uma matriz de confusão agregada entre folds.

## Limitações

A avaliação por agrupamento é complementar e deve ser interpretada com cautela:

- se houver poucos grupos, o número efetivo de folds pode ser reduzido automaticamente;
- se os grupos tiverem distribuição de classes muito desigual, alguns folds podem ser mais difíceis do que outros;
- quando o fallback sintético por linha é usado, a avaliação não mede generalização por obra nem por autor;
- os resultados continuam dependentes dos rótulos heurísticos adotados pelo projeto;
- a rotina usa um classificador textual leve baseado em TF-IDF para evitar acoplamento excessivo ao pipeline principal.
