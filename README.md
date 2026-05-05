# complexidade-cognitiva-ptbr

Projeto acadêmico desenvolvido para estudo, implementação e validação de uma abordagem computacional voltada à estimação automática da complexidade cognitiva em textos literários em português, utilizando Processamento de Linguagem Natural e Aprendizado de Máquina.

## Objetivo

Este repositório concentra a etapa de treinamento, experimentação e avaliação do projeto de TCC. O foco é investigar como métricas linguísticas interpretáveis e representações textuais podem contribuir para classificar automaticamente textos literários em níveis de complexidade cognitiva.

O pipeline foi planejado para:

- ler e validar um corpus textual rotulado;
- aplicar pré-processamento textual controlado;
- extrair métricas linguísticas interpretáveis;
- gerar representações numéricas dos textos;
- treinar modelos supervisionados de classificação;
- comparar experimentos;
- selecionar objetivamente o melhor modelo;
- avaliar o melhor modelo no conjunto de teste;
- gerar métricas finais, matriz de confusão, predições e relatório acadêmico.

## Escopo

Este repositório contempla somente a parte local de treinamento e validação experimental.

Não fazem parte deste escopo:

- API;
- interface web;
- banco de dados;
- dependência obrigatória de cloud;
- deploy em produção.

## Dados esperados

O dataset de entrada deve estar em `data/raw/dataset.csv` e conter, no mínimo, as colunas:

| Coluna | Descrição |
| --- | --- |
| `id` | Identificador único do texto |
| `texto` | Conteúdo textual a ser analisado |
| `target` | Classe de complexidade cognitiva atribuída ao texto |

O problema é tratado como classificação supervisionada multiclasse.

## Stack principal

- Python 3.11
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- PyYAML
- joblib
- pytest

## Estrutura do projeto

```text
complexidade-cognitiva-ptbr/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
│       └── features/
├── notebooks/
├── outputs/
│   ├── models/
│   ├── metrics/
│   ├── figures/
│   └── reports/
├── scripts/
│   ├── prepare_dataset.py
│   ├── build_features.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── run_pipeline.py
├── src/
│   └── complexidade_cognitiva_ptbr/
│       ├── config/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── evaluation/
│       ├── utils/
│       └── pipeline.py
├── tests/
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```

## Instalação

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
```

No Windows:

```bash
.venv\Scripts\activate
```

No Linux/macOS:

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Opcionalmente, instale o pacote em modo editável:

```bash
pip install -e .
```

## Execução

Coloque o dataset bruto em:

```text
data/raw/dataset.csv
```

Execute a preparação dos dados:

```bash
python scripts/prepare_dataset.py
```

Execute a extração de features linguísticas:

```bash
python scripts/build_features.py
```

Execute o treinamento e a seleção do melhor experimento:

```bash
python scripts/train_model.py
```

Execute a avaliação final no conjunto de teste:

```bash
python scripts/evaluate_model.py
```

Ou execute o pipeline disponível:

```bash
python scripts/run_pipeline.py
```

Para validar apenas configuração, logging e diretórios, sem exigir dataset:

```bash
python scripts/run_pipeline.py --bootstrap-only
```

## Etapas implementadas

### 1. Preparação dos dados

A etapa de preparação lê o CSV bruto, valida a estrutura mínima, limpa os textos de forma controlada e gera os subconjuntos:

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/preparation_report.json`

### 2. Extração de features linguísticas

A etapa de features gera métricas interpretáveis por texto, incluindo contagens estruturais, diversidade lexical, razão de pontuação, razão de números, conectivos, marcadores de subordinação e densidade lexical aproximada.

Arquivos gerados:

- `data/processed/features/train_features.csv`
- `data/processed/features/val_features.csv`
- `data/processed/features/test_features.csv`
- `data/processed/features/features_metadata.json`

### 3. Treinamento e seleção de modelos

A etapa de treinamento compara experimentos usando combinações entre representações e modelos supervisionados.

Representações suportadas:

- métricas linguísticas;
- TF-IDF de palavras;
- TF-IDF de caracteres;
- TF-IDF de palavras combinado com métricas linguísticas.

Modelos suportados:

- Regressão Logística;
- Linear SVM;
- Random Forest.

Métricas calculadas em validação:

- accuracy;
- precision macro;
- recall macro;
- f1 macro;
- f1 weighted;
- balanced accuracy.

Arquivos gerados:

- `outputs/models/best_model.joblib`
- `outputs/models/best_experiment.json`
- `outputs/models/experiment_results.csv`

### 4. Avaliação final e geração de artefatos

A etapa de avaliação carrega o melhor modelo selecionado, aplica o pipeline ao conjunto de teste e gera arquivos finais para análise acadêmica.

Métricas calculadas no teste:

- accuracy;
- precision macro;
- recall macro;
- f1 macro;
- f1 weighted;
- balanced accuracy.

Arquivos gerados:

- `outputs/metrics/final_metrics.json`
- `outputs/metrics/classification_report.json`
- `outputs/metrics/classification_report.txt`
- `outputs/metrics/test_predictions.csv`
- `outputs/figures/confusion_matrix.png`
- `outputs/reports/final_report.md`

## Configuração

A configuração central fica em:

```text
configs/config.yaml
```

Ela concentra parâmetros de projeto, dataset, split, features, representações, modelos, critério de seleção e diretórios de saída.

## Artefatos gerados pelo pipeline

A execução completa do pipeline deverá gerar:

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/preparation_report.json`
- `data/processed/features/train_features.csv`
- `data/processed/features/val_features.csv`
- `data/processed/features/test_features.csv`
- `data/processed/features/features_metadata.json`
- `outputs/models/best_model.joblib`
- `outputs/models/best_experiment.json`
- `outputs/models/experiment_results.csv`
- `outputs/metrics/final_metrics.json`
- `outputs/metrics/classification_report.json`
- `outputs/metrics/classification_report.txt`
- `outputs/metrics/test_predictions.csv`
- `outputs/figures/confusion_matrix.png`
- `outputs/reports/final_report.md`

## Organização do desenvolvimento

O desenvolvimento do repositório segue blocos progressivos:

1. estrutura base;
2. preparação dos dados;
3. extração de features linguísticas;
4. treinamento e seleção de modelos;
5. avaliação final e geração de artefatos;
6. testes, revisão e endurecimento.
