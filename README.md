# complexidade-cognitiva-ptbr

Pipeline acadêmico em Python para preparação de dados, extração de atributos linguísticos, treinamento, comparação e avaliação de modelos supervisionados voltados à estimação automática da complexidade cognitiva em textos literários em português.

Este repositório faz parte do TCC **Avaliação Automática de Complexidade Cognitiva em Textos Literários na Língua Portuguesa utilizando Processamento de Linguagem Natural e Aprendizado de Máquina**. O projeto combina métricas linguísticas interpretáveis e representações textuais, como TF-IDF, para classificar textos em níveis de complexidade cognitiva e gerar artefatos úteis para análise acadêmica.

## Objetivo

Desenvolver e avaliar uma abordagem computacional capaz de classificar textos literários em níveis de complexidade cognitiva a partir de um corpus textual rotulado.

O pipeline foi planejado para:

- ler e validar um dataset textual em CSV;
- aplicar pré-processamento textual controlado e reprodutível;
- dividir o corpus em treino, validação e teste;
- extrair métricas linguísticas interpretáveis;
- construir representações numéricas dos textos;
- treinar modelos supervisionados de classificação;
- comparar experimentos em conjunto de validação;
- selecionar objetivamente o melhor modelo;
- avaliar o melhor modelo no conjunto de teste;
- gerar métricas, predições, matriz de confusão e relatório final para apoio ao TCC.

## Escopo

Este repositório cobre a etapa local de treinamento, experimentação e avaliação do projeto.

Ficam fora do escopo deste repositório:

- API;
- interface web;
- banco de dados;
- deploy em cloud;
- execução como serviço de produção.

## Requisitos

- Python 3.11 ou superior compatível com as dependências do projeto;
- ambiente virtual Python;
- dataset CSV em UTF-8;
- dependências instaladas via `requirements.txt` ou `pyproject.toml`.

Dependências principais:

- pandas;
- numpy;
- scikit-learn;
- matplotlib;
- PyYAML;
- joblib;
- pytest.

## Estrutura do projeto

```text
complexidade-cognitiva-ptbr/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   └── processed/
│       └── features/
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   ├── reports/
│   ├── runs/
│   └── latest/
├── scripts/
│   ├── build_features.py
│   ├── evaluate_model.py
│   ├── prepare_dataset.py
│   ├── run_pipeline.py
│   ├── train_model.py
│   ├── explain_model.py
│   ├── predict_text.py
│   ├── predict_file.py
│   └── validate_artifacts.py
├── src/
│   └── complexidade_cognitiva_ptbr/
│       ├── __init__.py
│       ├── pipeline.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── io.py
│       │   └── preprocessing.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── reporting.py
│       ├── features/
│       │   ├── __init__.py
│       │   └── build_features.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── train.py
│       └── utils/
│           ├── __init__.py
│           ├── logging_utils.py
│           └── paths.py
├── tests/
├── .gitignore
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Dataset de entrada

O dataset bruto oficial deve ficar em:

```text
data/raw/dataset.csv
```

Por padrão, o CSV deve conter as seguintes colunas:

| Coluna | Obrigatória | Descrição |
| --- | --- | --- |
| `id` | Sim | Identificador único do texto. |
| `texto` | Sim | Texto literário a ser analisado. |
| `target` | Sim | Classe de complexidade cognitiva atribuída ao texto. |

O problema é tratado como classificação supervisionada multiclasse.

### Regras principais de validação

- o arquivo deve ser CSV;
- o arquivo deve estar em UTF-8;
- as colunas obrigatórias não podem estar ausentes;
- `id`, `texto` e `target` não podem conter valores nulos ou vazios;
- `id` deve ser único;
- o dataset precisa ter pelo menos duas classes distintas;
- textos que ficarem vazios após a limpeza são rejeitados;
- a divisão precisa gerar conjuntos não vazios de treino, validação e teste.

## Configuração

A configuração central fica em:

```text
configs/config.yaml
```

Ela controla:

- metadados do projeto;
- caminho do dataset bruto;
- nomes das colunas obrigatórias;
- proporções de treino, validação e teste;
- uso de estratificação;
- opções de limpeza textual;
- diretório de features;
- parâmetros de TF-IDF;
- modelos supervisionados habilitados;
- representações textuais avaliadas;
- métrica usada para selecionar o melhor experimento;
- diretórios de saída;
- formato e nível de logging.

Antes de rodar o pipeline completo, valide configuração, logging e diretórios:

```bash
python scripts/run_pipeline.py --bootstrap-only
```

## Instalação

Crie o ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente no Windows:

```bash
.venv\Scripts\activate
```

Ative o ambiente no Linux ou macOS:

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Execução do pipeline

Execute as etapas individualmente na ordem abaixo.

### 1. Preparação dos dados

```bash
python scripts/prepare_dataset.py
```

Arquivos gerados:

- `data/processed/train.csv`;
- `data/processed/val.csv`;
- `data/processed/test.csv`;
- `data/processed/preparation_report.json`.

### 2. Extração de features linguísticas

```bash
python scripts/build_features.py
```

Arquivos gerados:

- `data/processed/features/train_features.csv`;
- `data/processed/features/val_features.csv`;
- `data/processed/features/test_features.csv`;
- `data/processed/features/features_metadata.json`.

### 3. Treinamento e seleção do melhor modelo

```bash
python scripts/train_model.py
```

Arquivos gerados:

- `outputs/models/best_model.joblib`;
- `outputs/models/best_experiment.json`;
- `outputs/models/experiment_results.csv`.

### 4. Avaliação final

```bash
python scripts/evaluate_model.py
```

Arquivos gerados:

- `outputs/metrics/final_metrics.json`;
- `outputs/metrics/classification_report.json`;
- `outputs/metrics/classification_report.txt`;
- `outputs/metrics/test_predictions.csv`;
- `outputs/figures/confusion_matrix.png`;
- `outputs/reports/final_report.md`.

### Execução completa

```bash
python scripts/run_pipeline.py
```

## Execução com configuração alternativa

Todos os scripts aceitam o argumento `--config`:

```bash
python scripts/run_pipeline.py --config configs/config.yaml
```

Exemplo com outro arquivo de configuração:

```bash
python scripts/run_pipeline.py --config configs/config-experimento.yaml
```

## Representações suportadas

As representações disponíveis dependem da configuração, mas o pipeline suporta:

| Representação | Descrição |
| --- | --- |
| `linguistic_metrics` | Usa apenas métricas linguísticas interpretáveis. |
| `tfidf_word` | Usa TF-IDF baseado em palavras. |
| `tfidf_char` | Usa TF-IDF baseado em caracteres. |
| `tfidf_word_plus_linguistic_metrics` | Combina TF-IDF de palavras com métricas linguísticas. |

## Modelos suportados

| Modelo | Identificador de configuração |
| --- | --- |
| Regressão Logística | `logistic_regression` |
| Linear SVM | `linear_svm` |
| Random Forest | `random_forest` |

## Métricas calculadas

O pipeline calcula as seguintes métricas em validação e teste:

- `accuracy`;
- `precision_macro`;
- `recall_macro`;
- `f1_macro`;
- `f1_weighted`;
- `balanced_accuracy`.

A seleção do melhor experimento usa a métrica definida em `training.selection_metric` no arquivo `configs/config.yaml`.

## Testes

Execute a suíte de testes com:

```bash
pytest -q
```

Os testes cobrem:

- leitura e validação de configurações;
- criação de diretórios gerenciados;
- limpeza textual;
- validação do dataset;
- divisão em treino, validação e teste;
- extração de métricas linguísticas;
- treinamento mínimo;
- avaliação final e geração de artefatos.

## Melhorias experimentais implementadas

O pipeline inclui recursos de robustez e rastreabilidade:

- versionamento de execuções em `outputs/runs/<run_id>/`;
- ponteiro da última execução em `outputs/latest/run_id.txt`;
- fingerprint do dataset bruto sem alterar o arquivo oficial;
- detecção de vazamento entre treino, validação e teste;
- validação cruzada estratificada;
- busca configurável de hiperparâmetros;
- `ModelBundle` persistido para inferência local;
- scripts `predict_text.py` e `predict_file.py`;
- explicabilidade global por coeficientes TF-IDF, quando suportada;
- explicações locais, gráficos avançados e relatório final estendido;
- validação objetiva de artefatos com `scripts/validate_artifacts.py`;
- CI no GitHub Actions com lint, testes, cobertura e smoke test sem dataset oficial.

### Execução final recomendada

```bash
python scripts/run_pipeline.py --bootstrap-only
python scripts/run_pipeline.py
python scripts/validate_artifacts.py --profile complete
python scripts/predict_text.py --text "Texto literário de exemplo para classificação."
python -m pytest
ruff check .
```

### Smoke test sem dataset oficial

Para validar o projeto sem usar `data/raw/dataset.csv`, rode:

```bash
python scripts/run_pipeline.py --config configs/ci_smoke_config.yaml
python scripts/validate_artifacts.py --config configs/ci_smoke_config.yaml --profile smoke
python scripts/predict_text.py --config configs/ci_smoke_config.yaml --text "Texto literário de exemplo para classificação."
```

Veja também `docs/guia_validacao_final.md`.

## Versionamento no Git

Este repositório foi organizado para uso acadêmico no TCC. A regra principal é manter versionado tudo que permite reproduzir o experimento e deixar fora do Git aquilo que é gerado automaticamente, sensível ou pesado.

## Organização acadêmica dos artefatos

Os artefatos finais apoiam o capítulo de Resultados e Discussão do TCC:

| Artefato | Finalidade acadêmica |
| --- | --- |
| `data/processed/preparation_report.json` | Descreve validação, limpeza e divisão dos dados |
| `data/processed/features/features_metadata.json` | Documenta as métricas linguísticas geradas |
| `outputs/models/experiment_results.csv` | Compara os experimentos treinados |
| `outputs/models/best_experiment.json` | Registra o melhor experimento selecionado |
| `outputs/metrics/final_metrics.json` | Consolida as métricas finais no conjunto de teste |
| `outputs/metrics/classification_report.json` | Detalha precisão, revocação e F1 por classe |
| `outputs/metrics/classification_report.txt` | Registra o relatório textual de classificação |
| `outputs/metrics/test_predictions.csv` | Registra predições individuais no conjunto de teste |
| `outputs/figures/confusion_matrix.png` | Ilustra erros e acertos por classe |
| `outputs/reports/final_report.md` | Resume a avaliação final em formato textual |

## Fluxo recomendado de desenvolvimento

1. Ajustar `configs/config.yaml`;
2. validar estrutura com `python scripts/run_pipeline.py --bootstrap-only`;
3. inserir o dataset oficial em `data/raw/dataset.csv`;
4. rodar `python scripts/prepare_dataset.py`;
5. rodar `python scripts/build_features.py`;
6. rodar `python scripts/train_model.py`;
7. rodar `python scripts/evaluate_model.py`;
8. executar `pytest -q`;
9. revisar os artefatos gerados em `data/processed/` e `outputs/`;
10. selecionar os artefatos relevantes para uso no capítulo de Resultados e Discussão.

## Solução de problemas

### Dataset não encontrado

Confirme se o arquivo existe em:

```text
data/raw/dataset.csv
```

Ou ajuste `dataset.input_path` em `configs/config.yaml`.

### Split estratificado falhou

Quando há poucas amostras por classe, a estratificação pode ser inviável. Nesse caso, aumente o dataset, reduza o número de classes ou ajuste as proporções de treino, validação e teste.

### TF-IDF sem vocabulário

Esse erro costuma ocorrer quando os textos estão vazios, muito curtos ou quando `min_df` está alto demais para o tamanho do dataset. Revise os textos limpos e os parâmetros de TF-IDF no YAML.

## Observação sobre uso acadêmico

O pipeline não substitui a interpretação teórica do conceito de complexidade cognitiva. Portanto, ele fornece uma base computacional reprodutível para apoiar a análise experimental do TCC, permitindo comparar modelos, observar métricas linguísticas e discutir resultados quantitativos de forma documentada.
