# complexidade-cognitiva-ptbr

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-informational)
![Status](https://img.shields.io/badge/status-acad%C3%AAmico%20experimental-orange)

Pipeline experimental em Python para estimação automática de níveis operacionais de complexidade cognitiva em textos literários na língua portuguesa, desenvolvido no contexto do TCC **Avaliação Automática de Complexidade Cognitiva em Textos Literários na Língua Portuguesa utilizando Processamento de Linguagem Natural e Aprendizado de Máquina**.

O projeto aplica técnicas de processamento de linguagem natural e aprendizado de máquina para classificar trechos literários em três classes operacionais: `baixa`, `media` e `alta`. Essas classes representam uma aproximação heurística e metodológica de complexidade textual, construída a partir de critérios linguísticos e discursivos.

> Este repositório não é uma API, não é uma aplicação web e não substitui avaliação humana especializada. O objetivo é apoiar uma investigação experimental, reprodutível e documentada sobre sinais linguísticos associados à complexidade cognitiva textual.

## Visão geral

O repositório organiza um pipeline local de PLN e aprendizado de máquina para:

- construir um dataset rastreável a partir de fontes literárias públicas;
- segmentar textos em trechos controlados;
- atribuir rótulos heurísticos com base em critérios linguísticos;
- preparar divisões de treino, validação e teste;
- verificar possíveis vazamentos entre partições;
- extrair métricas linguísticas interpretáveis;
- treinar modelos supervisionados com diferentes representações textuais;
- executar validação cruzada tradicional;
- executar avaliação complementar por agrupamento de obra, autor ou fonte;
- realizar busca de hiperparâmetros;
- avaliar o melhor modelo no conjunto de teste;
- gerar relatórios, figuras e artefatos de explicabilidade;
- permitir inferência local por texto único ou arquivo CSV.

## Objetivo do projeto

O objetivo é desenvolver e avaliar uma abordagem automática para estimar níveis operacionais de complexidade cognitiva em trechos literários em português, combinando:

- métricas linguísticas interpretáveis;
- representações textuais por TF-IDF;
- modelos clássicos de aprendizado de máquina;
- validações experimentais com controle contra vazamento;
- análise de explicabilidade para apoiar interpretação acadêmica dos resultados.

A proposta busca verificar se sinais textuais e linguísticos permitem distinguir, de forma mensurável, trechos classificados em níveis `baixa`, `media` e `alta`, sempre considerando que os rótulos são heurísticos e dependem da rubrica operacional adotada.

## Escopo do repositório

Este repositório cobre o ciclo experimental principal do TCC:

- configuração do experimento em YAML;
- construção e auditoria do dataset;
- preparação dos dados;
- extração de features;
- treinamento e seleção de modelos;
- validação cruzada;
- avaliação por grupos;
- detecção de vazamento;
- busca de hiperparâmetros;
- avaliação final;
- explicabilidade;
- inferência local;
- testes automatizados;
- CI com execução smoke.

Os dados brutos, textos baixados, arquivos processados, modelos treinados e outputs locais são tratados como artefatos gerados. Em geral, eles não devem ser versionados diretamente no Git, pois podem ser recriados a partir dos scripts, configurações e catálogo de fontes públicas.

## O que o projeto faz

- Classifica trechos literários em português em níveis operacionais de complexidade cognitiva.
- Usa classes `baixa`, `media` e `alta`.
- Extrai métricas linguísticas como tamanho médio de sentença, diversidade lexical, proporção de palavras longas, pontuação, conectivos e marcadores de subordinação.
- Compara representações `linguistic_metrics`, `tfidf_word`, `tfidf_char` e `tfidf_word_plus_linguistic_metrics`.
- Avalia modelos `logistic_regression`, `linear_svm` e `random_forest`.
- Usa `f1_macro` como métrica principal de seleção na configuração padrão.
- Executa validação cruzada estratificada.
- Executa avaliação complementar com `GroupKFold` por obra, autor ou fonte.
- Verifica vazamento por ID, texto exato, texto normalizado, quase duplicatas e colunas semelhantes ao alvo.
- Gera relatórios em JSON, CSV, Markdown e figuras PNG.
- Salva bundles de modelo para inferência local.
- Possui configuração smoke para CI sem depender do dataset oficial.

## O que o projeto não faz

- Não substitui análise humana especializada sobre literatura, leitura ou cognição.
- Não produz uma medida absoluta ou definitiva de complexidade cognitiva.
- Não julga qualidade literária.
- Não fornece API HTTP ou interface web.
- Não garante generalização para qualquer gênero textual fora do corpus avaliado.
- Não elimina a necessidade de análise metodológica dos dados, rótulos e resultados.
- Não deve ser interpretado apenas por acurácia isolada ou por uma única execução.

## Fundamentação resumida da abordagem

A hipótese operacional do projeto é que certos sinais linguísticos e textuais podem se correlacionar com maior ou menor demanda interpretativa de trechos literários. Entre esses sinais estão extensão e variação de sentenças, diversidade lexical, densidade de conectivos, subordinação, pontuação, abstração, ambiguidade e padrões lexicais capturados por TF-IDF.

A abordagem combina duas dimensões:

1. **Métricas interpretáveis**, úteis para análise acadêmica e inspeção qualitativa.
2. **Representações vetoriais**, úteis para capturar padrões lexicais e estilísticos aprendidos pelos modelos.

A avaliação não busca provar uma verdade universal sobre complexidade cognitiva. Ela mede a aderência dos modelos à definição operacional proposta no dataset e discute os resultados dentro das limitações da rotulagem heurística.

## Observação metodológica sobre os rótulos

Os rótulos `baixa`, `media` e `alta` são heurísticos e operacionais. Eles são construídos a partir de uma rubrica que aproxima sinais linguísticos e discursivos associados à complexidade textual.

Isso significa que:

- o modelo aprende a reproduzir a rubrica usada no experimento;
- as classes não são verdades absolutas sobre os textos;
- os resultados devem ser interpretados como evidência experimental limitada;
- métricas altas podem indicar forte aderência aos critérios de rotulagem, não validação definitiva do conceito de complexidade cognitiva;
- a análise quantitativa deve ser acompanhada de leitura crítica dos relatórios, exemplos e limitações.

A documentação da rubrica está em [`docs/dataset/rubrica_rotulagem.md`](docs/dataset/rubrica_rotulagem.md).

## Estrutura do repositório

```text
complexidade-cognitiva-ptbr/
├── .github/workflows/
│   └── ci.yml
├── configs/
│   ├── config.yaml
│   ├── ci_smoke_config.yaml
│   └── dataset.yaml
├── data/
│   ├── external/
│   │   └── public_domain_sources.csv
│   ├── raw/ # gerado localmente
│   └── processed/ # gerado localmente
├── docs/
│   ├── avaliacao_por_grupo.md
│   ├── relatorio_execucao.md
│   └── dataset/
│       ├── README.md
│       ├── comandos_rapidos.md
│       ├── metodologia_construcao.md
│       └── rubrica_rotulagem.md
├── outputs/ # artefatos gerados localmente
├── scripts/
│   ├── build_features.py
│   ├── dataset_audit.py
│   ├── dataset_build.py
│   ├── dataset_download_sources.py
│   ├── dataset_promote.py
│   ├── evaluate_model.py
│   ├── explain_model.py
│   ├── group_cross_validation.py
│   ├── predict_file.py
│   ├── predict_text.py
│   ├── prepare_dataset.py
│   ├── run_pipeline.py
│   ├── train_model.py
│   └── validate_artifacts.py
├── src/
│   └── complexidade_cognitiva_ptbr/
│       ├── config/
│       ├── data/
│       ├── evaluation/
│       ├── features/
│       ├── models/
│       └── utils/
├── tests/
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Requisitos

- Python `3.11` ou superior.
- Sistema operacional compatível com ambiente Python local.
- Acesso à internet apenas para baixar fontes públicas do dataset, quando for reconstruí-lo.
- Dependências Python definidas em `pyproject.toml` e espelhadas em `requirements.txt`.

Principais bibliotecas usadas:

- `pandas`;
- `numpy`;
- `scikit-learn`;
- `nltk`;
- `matplotlib`;
- `PyYAML`;
- `joblib`;
- `pytest` para testes.

## Instalação

### Windows

```powershell
python -m venv .venv
.\venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

### Fluxo equivalente ao CI

O workflow de CI instala as dependências por `requirements.txt` e depois instala o pacote local em modo editável sem reinstalar dependências:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip check
```

Para desenvolvimento local, a instalação por `pyproject.toml` com `.[dev]` é a opção mais prática, pois inclui ferramentas de teste e qualidade configuradas no projeto.

## Execução dos testes

Para executar a suíte automatizada:

```bash
python -m pytest
```

Para executar um conjunto específico:

```bash
python -m pytest tests/test_group_cross_validation.py
python -m pytest tests/test_leakage.py
python -m pytest tests/test_inference.py
```

O projeto também possui configuração de lint no `pyproject.toml`:

```bash
ruff check .
```

A suíte cobre preparação de dados, extração de features, treinamento, avaliação, validação cruzada, busca de hiperparâmetros, vazamento, explicabilidade, inferência local, validação de artefatos e avaliação por agrupamento.

## Execução do pipeline completo

A execução completa depende da existência de `data/raw/dataset.csv`. Esse arquivo é um dado gerado localmente, portanto não foi versionado.

### 1. Baixar textos públicos

```bash
python scripts/dataset_download_sources.py --config configs/dataset.yaml
```

Esse comando lê `data/external/public_domain_sources.csv` e salva textos baixados em `data/external/public_domain_texts/`.

### 2. Construir dataset

```bash
python scripts/dataset_build.py --config configs/dataset.yaml
```

Esse comando gera o dataset candidato e, por padrão, também grava o dataset principal em `data/raw/dataset.csv`.

Para gerar apenas o candidato:

```bash
python scripts/dataset_build.py --config configs/dataset.yaml --no-write-main
```

### 3. Auditar dataset

```bash
python scripts/dataset_audit.py --config configs/dataset.yaml --input data/raw/dataset.csv
```

A auditoria gera relatórios de qualidade em `outputs/metrics/` e `outputs/reports/`.

### 4. Executar pipeline principal

```bash
python scripts/run_pipeline.py --config configs/config.yaml
```

O pipeline principal executa:

1. preparação do dataset em treino, validação e teste;
2. verificação de vazamento;
3. extração de features linguísticas;
4. validação cruzada tradicional;
5. treinamento e seleção de modelos;
6. avaliação final no conjunto de teste.

Quando `group_cross_validation.enabled` está como `true`, o script também tenta executar a avaliação complementar por agrupamento ao final da execução.

### 5. Validar artefatos finais

```bash
python scripts/validate_artifacts.py --config configs/config.yaml --profile complete
```

## Execução leve de CI/smoke

O projeto inclui uma configuração leve em `configs/ci_smoke_config.yaml`, com dataset fixture em `tests/fixtures/smoke_dataset.csv`. Essa execução existe para validar rapidamente instalação, testes, pipeline mínimo e inferência sem depender do dataset oficial.

```bash
python scripts/run_pipeline.py --config configs/ci_smoke_config.yaml
python scripts/validate_artifacts.py --config configs/ci_smoke_config.yaml --profile smoke
python scripts/predict_text.py \
  --config configs/ci_smoke_config.yaml \
  --text "Texto literário de exemplo para classificação."
```

A execução smoke não substitui a execução experimental completa. Ela serve para CI, sanidade do pipeline e detecção rápida de regressões.

## Scripts principais

| Script | Função |
| --- | --- |
| `scripts/dataset_download_sources.py` | Baixa textos públicos listados no catálogo `data/external/public_domain_sources.csv` |
| `scripts/dataset_build.py` | Constrói o dataset a partir dos textos baixados, aplica segmentação, escore heurístico e rotulagem |
| `scripts/dataset_audit.py` | Audita distribuição, duplicatas, concentração por obra/autor e ruídos do dataset |
| `scripts/dataset_promote.py` | Promove um dataset candidato para `data/raw/dataset.csv`, com opção de backup |
| `scripts/prepare_dataset.py` | Divide o dataset bruto em treino, validação e teste |
| `scripts/build_features.py` | Gera arquivos de métricas linguísticas para cada split |
| `scripts/train_model.py` | Treina modelos, compara experimentos, executa busca de hiperparâmetros quando habilitada e salva o melhor modelo |
| `scripts/evaluate_model.py` | Avalia o melhor modelo no conjunto de teste e gera métricas, relatórios e figuras |
| `scripts/explain_model.py` | Gera artefatos de explicabilidade global e local para o melhor modelo disponível |
| `scripts/group_cross_validation.py` | Executa avaliação complementar por `GroupKFold` usando obra, autor, fonte ou fallback sintético |
| `scripts/run_pipeline.py` | Executa o pipeline integrado |
| `scripts/validate_artifacts.py` | Valida a presença e integridade mínima dos artefatos gerados em uma execução |
| `scripts/predict_text.py` | Executa inferência local para um texto informado por linha de comando |
| `scripts/predict_file.py` | Executa inferência local em lote para um arquivo CSV |

## Configurações principais em YAML

### `configs/config.yaml`

Configuração principal do experimento. Define:

- caminhos de entrada e saída;
- nomes das colunas `id`, `texto` e `target`;
- proporção treino/validação/teste;
- limpeza textual;
- features linguísticas;
- TF-IDF por palavra e caractere;
- modelos e representações avaliadas;
- validação cruzada;
- verificação de vazamento;
- busca de hiperparâmetros;
- explicabilidade;
- versionamento de execuções;
- inferência local;
- relatórios visuais;
- avaliação complementar por agrupamento.

### `configs/ci_smoke_config.yaml`

Configuração reduzida para CI. Usa `tests/fixtures/smoke_dataset.csv`, menos features, menos folds e treinamento mais rápido. Não deve ser usada como avaliação acadêmica final.

### `configs/dataset.yaml`

Configuração de construção do dataset. Define catálogo de fontes, diretórios de textos, segmentação, pesos do escore heurístico, balanceamento e auditoria.

## Avaliações implementadas

### Avaliação tradicional treino/validação/teste

O dataset é dividido em treino, validação e teste conforme `configs/config.yaml`:

```yaml
train_size: 0.70
val_size: 0.15
test_size: 0.15
stratify: true
```

O treinamento usa treino e validação para selecionar o melhor experimento. O conjunto de teste é reservado para avaliação final.

### Validação cruzada tradicional

A validação cruzada está configurada como `stratified_kfold`, com 5 folds na configuração principal. Ela calcula métricas como:

- `accuracy`;
- `precision_macro`;
- `recall_macro`;
- `f1_macro`;
- `f1_weighted`;
- `balanced_accuracy`.

Os artefatos esperados incluem `cv_results.csv`, `cv_summary.json`, `cv_report.md` e figuras de resumo quando habilitadas.

### Avaliação complementar por obra, autor ou fonte

O projeto inclui uma avaliação complementar com `GroupKFold`, documentada em [`docs/avaliacao_por_grupo.md`](docs/avaliacao_por_grupo.md).

A rotina tenta selecionar grupos nesta ordem metodológica:

1. coluna preferencial `obra`;
2. fallbacks configurados, como `titulo`, `autor`, `fonte`, `source` e equivalentes;
3. fallback sintético por linha, caso nenhuma coluna de agrupamento esteja disponível.

Quando o agrupamento real por obra ou autor é usado, exemplos do mesmo grupo não aparecem simultaneamente em treino e validação dentro do mesmo fold. Isso torna a avaliação mais exigente e ajuda a verificar se o modelo depende excessivamente de padrões específicos de uma obra, autor ou fonte.

O fallback sintético por linha é seguro para execução e CI, mas não deve ser interpretado como evidência de generalização por obra ou autor.

Execução isolada:

```bash
python scripts/group_cross_validation.py --config configs/config.yaml
```

Forçando coluna de agrupamento:

```bash
python scripts/group_cross_validation.py --config configs/config.yaml --group-column autor
```

### Verificação de vazamento

A etapa de vazamento verifica:

- sobreposição de IDs entre splits;
- sobreposição exata de textos;
- sobreposição de textos normalizados;
- quase duplicatas por similaridade;
- colunas com aparência de alvo;
- falha em caso de vazamento crítico, quando configurado.

Essa etapa é importante porque o dataset é segmentado em janelas textuais e pode conter trechos próximos. Sem esse controle, métricas de validação poderiam ser infladas por exemplos muito semelhantes em partições diferentes.

### Busca de hiperparâmetros

A busca está habilitada em `configs/config.yaml`:

```yaml
hyperparameter_search:
  enabled: true
  strategy: "grid"
  refit_metric: "f1_macro"
```

Modelos e hiperparâmetros configurados:

| Modelo | Hiperparâmetros |
| --- | --- |
| `logistic_regression` | `C`, `max_iter`, `class_weight` |
| `linear_svm` | `C`, `class_weight` |
| `random_forest` | `n_estimators`, `max_depth`, `min_samples_split`, `class_weight` |

Na configuração smoke, a busca de hiperparâmetros fica desabilitada para manter o CI rápido.

### Explicabilidade

A explicabilidade está habilitada na configuração principal. O projeto gera artefatos como:

- termos TF-IDF mais relevantes por classe;
- relatório Markdown de explicabilidade;
- amostras de explicações locais;
- gráficos de termos por classe quando aplicável.

A explicabilidade por coeficientes é mais direta quando o melhor modelo é linear, como `logistic_regression` ou `linear_svm`. Para modelos não lineares, a interpretação por coeficientes diretos pode não estar disponível ou exigir leitura diferente.

## Inferência local

Após treinar o pipeline e gerar `outputs/latest/models/best_model_bundle.joblib`, é possível classificar um texto diretamente no terminal:

```bash
python scripts/predict_text.py \
  --text "O narrador reconstrói a memória de modo fragmentado, misturando dúvida, culpa e símbolos ambíguos."
```

Para saída JSON:

```bash
python scripts/predict_text.py \
  --text "Texto literário para classificar." \
  --json
```

Para usar um bundle específico:

```bash
python scripts/predict_text.py \
  --model-path outputs/latest/models/best_model_bundle.joblib \
  --text "Texto literário para classificar."
```

Inferência em lote por CSV:

```bash
python scripts/predict_file.py \
  --input data/external/exemplos_inferencia.csv \
  --output outputs/latest/predictions/predictions.csv \
  --text-column texto
```

A inferência local deve ser interpretada como aplicação do modelo experimental treinado, não como diagnóstico definitivo de complexidade literária.

## Artefatos gerados

Os principais artefatos esperados são:

| Etapa | Artefatos |
| --- | --- |
| Download de fontes | `outputs/metrics/dataset_download_manifest.json` |
| Construção do dataset | `data/raw/dataset_candidate.csv`, `data/raw/dataset.csv`, `docs/dataset_card.md` |
| Auditoria do dataset | `outputs/metrics/dataset_quality_report.json`, `outputs/reports/dataset_quality_report.md` |
| Preparação | `data/processed/train.csv`, `data/processed/val.csv`, `data/processed/test.csv`, `data/processed/preparation_report.json` |
| Features | `data/processed/features/*_features.csv`, `features_metadata.json` |
| Vazamento | `metrics/leakage_report.json`, `reports/leakage_report.md`, `near_duplicate_pairs.csv` quando aplicável |
| Validação cruzada | `metrics/cv_results.csv`, `metrics/cv_summary.json`, `reports/cv_report.md` |
| Treinamento | `models/best_model.joblib`, `models/best_model_bundle.joblib`, `models/best_experiment.json`, `models/experiment_results.csv` |
| Hiperparâmetros | `metrics/hyperparameter_search_results.csv`, `metrics/best_hyperparameters.json`, `reports/hyperparameter_search_report.md` |
| Avaliação final | `metrics/final_metrics.json`, `classification_report.*`, `test_predictions.csv`, `reports/final_report.md`, `reports/final_report_extended.md` |
| Figuras | matrizes de confusão, comparação de experimentos, distribuição de confiança, gráficos de features e termos |
| Explicabilidade | `explainability/tfidf_top_terms_by_class.*`, `tfidf_explainability_report.md`, `local_explanations_sample.csv` |
| Avaliação por grupo | `outputs/metrics/group_cross_validation_results.json`, `outputs/reports/group_cross_validation_report.md` |
| Inferência em lote | `outputs/latest/predictions/predictions.csv` |

As execuções versionadas ficam em:

```text
outputs/runs/<run_id>/
```

O ponteiro para a última execução fica em:

```text
outputs/latest/run_id.txt
```

## Documentação principal

A documentação existente em `docs/` deve ser usada como complemento ao README:

| Documento | Finalidade |
| --- | --- |
| [`docs/relatorio_execucao.md`](docs/relatorio_execucao.md) | Relatório técnico final de execução e visão metodológica geral |
| [`docs/avaliacao_por_grupo.md`](docs/avaliacao_por_grupo.md) | Explicação detalhada da avaliação complementar por obra, autor ou fonte |
| [`docs/dataset/README.md`](docs/dataset/README.md) | Guia rápido sobre o dataset |
| [`docs/dataset/comandos_rapidos.md`](docs/dataset/comandos_rapidos.md) | Comandos práticos para construção e auditoria do dataset |
| [`docs/dataset/metodologia_construcao.md`](docs/dataset/metodologia_construcao.md) | Metodologia de construção do dataset |
| [`docs/dataset/rubrica_rotulagem.md`](docs/dataset/rubrica_rotulagem.md) | Rubrica operacional de rotulagem heurística |

Arquivos como `docs/relatorio_execucao_final.md`, `docs/metodologia_experimental.md`, `docs/analise_resultados.md`, `docs/explicabilidade.md` e `docs/limitacoes_e_validade.md` não estavam presentes na estrutura analisada. Por isso, este README aponta apenas para os documentos realmente existentes.

## CI com GitHub Actions

O workflow de CI está em:

```text
.github/workflows/ci.yml
```

Ele executa em `push` e `pull_request` para as branches:

- `prod`;
- `main`;
- `master`.

O job principal:

1. configura Python `3.11`;
2. instala dependências;
3. instala o pacote local;
4. valida arquivos essenciais do smoke test;
5. executa `pytest`;
6. executa o pipeline leve com `configs/ci_smoke_config.yaml`;
7. valida artefatos smoke;
8. executa uma predição local simples.

O CI foi desenhado para não depender do dataset oficial, de modelos treinados previamente, de artefatos grandes ou de outputs locais.

## Reprodutibilidade

A reprodutibilidade é apoiada por:

- `random_state` fixado em `42`;
- configurações YAML versionáveis;
- catálogo de fontes públicas versionado em `data/external/public_domain_sources.csv`;
- scripts separados para download, construção, auditoria e execução experimental;
- logs de execução;
- snapshots de configuração;
- relatório de ambiente;
- fingerprint do dataset;
- manifesto por execução;
- ponteiro `outputs/latest/run_id.txt` para a execução mais recente.

Fluxo recomendado para reprodução completa:

```bash
python scripts/dataset_download_sources.py --config configs/dataset.yaml
python scripts/dataset_build.py --config configs/dataset.yaml
python scripts/dataset_audit.py --config configs/dataset.yaml --input data/raw/dataset.csv
python scripts/run_pipeline.py --config configs/config.yaml
python scripts/group_cross_validation.py --config configs/config.yaml
python scripts/validate_artifacts.py --config configs/config.yaml --profile complete
```

Fluxo recomendado para sanidade rápida:

```bash
python -m pytest
python scripts/run_pipeline.py --config configs/ci_smoke_config.yaml
python scripts/validate_artifacts.py --config configs/ci_smoke_config.yaml --profile smoke
```

## Limitações e ameaças à validade

As principais limitações do projeto são:

1. **Rotulagem heurística**: as classes são aproximações operacionais, não consenso humano especializado.
2. **Dependência da rubrica**: o modelo aprende os padrões definidos pelos critérios de rotulagem.
3. **Viés de corpus**: obras de domínio público podem concentrar estilos, períodos e vocabulários específicos.
4. **Segmentação por janelas**: trechos isolados podem perder contexto narrativo mais amplo.
5. **Sobreposição textual**: janelas próximas exigem controle rigoroso de vazamento e quase duplicidade.
6. **Padrões de obra ou autor**: métricas tradicionais podem ser otimistas quando trechos da mesma origem aparecem em diferentes partições.
7. **Explicabilidade limitada**: termos relevantes ajudam a interpretar sinais aprendidos, mas não provam complexidade cognitiva por si só.
8. **Generalização externa**: os resultados não devem ser extrapolados automaticamente para outros gêneros, períodos, domínios ou formas de leitura.

## Como interpretar os resultados

A interpretação dos resultados deve combinar múltiplas evidências:

- métricas finais em teste;
- estabilidade na validação cruzada;
- comparação entre representações;
- relatório de vazamento;
- avaliação complementar por obra/autor/fonte;
- matriz de confusão;
- métricas por classe;
- termos explicativos e exemplos locais;
- qualidade e distribuição do dataset.

Algumas leituras recomendadas:

- Se a validação tradicional for alta, mas a avaliação por grupo cair bastante, pode haver dependência de padrões específicos de obra, autor ou fonte.
- Se `linguistic_metrics` tiver desempenho competitivo, os indicadores interpretáveis capturam parte relevante da rubrica.
- Se TF-IDF superar muito as métricas linguísticas, padrões lexicais e estilísticos podem estar contribuindo fortemente.
- Se houver vazamento crítico, as métricas não devem ser usadas como evidência experimental válida até a correção.
- Se as métricas forem muito altas, a explicabilidade e os relatórios de duplicidade devem ser consultados antes de qualquer conclusão.

## Organização acadêmica sugerida

Para avaliação acadêmica, a leitura sugerida é:

1. este `README.md`, como porta de entrada do repositório;
2. `docs/relatorio_execucao.md`, para visão técnica e metodológica consolidada;
3. `docs/dataset/metodologia_construcao.md`, para entender a construção do dataset;
4. `docs/dataset/rubrica_rotulagem.md`, para entender os rótulos heurísticos;
5. `docs/avaliacao_por_grupo.md`, para entender a avaliação complementar por obra/autor;
6. relatórios gerados em `outputs/runs/<run_id>/reports/`, quando disponíveis localmente;
7. métricas e figuras geradas em `outputs/runs/<run_id>/metrics/` e `outputs/runs/<run_id>/figures/`.

## Autor

**Vitor Oliveira Silva**

Projeto desenvolvido como parte do TCC **Avaliação Automática de Complexidade Cognitiva em Textos Literários na Língua Portuguesa utilizando Processamento de Linguagem Natural e Aprendizado de Máquina**.