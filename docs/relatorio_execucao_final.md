# Relatório técnico final de execução

## 1. Visão geral do projeto

O projeto `complexidade-cognitiva-ptbr` implementa um pipeline local para estimar automaticamente níveis operacionais de complexidade cognitiva em trechos literários em português. A abordagem combina processamento de linguagem natural, métricas linguísticas interpretáveis e modelos clássicos de aprendizado de máquina para classificar trechos textuais em três classes operacionais: `baixa`, `media` e `alta`.

A proposta não pretende substituir avaliação humana especializada, nem produzir julgamento definitivo sobre qualidade literária. O sistema organiza uma aproximação computacional reprodutível, baseada em sinais linguísticos e discursivos, para apoiar análise experimental no contexto acadêmico do TCC.

## 2. Objetivo do experimento

O experimento tem como objetivo avaliar se representações textuais e métricas linguísticas permitem distinguir, de forma mensurável, níveis operacionais de complexidade cognitiva em trechos literários em português.

De forma específica, o experimento busca:

- construir um dataset rastreável a partir de obras literárias em português de domínio público;
- segmentar os textos em janelas controladas;
- atribuir rótulos heurísticos com base em uma rubrica operacional;
- extrair indicadores linguísticos interpretáveis;
- comparar representações textuais baseadas em TF-IDF e métricas linguísticas;
- avaliar modelos supervisionados com validação cruzada, busca de hiperparâmetros e teste final;
- gerar artefatos de avaliação, explicabilidade, inferência local e reprodutibilidade.

## 3. Escopo do repositório

O repositório cobre o ciclo experimental principal do projeto, incluindo:

- configuração do experimento em `configs/config.yaml`;
- configuração de construção do dataset em `configs/dataset.yaml`;
- configuração leve para CI/smoke test em `configs/ci_smoke_config.yaml`;
- scripts de download, construção, auditoria e promoção do dataset;
- scripts de preparação, extração de features, treinamento, avaliação e explicabilidade;
- scripts de inferência local para texto único e arquivo CSV;
- validação de artefatos gerados por execução versionada;
- documentação metodológica sobre construção do dataset e rubrica de rotulagem.

O repositório não deve depender de dataset real versionado, artefatos pesados, modelos treinados ou outputs locais para executar validações rápidas. O dataset final é recriado localmente a partir das fontes registradas, e os artefatos de execução são gerados em diretórios de saída.

## 4. Descrição do dataset

O dataset é construído a partir de obras literárias em português disponíveis em acervos públicos, com destaque para fontes textuais do Project Gutenberg e catálogo local em `data/external/public_domain_sources.csv`.

A unidade de classificação é o trecho textual, não a obra completa. Isso significa que uma mesma obra pode contribuir com exemplos de diferentes níveis de complexidade, conforme as características de cada janela extraída.

As colunas esperadas no dataset gerado são:

| coluna | finalidade |
| --- | --- |
| `id` | identificador do trecho |
| `texto` | trecho textual usado pelo modelo |
| `target` | rótulo operacional: `baixa`, `media` ou `alta` |
| `fonte` | acervo de origem |
| `autor` | autor da obra |
| `obra` | obra de referência |
| `capitulo` | marcador aproximado da janela textual |
| `tipo_trecho` | tipo operacional do trecho |
| `criterio_rotulo` | resumo do critério de rotulagem |
| `origem_url` | URL textual de origem |
| `janela_inicio` | índice aproximado inicial da janela |
| `janela_fim` | índice aproximado final da janela |

Para o pipeline principal, as colunas obrigatórias são `id`, `texto` e `target`. As demais colunas documentam a origem e a construção dos exemplos.

## 5. Critério de rotulagem e observação sobre rótulos heurísticos

Os rótulos do dataset são heurísticos e operacionais. Eles representam uma aproximação construída a partir de indicadores linguísticos e discursivos, e não uma avaliação humana definitiva de complexidade cognitiva ou valor literário.

A rotulagem considera três classes:

| classe | interpretação operacional |
| --- | --- |
| `baixa` | trechos com estrutura mais direta, menor densidade abstrata e menor exigência inferencial |
| `media` | trechos com algum grau de reflexão, memória, tensão implícita ou articulação interpretativa moderada |
| `alta` | trechos com maior densidade abstrata, ambiguidade, simbolismo, deslocamento temporal ou maior exigência inferencial |

O processo reduz o risco de o rótulo ser determinado apenas pelo comprimento do texto. Para isso, o escore de complexidade é dividido em tercis dentro de faixas de tamanho. Assim, trechos de extensão semelhante podem receber rótulos diferentes conforme seus sinais linguísticos e discursivos.

A interpretação dos resultados deve considerar que o modelo aprende padrões associados à rubrica adotada. Logo, o desempenho quantitativo mede aderência aos rótulos operacionais do experimento, não validação absoluta de complexidade literária.

## 6. Pipeline experimental

O fluxo experimental previsto é composto pelas seguintes etapas:

1. download das fontes textuais públicas;
2. limpeza textual e remoção de ruído editorial;
3. segmentação das obras em janelas controladas;
4. cálculo de indicadores linguísticos e discursivos;
5. rotulagem heurística por escore operacional;
6. auditoria do dataset gerado;
7. preparação do dataset em treino, validação e teste;
8. verificação de vazamento de dados;
9. extração de features linguísticas e vetoriais;
10. validação cruzada;
11. busca de hiperparâmetros;
12. treinamento do melhor modelo;
13. avaliação final em conjunto de teste;
14. geração de relatórios, figuras e artefatos de explicabilidade;
15. disponibilização de inferência local em texto único ou arquivo CSV.

A execução completa do pipeline principal é feita por:

```powershell
python scripts/run_pipeline.py
```

A validação dos artefatos finais pode ser feita por:

```powershell
python scripts/validate_artifacts.py --profile complete
```

## 7. Extração de métricas linguísticas

A construção do dataset e o pipeline de features consideram métricas linguísticas interpretáveis. Entre os indicadores documentados estão:

- número de palavras;
- número de sentenças;
- média de palavras por sentença;
- diversidade lexical aproximada;
- proporção de palavras longas;
- frequência de conectivos e marcadores discursivos;
- frequência de marcadores de subordinação;
- densidade de pontuação;
- frequência de marcadores abstratos;
- frequência de marcadores de ambiguidade;
- frequência de marcadores temporais;
- presença aproximada de diálogo.

Essas métricas cumprem duas funções. Primeiro, contribuem para a construção dos rótulos heurísticos. Segundo, podem ser usadas como features interpretáveis no treinamento, permitindo comparar modelos que usam apenas indicadores linguísticos com modelos que usam representações vetoriais baseadas no texto.

## 8. Representações textuais e features

A configuração principal prevê as seguintes representações:

| representação | descrição |
| --- | --- |
| `linguistic_metrics` | conjunto de métricas linguísticas interpretáveis extraídas do texto |
| `tfidf_word` | vetorização TF-IDF por palavras, com n-gramas de 1 a 2 termos |
| `tfidf_char` | vetorização TF-IDF por caracteres, com n-gramas de 3 a 5 caracteres |
| `tfidf_word_plus_linguistic_metrics` | combinação entre TF-IDF por palavras e métricas linguísticas |

Na configuração principal, o TF-IDF por palavras usa até 5000 features, e o TF-IDF por caracteres usa até 3000 features. A configuração smoke de CI reduz o espaço de features para acelerar validações e evitar dependência do dataset oficial.

## 9. Modelos avaliados

A configuração principal prevê a avaliação de três famílias de modelos supervisionados:

| modelo | observação técnica |
| --- | --- |
| `logistic_regression` | modelo linear, adequado para análise por coeficientes quando combinado a TF-IDF |
| `linear_svm` | classificador linear robusto para texto esparso |
| `random_forest` | modelo não linear baseado em árvores, útil como comparação com representações tabulares |

A métrica de seleção definida é `f1_macro`, apropriada para comparação entre classes quando se deseja tratar os rótulos com peso equivalente, independentemente da distribuição observada.

## 10. Validação cruzada

A validação cruzada está habilitada na configuração principal com `StratifiedKFold`, 5 divisões, embaralhamento e `random_state` fixado em 42.

As métricas configuradas para validação cruzada são:

- `accuracy`;
- `precision_macro`;
- `recall_macro`;
- `f1_macro`;
- `f1_weighted`;
- `balanced_accuracy`.

A validação estratificada é importante porque preserva a proporção das classes nas divisões, reduzindo variações artificiais causadas por partições desequilibradas. A análise de validação cruzada deve ser usada para observar estabilidade do desempenho, não apenas o maior valor médio.

## 11. Busca de hiperparâmetros

A busca de hiperparâmetros está habilitada na configuração principal com estratégia de grade (`grid`) e métrica de refit `f1_macro`.

Espaços configurados:

| modelo | hiperparâmetros avaliados |
| --- | --- |
| `logistic_regression` | `C`, `max_iter`, `class_weight` |
| `linear_svm` | `C`, `class_weight` |
| `random_forest` | `n_estimators`, `max_depth`, `min_samples_split`, `class_weight` |

A configuração smoke usada para CI desativa a busca de hiperparâmetros e restringe o treinamento a uma regressão logística com TF-IDF por palavras. Essa escolha torna a validação rápida e reprodutível, sem depender do dataset completo.

## 12. Avaliação final em teste

A configuração principal define divisão do dataset em treino, validação e teste na proporção 70%, 15% e 15%, com estratificação habilitada.

O conjunto de teste deve ser usado apenas para avaliação final do modelo selecionado. A escolha do modelo e dos hiperparâmetros deve ocorrer anteriormente, por validação cruzada e busca configurada, para reduzir viés de seleção sobre o teste.

Até a presente revisão documental, não foram encontrados artefatos numéricos finais versionados ou anexados com resultados consolidados de teste, como `classification_report`, matriz de confusão final ou resumo do melhor experimento. Por esse motivo, este relatório não declara valores de acurácia, precisão, revocação ou F1. Esses números devem ser obtidos a partir dos artefatos gerados após a execução local do pipeline.

## 13. Métricas principais

As métricas principais do experimento são:

| métrica | finalidade |
| --- | --- |
| `f1_macro` | métrica principal de seleção, trata as classes com peso equivalente |
| `precision_macro` | mede a precisão média entre classes |
| `recall_macro` | mede a cobertura média entre classes |
| `f1_weighted` | considera o suporte de cada classe |
| `balanced_accuracy` | avalia desempenho ajustado ao equilíbrio entre classes |
| `accuracy` | mede a proporção geral de acertos |

Como os rótulos são heurísticos, a análise das métricas deve ser acompanhada de cautela metodológica. Resultados altos podem indicar aderência do modelo à rubrica operacional, mas não provam que o sistema mede complexidade cognitiva em sentido amplo ou definitivo.

## 14. Análise de resultados

A análise dos resultados deve considerar três dimensões:

### 14.1 Desempenho preditivo

O desempenho deve ser analisado em validação cruzada e em teste final. A comparação entre `f1_macro`, `balanced_accuracy` e métricas por classe é mais informativa do que a acurácia isolada, especialmente em tarefas multiclasses.

### 14.2 Estabilidade experimental

A validação cruzada permite observar se o modelo mantém desempenho consistente em diferentes partições. Grandes variações entre folds podem indicar sensibilidade ao particionamento, dataset pequeno, concentração de autores/obras ou sinais muito específicos.

### 14.3 Natureza dos sinais aprendidos

A comparação entre representações é central para a interpretação técnica:

- bom desempenho com `linguistic_metrics` sugere que os indicadores interpretáveis capturam parte relevante da rubrica;
- melhora com `tfidf_word` ou `tfidf_char` indica que padrões lexicais e estilísticos específicos contribuem para a classificação;
- diferença grande entre validação e teste pode sugerir sobreajuste;
- resultados muito elevados devem ser analisados junto aos relatórios de vazamento e duplicidade.

Como não há métricas finais anexadas nesta revisão, a conclusão quantitativa deve ser preenchida somente após consulta aos artefatos gerados pelo pipeline.

## 15. Explicabilidade por TF-IDF e coeficientes

A explicabilidade está habilitada na configuração principal. O projeto prevê geração de artefatos globais e locais, incluindo termos TF-IDF mais relevantes por classe quando o modelo selecionado permite interpretação por coeficientes.

Esse tipo de explicabilidade é especialmente adequado para modelos lineares, como regressão logística e SVM linear, pois os coeficientes ajudam a identificar quais termos ou n-gramas mais contribuem para cada classe. Para modelos não lineares, como Random Forest, a interpretação por coeficientes diretos não se aplica da mesma forma; nesse caso, a explicabilidade deve usar recursos compatíveis com o modelo disponível.

A análise dos termos mais associados a cada classe deve ser feita com cuidado. Termos explicativos não devem ser interpretados isoladamente como prova de complexidade cognitiva, mas como indícios estatísticos aprendidos a partir da rubrica e do corpus.

## 16. Verificação de vazamento de dados

A configuração principal inclui verificações de vazamento, com os seguintes controles:

- sobreposição de IDs entre partições;
- sobreposição exata de textos;
- sobreposição de textos normalizados;
- detecção de quase duplicatas;
- verificação de colunas com aparência de alvo;
- falha em caso de vazamento crítico.

A configuração define limiar de quase duplicidade em `0.92`. Esse ponto é relevante porque o dataset é criado por janelas textuais com sobreposição moderada. Sem controle de duplicatas ou quase duplicatas, trechos muito semelhantes poderiam aparecer em partições diferentes e inflar artificialmente as métricas.

A auditoria do dataset também verifica duplicatas, distribuição por classe, médias de tamanho, concentração por autor/obra, ruído editorial e sentenças repetidas. Esses relatórios devem ser consultados antes de considerar uma execução como válida.

## 17. Artefatos gerados

Os principais artefatos previstos são:

| etapa | artefatos esperados |
| --- | --- |
| download de fontes | `outputs/metrics/dataset_download_manifest.json` |
| construção do dataset | `data/raw/dataset_candidate.csv`, `data/raw/dataset.csv`, `docs/dataset_card.md` |
| auditoria do dataset | `outputs/metrics/dataset_quality_report.json`, `outputs/reports/dataset_quality_report.md` |
| preparação e features | arquivos em `data/processed` e `data/processed/features`, conforme configuração |
| treinamento | modelos e metadados em `outputs/models` e/ou `outputs/latest/models` |
| avaliação | métricas e relatórios em `outputs/metrics`, `outputs/reports` e `outputs/figures` |
| execuções versionadas | `outputs/runs/<run_id>/` e ponteiro em `outputs/latest/run_id.txt` |
| explicabilidade | artefatos em `outputs/latest/explainability` e figuras em `outputs/figures` |
| inferência local | predições em `outputs/latest/predictions/predictions.csv`, quando usado `predict_file.py` |

## 18. Limitações e ameaças à validade

As principais limitações metodológicas são:

1. **Rotulagem heurística**: os rótulos são gerados por uma rubrica operacional e não por consenso humano especializado.
2. **Dependência dos indicadores definidos**: o modelo aprende padrões alinhados aos critérios de construção do dataset.
3. **Viés de fonte textual**: obras de domínio público podem concentrar estilos, períodos literários e normas linguísticas específicas.
4. **Segmentação por janelas**: a análise por trecho pode perder contexto narrativo mais amplo da obra.
5. **Sobreposição textual**: janelas com sobreposição exigem controle rigoroso de vazamento e quase duplicidade.
6. **Interpretação de métricas**: bom desempenho quantitativo não deve ser interpretado como validação completa de complexidade cognitiva.
7. **Ausência de métricas finais versionadas nesta revisão**: sem artefatos finais anexados, não é possível declarar resultados numéricos consolidados neste documento.

Essas limitações não invalidam o experimento, mas delimitam corretamente o alcance das conclusões.

## 19. Reprodutibilidade

A reprodutibilidade é apoiada por:

- configurações YAML versionáveis;
- `random_state` fixado em 42;
- catálogo de fontes públicas;
- scripts separados para download, construção, auditoria e promoção do dataset;
- pipeline executável por linha de comando;
- validação de artefatos finais;
- criação de ponteiro para a última execução;
- salvamento de snapshot de configuração, ambiente e fingerprint do dataset.

Fluxo recomendado:

```powershell
python scripts/dataset_download_sources.py
python scripts/dataset_build.py
python scripts/dataset_audit.py
python scripts/run_pipeline.py
python scripts/validate_artifacts.py --profile complete
python scripts/predict_text.py --text "O narrador reorganiza lembranças, símbolos e ambiguidades para construir uma interpretação instável dos acontecimentos."
python -m pytest
ruff check .
```

Para CI, a configuração smoke deve usar dataset leve de fixture e execução reduzida, evitando dependência do dataset oficial e de artefatos grandes.

## 20. Conclusão técnica

O projeto apresenta uma estrutura experimental adequada para investigar classificação automática de níveis operacionais de complexidade cognitiva em trechos literários em português. A arquitetura cobre construção reprodutível de dataset, extração de métricas linguísticas, representações TF-IDF, avaliação de modelos supervisionados, validação cruzada, busca de hiperparâmetros, detecção de vazamento, explicabilidade e inferência local.

A principal força técnica do projeto está na combinação entre rastreabilidade, separação clara de etapas e preocupação explícita com limitações metodológicas. A principal cautela é que os rótulos representam uma aproximação heurística, portanto os resultados devem ser interpretados como aderência a uma definição operacional de complexidade, não como substituição da avaliação humana.
