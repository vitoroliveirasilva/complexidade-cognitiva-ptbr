# Metodologia de construção do dataset

## Visão geral

O dataset é construído a partir de obras literárias em português disponíveis em acervos públicos, principalmente Project Gutenberg. O processo é automatizado por scripts locais que baixam, limpam, segmentam, rotulam e auditam os textos.

A unidade de classificação não é a obra inteira, mas o trecho textual extraído. Assim, uma mesma obra pode gerar trechos classificados em diferentes níveis de complexidade cognitiva.

## Fontes

As fontes são registradas no arquivo:

```text
data/external/public_domain_sources.csv
```

Esse arquivo funciona como catálogo bibliográfico e operacional. Cada linha representa uma fonte textual a ser baixada e processada.

Campos principais:

| campo | finalidade |
| --- | --- |
| `slug` | identificador curto e estável da fonte |
| `title` | título exibido da obra |
| `author` | autor |
| `work` | obra de referência |
| `source` | acervo de origem |
| `source_id` | identificador da obra no acervo, quando disponível |
| `record_url` | página bibliográfica da obra |
| `text_url` / `url` | texto bruto usado pelo script |
| `language` | idioma |
| `genre` | gênero textual |
| `original_year` | ano original aproximado |
| `literary_period` | período ou movimento literário de referência |
| `expected_complexity_profile` | perfil textual esperado da obra, apenas orientativo |
| `dataset_priority` | prioridade de uso no corpus |
| `methodological_notes` | observações sobre utilidade metodológica |
| `license_notes` | observações sobre domínio público/licença |

A coluna `expected_complexity_profile` não define o rótulo final dos trechos. Ela apenas orienta a diversidade do corpus.

## Download

O download das obras é feito por:

```powershell
python scripts/dataset_download_sources.py
```

O script lê o catálogo de fontes, baixa os textos indicados em `url` e salva os arquivos em uma pasta local de textos brutos.

Os textos baixados são artefatos reproduzíveis e não são versionados no Git.

## Limpeza textual

Durante a construção do dataset, o processamento remove ou reduz, quando possível:

- cabeçalhos e rodapés do Project Gutenberg;
- trechos de licença;
- linhas editoriais;
- informações técnicas de transcrição;
- índices e sumários;
- marcas não literárias;
- blocos textuais sem proporção mínima de texto alfabético.

Essa etapa busca preservar o conteúdo literário e reduzir interferências editoriais no treinamento.

## Segmentação

As obras são segmentadas em janelas textuais controladas.

Configuração padrão:

| parâmetro | valor |
| --- | --- |
| mínimo de palavras | 80 |
| alvo de palavras | 160 |
| máximo de palavras | 220 |
| sobreposição | moderada |

A sobreposição permite ampliar o número de registros sem duplicar textos completos. A auditoria posterior verifica duplicatas e repetições relevantes.

## Cálculo de indicadores

Para cada trecho, o script calcula indicadores linguísticos e discursivos, incluindo:

- número de palavras;
- número de sentenças;
- média de palavras por sentença;
- diversidade lexical aproximada;
- proporção de palavras longas;
- frequência de conectivos e marcadores discursivos;
- marcadores de subordinação;
- densidade de pontuação;
- marcadores de abstração;
- marcadores de ambiguidade;
- marcadores temporais;
- presença aproximada de diálogo.

Esses indicadores compõem um escore operacional de complexidade.

## Rotulagem

A rotulagem é feita por uma rubrica operacional. O script calcula um escore linguístico para cada trecho e distribui os exemplos em três classes:

```text
baixa
media
alta
```

Para reduzir a dependência do comprimento textual, os escores são divididos em tercis dentro de faixas de tamanho. Assim, a classe não é definida diretamente pelo número de palavras.

O rótulo final é armazenado na coluna:

```text
target
```

A justificativa resumida do processo de rotulagem é registrada em:

```text
criterio_rotulo
```

## Balanceamento

A construção limita a concentração de exemplos por obra e por classe. O objetivo é evitar que uma única obra, autor ou estilo domine o dataset.

O processo também busca manter distribuição equilibrada entre as classes.

## Auditoria

Após a geração, o dataset deve ser auditado com:

```powershell
python scripts/dataset_audit.py
```

A auditoria verifica:

- total de registros;
- distribuição por classe;
- duplicatas de ID;
- duplicatas exatas de texto;
- duplicatas normalizadas;
- médias de tamanho por classe;
- distribuição por faixas de tamanho;
- concentração por autor;
- concentração por obra;
- presença de ruído editorial;
- sentenças repetidas com maior frequência.

Relatórios gerados:

```text
outputs/metrics/dataset_quality_report.json
outputs/reports/dataset_quality_report.md
```

## Uso no treinamento

Após a geração e auditoria do dataset, o pipeline principal usa:

```text
data/raw/dataset.csv
```

A execução completa é feita com:

```powershell
python scripts/run_pipeline.py
```

O pipeline realiza preparação, detecção de vazamento, extração de features, validação cruzada, busca de hiperparâmetros, treinamento, avaliação, explicabilidade e geração de relatórios.

## Reprodutibilidade

Para reproduzir o dataset, são necessários:

```text
configs/dataset.yaml
data/external/public_domain_sources.csv
scripts/dataset_download_sources.py
scripts/dataset_build.py
scripts/dataset_audit.py
```

O dataset final pode ser recriado localmente a qualquer momento a partir das fontes públicas registradas.
