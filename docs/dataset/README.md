# Dataset - Guia rápido
> Relatório técnico final: consulte [`docs/relatorio_execucao_final.md`](../relatorio_execucao_final.md) para a síntese acadêmica/técnica da execução experimental do projeto.


Este diretório documenta a construção do dataset utilizado.

O dataset é gerado localmente a partir de obras literárias em português disponíveis em acervos públicos. O repositório versiona o catálogo de fontes, os scripts de construção e a documentação metodológica. O arquivo final `data/raw/dataset.csv` é produzido pelo processo automatizado e pode ser recriado a partir das fontes registradas.

## Objetivo

Construir um dataset metodologicamente consistente para apoiar a classificação automática de complexidade cognitiva em textos literários em português.

A construção busca favorecer exemplos com diferentes níveis de exigência linguística e interpretativa, evitando que a classificação dependa de sinais superficiais ou artificiais.

O processo prioriza:

- uso de fontes públicas e rastreáveis;
- preservação de metadados de origem;
- segmentação controlada dos textos;
- rotulagem operacional baseada em indicadores linguísticos e discursivos;
- auditoria de duplicatas, distribuição de classes e equilíbrio textual;
- reprodutibilidade por meio de scripts, configuração e catálogo de fontes.

## Arquivos principais

```text
configs/dataset.yaml
data/external/public_domain_sources.csv
scripts/dataset_download_sources.py
scripts/dataset_build.py
scripts/dataset_audit.py
scripts/dataset_promote.py
docs/dataset/
```

## Fluxo recomendado

```powershell
python scripts/dataset_download_sources.py
python scripts/dataset_build.py
python scripts/dataset_audit.py
```

O comando de construção gera:

```text
data/raw/dataset_candidate.csv
data/raw/dataset.csv
docs/dataset_card.md
```

O dataset principal é gerado em:

```text
data/raw/dataset.csv
```

## Catálogo de fontes

O arquivo abaixo:

```text
data/external/public_domain_sources.csv
```

Registra as obras usadas como fonte do corpus, incluindo autor, obra, URL textual, fonte, gênero, período literário e observações metodológicas.

Esse arquivo não contém os textos brutos completos, apenas funciona como catálogo reprodutível para que os scripts baixem as obras e gerem o dataset localmente.

## Dataset gerado

Por padrão, o dataset final não é versionado no GitHub, pois o mesmo é recriado com os scripts do projeto.

Colunas esperadas no dataset gerado:

```text
id
texto
target
fonte
autor
obra
capitulo
tipo_trecho
criterio_rotulo
origem_url
janela_inicio
janela_fim
```

Apenas as colunas `id`, `texto` e `target` são obrigatórias para o pipeline principal de treinamento. As demais colunas documentam origem e construção do trecho.

## Execução do pipeline após gerar o dataset

Depois de gerar e auditar o dataset, execute:

```powershell
python scripts/run_pipeline.py
python scripts/validate_artifacts.py --profile complete
python scripts/predict_text.py --text "O narrador organiza memórias, imagens simbólicas e ambiguidades que exigem interpretação cuidadosa."
python -m pytest
ruff check .
```

## Observação metodológica

Os rótulos do dataset são operacionais e heurísticos. Eles representam uma estimativa construída a partir de indicadores linguísticos e discursivos, não uma avaliação humana definitiva de valor literário.

A interpretação dos resultados deve considerar essa natureza operacional, especialmente nas análises acadêmicas sobre complexidade cognitiva.
