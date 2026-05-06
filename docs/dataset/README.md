# Dataset - Guia rápido

Este pacote gera uma versão do dataset principal a partir de textos literários em português de domínio público.

## Objetivo

Construir um dataset metodologicamente consistente, capaz de representar diferentes níveis de complexidade cognitiva sem depender de sinais superficiais ou artificiais.

A construção do dataset busca evitar, especialmente:

- Classes diferenciadas predominantemente pelo tamanho dos textos;
- Padrões repetitivos, frases recorrentes ou templates por classe influenciem indevidamente o aprendizado do modelo;
- Atalhos estatísticos em vez de características linguísticas e interpretativas mais relevantes aprendidos pelo classificador;
- Resultados experimentais inflados por vieses de construção do corpus.

Dessa forma, o dataset deve favorecer uma avaliação mais realista, equilibrada e defensável da tarefa de classificação automática de complexidade cognitiva em textos literários.

## Fluxo recomendado

```powershell
python scripts/dataset_download_sources.py
python scripts/dataset_build.py
python scripts/dataset_audit.py
```

O `dataset_build.py` gera:

```text
data/raw/dataset_candidate.csv
data/raw/dataset.csv
docs/dataset_card.md
```

Depois, rode o pipeline normal:

```powershell
python scripts/run_pipeline.py
python scripts/validate_artifacts.py --profile complete
python scripts/predict_text.py --text "O narrador organiza memórias, imagens simbólicas e ambiguidades que exigem interpretação cuidadosa."
```
