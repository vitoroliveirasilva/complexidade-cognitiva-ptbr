# Rubrica operacional de rotulagem - Dataset

A rotulagem do dataset é heurística e operacional, portanto ela não substitui avaliação humana especializada.

## Baixa complexidade

Trechos com menor densidade interpretativa, menor presença de ambiguidade, menor exigência inferencial e estrutura mais direta.

Sinais esperados:

- menor média de palavras por sentença;
- menor frequência de marcadores abstratos;
- menor presença de ambiguidade/simbolismo;
- narrativa mais concreta ou descritiva.

## Média complexidade

Trechos com reflexão moderada, memória, relações emocionais, descrição com algum valor interpretativo e tensão parcialmente implícita.

Sinais esperados:

- presença moderada de conectivos e subordinação;
- vocabulário um pouco mais abstrato;
- algum deslocamento temporal ou emocional;
- necessidade moderada de inferência.

## Alta complexidade

Trechos com maior densidade abstrata, ambiguidade, simbolismo, deslocamento de perspectiva, instabilidade de memória ou maior necessidade de inferência.

Sinais esperados:

- maior média de palavras por sentença;
- maior frequência de marcadores abstratos;
- maior presença de ambiguidade, metáfora, silêncio, perspectiva, inferência;
- maior articulação discursiva.

## Controle contra viés de tamanho

O dataset não define classe diretamente por tamanho. O escore é dividido em tercis dentro de faixas de tamanho, reduzindo o risco de `baixa = curto` e `alta = longo`.
