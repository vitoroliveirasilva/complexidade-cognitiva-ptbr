# Rubrica operacional de rotulagem

A rotulagem do dataset é heurística e operacional. Ela busca estimar níveis de complexidade cognitiva em trechos literários com base em indicadores linguísticos e discursivos.

Essa rubrica não substitui avaliação humana especializada. Seu objetivo é permitir a construção reprodutível de um corpus rotulado para treinamento e avaliação de modelos de aprendizado de máquina.

## Unidade de análise

A unidade de análise é o trecho textual extraído da obra, não a obra completa.

Uma mesma obra pode contribuir com trechos classificados como `baixa`, `media` ou `alta`, de acordo com as características linguísticas e interpretativas de cada janela textual.

## Baixa complexidade

Trechos de baixa complexidade tendem a apresentar menor densidade interpretativa e estrutura mais direta.

Sinais esperados:

- sequência narrativa mais linear;
- vocabulário mais concreto;
- menor presença de abstração;
- menor ambiguidade;
- menor exigência inferencial;
- períodos mais simples;
- menor articulação simbólica;
- menor necessidade de reconstrução de sentidos implícitos.

Exemplos de perfil textual:

- descrição direta de uma ação;
- cena cotidiana com baixa tensão interpretativa;
- sequência factual de acontecimentos;
- trecho com pouca sobreposição temporal ou simbólica.

## Média complexidade

Trechos de média complexidade apresentam algum grau de reflexão, memória, descrição interpretativa ou tensão parcialmente implícita.

Sinais esperados:

- presença moderada de conectivos;
- alguma subordinação;
- vocabulário mais variado;
- relação entre cena concreta e interpretação;
- presença de memória ou passagem do tempo;
- conflito emocional moderado;
- necessidade intermediária de inferência.

Exemplos de perfil textual:

- uma descrição que sugere estado emocional;
- uma lembrança associada a objetos ou espaços;
- uma cena cotidiana com tensão implícita;
- um diálogo ou acontecimento que exige interpretação moderada.

## Alta complexidade

Trechos de alta complexidade tendem a exigir maior esforço de integração, inferência e interpretação.

Sinais esperados:

- maior densidade abstrata;
- presença de ambiguidade;
- simbolismo;
- metáforas relevantes;
- deslocamento de perspectiva;
- instabilidade de memória;
- narrador pouco confiável ou ambíguo;
- sobreposição temporal;
- reflexão moral, filosófica ou existencial;
- sentidos implícitos que precisam ser reconstruídos pelo leitor.

Exemplos de perfil textual:

- trecho em que a memória altera a interpretação do presente;
- cena em que objetos concretos ganham valor simbólico;
- narrativa com lacunas, silêncio ou ambiguidade;
- reflexão que articula culpa, desejo, julgamento, identidade ou linguagem.

## Indicadores usados pelo script

A rotulagem considera um escore composto por indicadores como:

- média de palavras por sentença;
- proporção de palavras longas;
- diversidade lexical;
- frequência de marcadores de subordinação;
- frequência de marcadores discursivos;
- densidade de pontuação;
- frequência de marcadores abstratos;
- frequência de marcadores de ambiguidade;
- frequência de marcadores temporais;
- penalização leve para trechos predominantemente dialogais, quando aplicável.

## Controle contra viés de tamanho

O rótulo não é definido diretamente pelo tamanho do trecho.

Para reduzir esse risco, o script calcula o escore de complexidade e divide os resultados em tercis dentro de faixas de tamanho. Com isso, trechos de comprimentos semelhantes podem ser classificados em níveis diferentes, dependendo de seus sinais linguísticos e discursivos.

## Interpretação acadêmica

Os rótulos devem ser entendidos como uma aproximação operacional de complexidade cognitiva textual.

A análise dos resultados deve considerar que:

- o modelo aprende a partir da rubrica usada;
- a rubrica aproxima dimensões linguísticas e discursivas;
- a classificação não representa julgamento literário definitivo;
- trechos curtos ou muito específicos podem exigir interpretação manual complementar;
- a avaliação quantitativa deve ser acompanhada de análise qualitativa.
