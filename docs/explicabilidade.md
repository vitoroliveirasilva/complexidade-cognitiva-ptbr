# Explicabilidade e Interpretação dos Resultados

## Finalidade

A explicabilidade neste projeto tem a função de transformar o resultado computacional em análise compreensível com o objetivo de não apenas informar que o modelo obteve bom desempenho, mas discutir quais sinais textuais podem ter contribuído para a separação entre os níveis de complexidade cognitiva.

## Famílias de representação avaliadas

O pipeline avaliou duas famílias principais de representação:

1. **Métricas linguísticas interpretáveis:** atributos calculados diretamente a partir dos textos.
2. **Representações TF-IDF:** vetores estatísticos baseados na frequência ponderada de termos ou caracteres.

A combinação dessas famílias permite discutir tanto desempenho preditivo quanto sinais textuais observáveis.

## Experimento selecionado

| Campo | Valor |
| --- | --- |
| Experimento | `tfidf_word__logistic_regression` |
| Representação | `tfidf_word` |
| Modelo | `logistic_regression` |
| Métrica de seleção | `f1_macro` |
| Pontuação em validação | `1.000000` |

O experimento selecionado usou TF-IDF de palavras com Regressão Logística. Isso indica que padrões lexicais foram altamente informativos para distinguir as classes no conjunto experimental.

## Interpretação do TF-IDF

O TF-IDF representa os textos por termos ponderados conforme sua frequência no documento e sua relevância relativa no corpus. O desempenho perfeito dessa representação sugere que o vocabulário dos textos contém sinais muito fortes de separação entre `baixa`, `media` e `alta` complexidade.

Em termos acadêmicos, esse resultado pode indicar que:

- textos de baixa complexidade usam vocabulário mais direto e cotidiano;
- textos de média complexidade apresentam maior elaboração descritiva;
- textos de alta complexidade incorporam vocabulário mais abstrato, denso ou interpretativo.

No entanto, essa interpretação deve ser cuidadosa. O modelo pode estar aprendendo padrões lexicais específicos do dataset, e não uma noção universal de complexidade cognitiva.

## Papel das métricas linguísticas

Embora o melhor desempenho tenha sido obtido com TF-IDF, os experimentos baseados apenas em métricas linguísticas também obtiveram desempenho muito alto. Isso mostra que atributos interpretáveis capturaram parte relevante da separação entre classes.

As métricas consideradas foram:

- `num_caracteres`: Quantidade total de caracteres do texto limpo.
- `num_palavras`: Quantidade total de tokens lexicais identificados no texto.
- `num_sentencas`: Quantidade aproximada de sentenças identificadas por pontuação final.
- `media_palavras_por_sentenca`: Média de palavras por sentença.
- `media_caracteres_por_palavra`: Média de caracteres por palavra.
- `maior_sentenca_palavras`: Comprimento da maior sentença em número de palavras.
- `variancia_tamanho_sentencas`: Variância do número de palavras por sentença.
- `palavras_unicas`: Quantidade de palavras únicas em caixa baixa.
- `type_token_ratio`: Razão entre palavras únicas e total de palavras.
- `diversidade_lexical`: Medida de diversidade lexical equivalente ao type-token ratio nesta versão.
- `razao_palavras_longas`: Proporção de palavras com tamanho maior ou igual ao limite configurado.
- `razao_palavras_repetidas`: Proporção de tokens que aparecem mais de uma vez no texto.
- `razao_pontuacao`: Proporção de caracteres de pontuação em relação ao total de caracteres.
- `razao_numeros`: Proporção de tokens numéricos em relação ao total de palavras/tokens.
- `frequencia_conectivos`: Proporção de conectivos simples em relação ao total de palavras.
- `frequencia_marcadores_subordinacao`: Proporção de marcadores de subordinação em relação ao total de palavras.
- `densidade_lexical_aproximada`: Proporção aproximada de palavras de conteúdo em relação ao total de palavras.

Essas métricas apoiam a discussão acadêmica porque possuem significado textual direto. Elas permitem relacionar os resultados a aspectos como extensão, densidade lexical, estrutura sentencial e presença de elementos conectivos ou subordinativos.

## Explicabilidade direta e indireta

A explicabilidade direta vem das métricas linguísticas, pois cada coluna possui interpretação textual clara. A explicabilidade indireta vem do TF-IDF, pois o modelo aprende pesos associados a termos e padrões lexicais.

## Conclusão interpretável

Os resultados indicam que o pipeline conseguiu capturar sinais textuais fortemente associados às classes de complexidade no dataset utilizado. As métricas linguísticas oferecem suporte interpretável, enquanto o TF-IDF apresentou maior desempenho preditivo.

Essa combinação fortalece a proposta do TCC ao unir desempenho computacional e possibilidade de leitura crítica dos atributos textuais.
