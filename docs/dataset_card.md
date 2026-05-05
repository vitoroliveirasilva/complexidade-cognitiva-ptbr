# Dataset Card - Complexidade Cognitiva PT-BR

## Identificação

- **Nome do arquivo:** `data/raw/dataset.csv`
- **Projeto:** `complexidade-cognitiva-ptbr`
- **Versão do projeto:** `0.1.0`
- **Tipo de tarefa:** classificação supervisionada multiclasse
- **Idioma:** português brasileiro
- **Domínio:** textos literários curtos e trechos narrativos voltados à avaliação experimental de complexidade cognitiva
- **Classes:** `baixa`, `media` e `alta`
- **Codificação recomendada:** UTF-8 ou UTF-8 com BOM

## Finalidade do dataset

Este dataset foi organizado para apoiar a validação inicial de um pipeline acadêmico de Processamento de Linguagem Natural e Aprendizado de Máquina para estimação automática da complexidade cognitiva em textos literários em português.

A base permite exercitar, de forma reprodutível, as etapas de validação do corpus, limpeza textual, extração de métricas linguísticas interpretáveis, treinamento supervisionado, comparação de experimentos e avaliação final.

## Composição

- **Total de registros:** 9000
- **Colunas:** `id`, `texto`, `target`
- **Distribuição de classes:**

- `alta`: 3000 exemplos
- `baixa`: 3000 exemplos
- `media`: 3000 exemplos

A distribuição balanceada foi adotada para reduzir o impacto de desbalanceamento durante os experimentos iniciais e facilitar a comparação entre modelos.

## Descrição das colunas

| Coluna | Tipo esperado | Obrigatória | Descrição |
| --- | --- | --- | --- |
| `id` | inteiro ou texto identificador | Sim | Identificador único de cada texto |
| `texto` | texto | Sim | Conteúdo textual analisado pelo pipeline |
| `target` | categoria | Sim | Classe de complexidade cognitiva atribuída ao texto |

## Estatísticas textuais observadas

| Métrica | Valor |
| --- | ---: |
| Mínimo de palavras | 10 |
| Média de palavras | 35.65 |
| Mediana de palavras | 39.0 |
| Máximo de palavras | 73 |
| Mínimo de caracteres | 48 |
| Média de caracteres | 229.56 |
| Mediana de caracteres | 258.0 |
| Máximo de caracteres | 474 |

## Critérios de validação aplicados pelo pipeline

Durante a etapa de preparação, o pipeline verifica se:

- o arquivo existe no caminho configurado;
- o arquivo está em formato CSV;
- as colunas obrigatórias estão presentes;
- os campos `id`, `texto` e `target` não possuem valores nulos;
- os campos obrigatórios não possuem valores vazios;
- a coluna `id` não possui duplicidade;
- existem pelo menos duas classes distintas;
- os textos não ficam vazios após a limpeza;
- a divisão entre treino, validação e teste gera subconjuntos não vazios.

Na execução oficial registrada, não foram encontrados valores nulos, campos obrigatórios vazios, textos duplicados exatos ou avisos relevantes.

## Divisão experimental

A divisão configurada foi:

| Subconjunto | Proporção | Registros | Distribuição por classe |
| --- | ---: | ---: | --- |
| Treino | 70% | 6300 | 2100 por classe |
| Validação | 15% | 1350 | 450 por classe |
| Teste | 15% | 1350 | 450 por classe |

A estratificação foi aplicada com sucesso, mantendo a distribuição balanceada das classes nos três subconjuntos.

## Limitações conhecidas

O dataset utilizado nesta etapa deve ser interpretado como base sintética, curada ou controlada para validação experimental inicial do pipeline. Métricas muito elevadas, especialmente resultados perfeitos, indicam que os padrões entre classes podem estar fortemente separáveis no conjunto atual.

Assim, os resultados devem ser apresentados no TCC como evidência de funcionamento e reprodutibilidade do método, não como comprovação definitiva de generalização para qualquer texto literário real.

## Uso recomendado

Este dataset é adequado para:

- validar a arquitetura do pipeline;
- testar o fluxo completo de treino e avaliação;
- comparar representações e modelos em ambiente controlado;
- gerar artefatos acadêmicos iniciais;
- apoiar a escrita do capítulo de Resultados e Discussão.

Para trabalhos futuros, recomenda-se complementar esta base com corpus real rotulado, revisão humana dos rótulos e validação externa em textos não gerados ou não curados pela mesma estratégia de construção.

## Considerações éticas e de reprodutibilidade

Caso o dataset oficial do TCC seja mantido no repositório, ele deve conter apenas textos autorizados, sintéticos, de domínio público ou adequadamente licenciados. Datasets privados, sensíveis ou sujeitos a restrição autoral não devem ser versionados publicamente.
