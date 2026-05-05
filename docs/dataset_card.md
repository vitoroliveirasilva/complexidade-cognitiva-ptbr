# Dataset Card - Complexidade Cognitiva PT-BR

## Identificação

- **Arquivo:** `data/raw/dataset.csv`
- **Projeto:** `complexidade-cognitiva-ptbr`
- **Versão do projeto:** `0.1.0`
- **Idioma:** português brasileiro
- **Tarefa:** classificação supervisionada multiclasse
- **Classes:** `alta`, `baixa` e `media`
- **Domínio experimental:** textos literários curtos e trechos narrativos voltados à avaliação de complexidade cognitiva

## Finalidade

O dataset foi usado para validar um pipeline acadêmico de Processamento de Linguagem Natural e Aprendizado de Máquina voltado à estimação automática de níveis de complexidade cognitiva em textos literários em português.

A base permite executar, de forma reprodutível, as etapas de validação, limpeza textual, extração de métricas linguísticas interpretáveis, treinamento supervisionado, comparação de experimentos e avaliação final.

## Composição

- **Total de registros:** 9000
- **Colunas:** `id`, `texto`, `target`
- **Distribuição de classes:**

| Classe | Quantidade |
| --- | ---: |
| `alta` | 3000 |
| `baixa` | 3000 |
| `media` | 3000 |

A distribuição balanceada reduz o impacto do desbalanceamento entre classes nos experimentos iniciais e facilita a comparação entre modelos.

## Descrição das colunas

| Coluna | Tipo esperado | Obrigatória | Descrição |
| --- | --- | --- | --- |
| `id` | inteiro ou texto identificador | Sim | Identificador único de cada texto. |
| `texto` | texto | Sim | Conteúdo textual analisado pelo pipeline. |
| `target` | categoria | Sim | Classe de complexidade cognitiva atribuída ao texto. |

## Validações aplicadas

Na execução oficial, o pipeline verificou:

- existência e leitura do CSV configurado;
- presença das colunas obrigatórias;
- ausência de valores nulos em `id`, `texto` e `target`;
- ausência de campos obrigatórios vazios;
- unicidade da coluna `id`;
- existência de pelo menos duas classes;
- ausência de duplicidade textual exata;
- ausência de duplicidade textual normalizada;
- integridade dos textos após a limpeza;
- possibilidade de divisão em treino, validação e teste.

## Resultado da validação

| Item | Resultado |
| --- | --- |
| Valores nulos em colunas obrigatórias | Não encontrados |
| Campos obrigatórios vazios | Não encontrados |
| Textos duplicados exatos | Não encontrados |
| Textos duplicados normalizados | Não encontrados |
| Textos vazios após limpeza | Não encontrados |
| Avisos relevantes | Não registrados |

## Divisão experimental

| Subconjunto | Proporção | Registros | Distribuição por classe |
| --- | ---: | ---: | --- |
| Treino | 70% | 6300 | 2100 por classe |
| Validação | 15% | 1350 | 450 por classe |
| Teste | 15% | 1350 | 450 por classe |

A estratificação foi aplicada com sucesso e manteve o balanceamento das classes nos três subconjuntos.

## Uso recomendado

O dataset é adequado para:

- validar o funcionamento do pipeline;
- testar a extração de métricas linguísticas;
- comparar modelos supervisionados em cenário controlado;
- gerar artefatos acadêmicos iniciais;
- apoiar a escrita do capítulo de Resultados e Discussão.

## Limitações do dataset

O dataset deve ser interpretado como sintético, curado ou controlado. Assim, resultados muito elevados indicam que as classes possuem sinais lexicais e estruturais fortemente separáveis.

Para generalização acadêmica mais ampla, recomenda-se complementar esta base com textos literários reais, rótulos revisados por avaliadores profissionais e validação externa em corpus independente.

## Considerações éticas e de reprodutibilidade

Caso o dataset seja versionado publicamente, ele deve conter apenas textos sintéticos, autorizados, de domínio público ou adequadamente licenciados. Dados privados, sensíveis ou protegidos por restrições autorais não devem ser publicados no repositório.
