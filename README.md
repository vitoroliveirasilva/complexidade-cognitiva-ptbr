# complexidade-cognitiva-ptbr

Projeto desenvolvido para estudo e experimentação de uma abordagem computacional voltada à estimação automática da complexidade cognitiva em textos literários em português.

## Objetivo

Este repositório concentra a etapa de treinamento e validação do projeto, com foco em:

* preparação e organização dos dados;
* extração de métricas linguísticas;
* geração de atributos para os modelos;
* treinamento e avaliação de algoritmos de aprendizado de máquina;
* análise dos resultados obtidos.

A proposta do trabalho é investigar como características linguísticas e textuais podem contribuir para a classificação automática da complexidade cognitiva de textos.

## Escopo do repositório

Este repositório contempla apenas a parte de treinamento da solução. Elementos como API, interface web ou integrações externas se existirem, serão mantidos em repositórios separados.

## Tecnologias previstas

* Python
* Processamento de Linguagem Natural
* Aprendizado de Máquina
* Bibliotecas de análise e manipulação de dados

## Estrutura inicial do projeto

```bash
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── outputs/
│   ├── metrics/
│   ├── figures/
│   └── models/
├── tests/
├── requirements.txt
└── README.md
```

## Etapas previstas

1. Coleta e organização dos textos
2. Pré-processamento dos dados
3. Extração de métricas linguísticas
4. Treinamento dos modelos
5. Avaliação dos resultados
6. Análise comparativa e documentação

## Como executar o projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/vitoroliveirasilva/complexidade-cognitiva-ptbr.git
cd complexidade-cognitiva-ptbr
```

### 2. Criar e ativar um ambiente virtual

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

## Status do projeto

Em desenvolvimento.

## Observações

Os dados, artefatos gerados e modelos treinados não serão versionados no repositório.

## Autor

Vitor Oliveira Silva