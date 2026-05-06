# Metodologia de construção do Dataset

## Fontes

A usa obras literárias em português disponíveis em acervos públicos, principalmente Project Gutenberg.

Cada registro mantém metadados de origem:

- fonte;
- autor;
- obra;
- URL;
- janela textual aproximada.

## Limpeza

O processamento remove, quando possível:

- cabeçalhos e rodapés do Project Gutenberg;
- linhas editoriais;
- licença;
- índices/sumários;
- marcas técnicas não literárias.

## Segmentação

As obras são quebradas em janelas textuais com:

- mínimo: 80 palavras;
- alvo: 160 palavras;
- máximo: 220 palavras;
- sobreposição moderada para aumentar registros sem duplicar textos inteiros.

## Rotulagem

A rotulagem usa um escore linguístico composto por:

- média de palavras por sentença;
- proporção de palavras longas;
- diversidade lexical;
- marcadores de subordinação;
- marcadores discursivos;
- densidade de pontuação;
- marcadores de abstração;
- marcadores de ambiguidade;
- marcadores temporais.

Depois, o script divide os escores em tercis dentro de faixas de tamanho.

## Auditoria obrigatória

Antes de usar o dataset, rode:

```powershell
python scripts/dataset_audit.py
```
