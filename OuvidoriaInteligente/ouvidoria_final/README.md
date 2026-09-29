# Ouvidoria Inteligente — Triagem Semântica de Manifestações Cidadãs

Desafio Integrador — Processamento de Linguagem Natural / Engenharia de IA
Tema: Representações Vetoriais, Busca Semântica e Chunking

## Nota sobre o dataset

O enunciado do desafio faz referência a um arquivo `manifestacoes.json`,
contendo 40 manifestações reais anonimizadas, a ser usado como base do
projeto. Não conseguimos localizar esse arquivo dentro do prazo da entrega.
Para não travar o desenvolvimento, geramos um **dataset sintético
equivalente** (`manifestacoes.json` neste repositório), reproduzindo todas as
restrições descritas no enunciado — ver `gerar_dataset.py` para o código de
geração e validação, e `RELATORIO.pdf` para os detalhes.

Caso o dataset oficial apareça depois, basta substituir o arquivo
`manifestacoes.json` (mesmo formato: `id`, `data`, `categoria_oficial`,
`texto`) e reexecutar os notebooks e o app — nenhum código precisa mudar.

## Estrutura do repositório

| Arquivo | Entrega | Descrição |
|---|---|---|
| `manifestacoes.json` | — | Base de 40 manifestações (sintética, ver nota acima) |
| `duplicatas_reais.json` | — | Gabarito de duplicatas reais conhecidas (usado na Entrega 2) |
| `gerar_dataset.py` | — | Script de geração/validação do dataset sintético |
| `analise_comparativa.ipynb` | 1 | BoW × TF-IDF × Embeddings, pares do enunciado |
| `deteccao_duplicatas.ipynb` | 2 | Função `detectar_duplicatas`, heatmap, falsos positivos/negativos |
| `chunking_manifestacoes.ipynb` | 3 | Chunking com LangChain, 2 configurações, visualização 2D |
| `app_ouvidoria.py` | 4 | App Streamlit com 4 abas |
| `RELATORIO.pdf` | — | Relatório técnico (até 5 páginas) |

## Como rodar

```bash
pip install -r requirements.txt

# Gerar/regerar o dataset (opcional — já vem gerado)
python gerar_dataset.py

# Notebooks (Jupyter)
jupyter notebook analise_comparativa.ipynb

# App Streamlit
streamlit run app_ouvidoria.py
```

O app abre em `http://localhost:8501`.

**Importante:** os notebooks e o app usam `sentence-transformers`, que baixa
um modelo pré-treinado do Hugging Face na primeira execução — é necessário
ter conexão à internet nesse momento (o download é único, depois fica em
cache local).

## Autoria

Equipe (grupo de 3):
- Vitor Nóbrega de Souza
- Max Samuel Caitano Cavalcante
- Augusto Lustosa de Alencar e Fontoura

Ver `RELATORIO.pdf`, seção "Equipe", para a divisão de responsabilidades.

## Transparência

Este projeto foi desenvolvido com auxílio de ferramentas de Inteligência
Artificial Generativa (Claude, da Anthropic) — ver nota de transparência ao
final do `RELATORIO.pdf`.
