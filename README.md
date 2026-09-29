# Projetos de Recuperação de Informação e PLN

Este repositório reúne três projetos que fizemos para estudar, na prática, como um computador encontra informação em textos. Cada um parte de uma ideia diferente de busca, e juntos formam uma espécie de linha do tempo: começamos com a busca por palavras, passamos por uma combinação de palavras e significado, e chegamos à busca puramente semântica.

## Os projetos

### AgroSearch
Um motor de busca para manuais técnicos de agricultura, criado para o Laboratório Prático 04 da disciplina de Tópicos Avançados (UNIPÊ). O técnico digita uma consulta e o sistema devolve os trechos mais relevantes, ranqueados por TF-IDF, com similaridade de cosseno como extra.

O ponto central é que tudo foi feito do zero: o pré-processamento (tokenização, normalização, stopwords e stemming), o índice invertido e as fórmulas de TF-IDF. Não usamos scikit-learn nem outras bibliotecas de alto nível, porque a ideia era entender o que acontece por dentro. A interface é um app Streamlit em que dá para ligar e desligar etapas do pipeline e ver o vocabulário mudar.

### HealthSearch
Um sistema de busca em documentos médicos feito para comparar estratégias. Ele roda a mesma consulta por três caminhos: busca léxica com BM25, busca semântica com embeddings e uma combinação dos dois rankings usando Reciprocal Rank Fusion (RRF). Assim dá para ver onde cada método acerta, onde falha e quanto a fusão ajuda.

Usa Python, Streamlit, Pandas, NumPy, NLTK, Rank-BM25, Sentence Transformers e Scikit-learn.

### Ouvidoria Inteligente
Um desafio integrador de PLN e Engenharia de IA sobre triagem de manifestações de cidadãos. Aqui o foco é em representações vetoriais, busca semântica e chunking. O projeto compara BoW, TF-IDF e embeddings, detecta manifestações duplicadas, testa diferentes configurações de chunking e reúne tudo em um app Streamlit com quatro abas.

## Como os três se conectam

|       Projeto         |        Tipo de busca        |            Ideia principal             |
|-----------------------|-----------------------------|----------------------------------------|
| AgroSearch            | Léxica                      | Índice invertido e TF-IDF feitos à mão |
| HealthSearch          | Léxica, semântica e híbrida | BM25 + embeddings + RRF                |
| Ouvidoria Inteligente | Semântica                   | Embeddings, duplicatas e chunking      |

## Como rodar

Cada projeto tem seu próprio README com as instruções detalhadas e as dependências. Em geral, o caminho é instalar os requisitos e abrir o app com:
```
_____________________________
bash                        |
streamlit run nome_do_app.py|
                            |
____________________________|
```
Os projetos que usam Sentence Transformers baixam um modelo pré-treinado na primeira execução, então é preciso ter internet nesse momento.

## Autoria

Mantido por Max Samuel Caitano Cavalcante. A autoria de cada projeto, incluindo os colegas de equipe, está descrita no README de cada um.

## Uso de inteligência artificial

Usamos ferramentas de IA generativa (Claude, da Anthropic) como apoio: para esclarecer conceitos, estruturar código, encontrar e corrigir erros e ajudar na documentação. Todo o material foi revisado, adaptado e testado por nós, e a responsabilidade pelo conteúdo e pelo funcionamento final é nossa.
