# 🏥 HealthSearch

## Sobre o projeto

O HealthSearch é um sistema de recuperação de informação médica desenvolvido para comparar diferentes estratégias de busca textual.

O projeto utiliza três abordagens principais:

- Busca léxica utilizando BM25;
- Busca semântica utilizando embeddings;
- Combinação dos rankings utilizando Reciprocal Rank Fusion (RRF).

## Objetivo

O objetivo é avaliar como diferentes métodos de recuperação de informação classificam documentos médicos de acordo com uma consulta realizada pelo usuário.

## Tecnologias

- Python
- Streamlit
- Pandas
- NumPy
- NLTK
- Rank-BM25
- Sentence Transformers
- Scikit-learn

## Arquitetura

Consulta
   |
   +-------------------+
   |                   |
   v                   v
 BM25             Embeddings
   |                   |
   v                   v
Ranking             Similaridade
Léxico              Semântica
   |                   |
   +---------+---------+
             |
             v
            RRF
             |
             v
      Ranking Híbrido

Ferramentas de Inteligência Artificial foram utilizadas como apoio durante o desenvolvimento do projeto, principalmente para esclarecimento de conceitos, estruturação do código, identificação e correção de erros e apoio na documentação.

A implementação foi analisada, adaptada e testada pelo responsável pelo projeto, que permanece responsável pelo conteúdo e funcionamento final da aplicação.
