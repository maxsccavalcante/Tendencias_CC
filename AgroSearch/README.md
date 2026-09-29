AgroSearch — Motor de Busca Inteligente

Laboratório Prático 04 (Desafio Integrador) · Tópicos Avançados 

Protótipo de motor de busca textual para os manuais técnicos da AgroTech Solutions. O técnico digita uma consulta e o sistema retorna os documentos mais relevantes, ranqueados por TF-IDF.

Como executar:
______________________________
bash                         |
pip install streamlit pandas |
streamlit run agrosearch.py  |         
_____________________________|

O que o app faz
Fase	                |  Descrição
1. Pré-processamento	|  Tokenização → normalização (minúsculas e sem acentos) → stopwords → stemming. Checkboxes na barra lateral ligam/desligam stopwords e stemming e mostram o vocabulário mudando.
2. Índice invertido	  |  Mapeamento termo → [documentos] construído com os tokens já limpos, exibido em tabela (com DF) e em JSON.
3. Busca e ranking    |  Calcula TF, IDF e TF-IDF acumulado, ordena do maior para o menor e destaca o documento vencedor.
+ Bônus               |	 Similaridade de cosseno entre o vetor da consulta e o de cada documento, com opção de ordenar por ela.
  
Fórmulas
TF(t,d) = f(t,d) / |d| (|d| = nº de tokens do documento após o pré-processamento)
IDF(t) = ln(N / DF(t)) (IDF = 0 se o termo não existe na coleção)
TF-IDF(t,d) = TF(t,d) · IDF(t)
Score(q,d) = Σ TF-IDF(t,d) para os termos t da consulta
cos(q,d) = (q · d) / (‖q‖ · ‖d‖)
Restrição técnica

Nenhuma biblioteca de alto nível de RI/PLN foi usada (sem scikit-learn, TfidfVectorizer ou NLTK). Pipeline, stemmer, índice invertido, TF-IDF e cosseno foram implementados do zero com math, re e unicodedata. O pandas é usado apenas para exibir tabelas.

Consultas para testar
lagartas na soja → Doc 2 vence, seguido do Doc 4
irrigação → Docs 5 e 1
controle biológico lagartas → Doc 2 com folga
lagarta (singular) → 0 resultados sem stemming, 2 com stemming

A base de documentos (5 manuais de exemplo) é editável na barra lateral do app.

Arquivos
agrosearch.py — aplicação Streamlit completa
Relatorio_AgroSearch.pdf — relatório (2 páginas)

Equipe:
Max Cavalcante
