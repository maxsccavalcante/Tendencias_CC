# -*- coding: utf-8 -*-
"""
AgroSearch - Motor de Busca Inteligente
Laboratório Prático 04 - Desafio Integrador
Disciplina: Tópicos Avançados - Recuperação de Informação / PLN (UNIPÊ)

Como executar:
    pip install streamlit pandas
    streamlit run agrosearch.py

Integra os 3 pilares vistos nas Aulas 1, 2 e 3:
    Fase 1 - Pipeline de pré-processamento (tokenização, normalização,
             stopwords e stemming) com checkboxes para ligar/desligar etapas.
    Fase 2 - Índice invertido em memória (Termo -> [IDs dos documentos]).
    Fase 3 - Busca e ranqueamento por TF-IDF (+ bônus: similaridade de cosseno).

Restrição do desafio: NENHUMA biblioteca de alto nível de RI/PLN é usada
(sem scikit-learn, sem TfidfVectorizer, sem NLTK). O pipeline, o índice
invertido e o cálculo de TF-IDF/cosseno foram implementados do zero, usando
apenas a biblioteca padrão do Python (math, re, unicodedata) e o pandas
apenas para EXIBIR tabelas.
"""

import math
import re
import unicodedata

import pandas as pd
import streamlit as st

# =============================================================================
# BASE DE DOCUMENTOS (hardcode sugerido no enunciado)
# =============================================================================
DOCUMENTOS_PADRAO = [
    "A soja requer irrigação constante durante o período de floração para garantir a produtividade.",
    "O controle biológico de lagartas na soja pode ser feito com a vespa Trichogramma.",
    "A adubação verde com leguminosas melhora o nitrogênio no solo para o milho.",
    "Lagartas desfolhadoras causam grande prejuízo na cultura da soja e do algodão.",
    "A irrigação por gotejamento economiza água e é ideal para o cultivo orgânico.",
]

# Stopwords já escritas SEM acento e em minúsculas, pois a remoção acontece
# depois da normalização (ex.: "é" vira "e", "até" vira "ate").
STOPWORDS_PT = {
    "a", "as", "o", "os", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem", "sob", "sobre", "entre", "ate", "durante",
    "e", "ou", "que", "se", "ao", "aos", "pelo", "pela", "pelos", "pelas",
    "ser", "sao", "foi", "pode", "podem", "como", "mais", "muito",
    "seu", "sua", "seus", "suas", "este", "esta", "isso", "ele", "ela",
}


# =============================================================================
# FASE 1 - PIPELINE DE PRÉ-PROCESSAMENTO
# =============================================================================
def tokenizar(texto):
    """Etapa 1: quebra o texto em tokens (sequências de letras/dígitos)."""
    return re.findall(r"\b\w+\b", texto)


def normalizar(token):
    """Etapa 2: minúsculas + remoção de acentos (ç -> c, ã -> a, é -> e ...)."""
    token = token.lower()
    decomposto = unicodedata.normalize("NFD", token)
    return "".join(c for c in decomposto if unicodedata.category(c) != "Mn")


def remover_stopwords(tokens, stopwords=STOPWORDS_PT):
    """Etapa 3: descarta palavras funcionais que não ajudam a distinguir documentos."""
    return [t for t in tokens if t not in stopwords]


# --- Etapa 4: Stemming (redução à raiz) --------------------------------------
# Stemmer de sufixos para o português, inspirado no RSLP e simplificado.
# Cada regra é (sufixo, substituição, tamanho mínimo do radical restante).
# Em cada grupo vale a regra do MAIOR sufixo que casar.
def _ordenar(regras):
    return sorted(regras, key=lambda r: -len(r[0]))


_REGRAS_PLURAL = _ordenar([
    ("oes", "ao", 2), ("aes", "ao", 2), ("ais", "al", 2), ("eis", "el", 2),
    ("res", "r", 3), ("s", "", 3),
])
_REGRAS_ADVERBIO = _ordenar([("mente", "", 4)])
_REGRAS_SUBSTANTIVO = _ordenar([
    ("amento", "", 3), ("imento", "", 3), ("mento", "", 4),
    ("acao", "", 3), ("icao", "", 3), ("ucao", "", 3), ("cao", "", 3),
    ("idade", "", 3), ("adora", "", 3), ("ador", "", 3),
    ("ismo", "", 3), ("ista", "", 3),
    ("osa", "", 3), ("oso", "", 3), ("ica", "", 3), ("ico", "", 3),
    ("avel", "", 3), ("ivel", "", 3), ("ante", "", 3), ("ente", "", 3),
    ("eira", "", 3), ("eiro", "", 3), ("al", "", 4), ("ao", "", 4), ("io", "", 4),
])
_REGRAS_VERBO = _ordenar([("am", "", 4), ("ar", "", 4), ("er", "", 4), ("ir", "", 4)])


def _aplicar_regras(palavra, regras):
    """Aplica a primeira regra (maior sufixo) que casar. Retorna (palavra, mudou)."""
    for sufixo, troca, min_radical in regras:
        if palavra.endswith(sufixo) and len(palavra) - len(sufixo) >= min_radical:
            if sufixo == "s" and palavra.endswith("ss"):
                continue
            return palavra[: len(palavra) - len(sufixo)] + troca, True
    return palavra, False


def stemmer_ptbr(palavra):
    """Reduz uma palavra (já normalizada) ao seu radical aproximado."""
    if len(palavra) <= 3 or not palavra.isalpha():
        return palavra
    palavra, _ = _aplicar_regras(palavra, _REGRAS_PLURAL)      # lagartas -> lagarta
    palavra, _ = _aplicar_regras(palavra, _REGRAS_ADVERBIO)    # rapidamente -> rapida
    palavra, mudou = _aplicar_regras(palavra, _REGRAS_SUBSTANTIVO)  # irrigacao -> irrig
    if not mudou:
        palavra, _ = _aplicar_regras(palavra, _REGRAS_VERBO)   # garantir -> garant
    if len(palavra) >= 4 and palavra[-1] in "aeo":             # soja -> soj
        palavra = palavra[:-1]
    return palavra


def preprocessar(texto, usar_stopwords=True, usar_stemming=True):
    """Executa o pipeline completo e devolve o resultado de CADA etapa."""
    tokens = tokenizar(texto)
    normalizados = [normalizar(t) for t in tokens]
    sem_stop = remover_stopwords(normalizados) if usar_stopwords else list(normalizados)
    finais = [stemmer_ptbr(t) for t in sem_stop] if usar_stemming else list(sem_stop)
    return {
        "tokens": tokens,
        "normalizados": normalizados,
        "sem_stopwords": sem_stop,
        "finais": finais,
    }


# =============================================================================
# FASE 2 - ÍNDICE INVERTIDO (Termo -> [IDs de documentos])
# =============================================================================
def construir_indice_invertido(docs_tokens):
    """
    docs_tokens: {"Doc 1": [tokens finais], ...}
    Retorna {termo: [ids dos documentos que contêm o termo]} (sem repetições).
    """
    indice = {}
    for id_doc, tokens in docs_tokens.items():
        for termo in tokens:
            postings = indice.setdefault(termo, [])
            if id_doc not in postings:
                postings.append(id_doc)
    return dict(sorted(indice.items()))


# =============================================================================
# FASE 3 - TF, IDF, TF-IDF E COSSENO (tudo do zero)
# =============================================================================
def calcular_tf(tokens, termo):
    """TF(t, d) = ocorrências de t em d / total de tokens de d."""
    return tokens.count(termo) / len(tokens) if tokens else 0.0


def calcular_idf(n_docs, df):
    """IDF(t) = ln(N / DF(t)); se o termo não existe na coleção, IDF = 0."""
    return math.log(n_docs / df) if df > 0 else 0.0


def ranquear(tokens_query, docs_tokens, indice):
    """
    Calcula, para cada documento:
      - TF-IDF acumulado = soma de TF(t,d) * IDF(t) para os termos t da consulta;
      - similaridade de cosseno entre o vetor TF-IDF da consulta e o do documento.
    Retorna (resultados_por_doc, detalhes_por_termo, tabela_idf).
    """
    n_docs = len(docs_tokens)
    termos_query = list(dict.fromkeys(tokens_query))  # únicos, mantendo a ordem

    # IDF dos termos da consulta e de todo o vocabulário (usa o índice invertido)
    idf = {t: calcular_idf(n_docs, len(indice.get(t, []))) for t in termos_query}
    idf_vocab = {t: calcular_idf(n_docs, len(p)) for t, p in indice.items()}

    tabela_idf = [
        {
            "Termo": t,
            "DF": len(indice.get(t, [])),
            "N": n_docs,
            "IDF = ln(N/DF)": idf[t],
            "Documentos (índice)": ", ".join(indice.get(t, [])) or "— não encontrado —",
        }
        for t in termos_query
    ]

    # Vetor TF-IDF da consulta (TF calculado sobre os tokens da própria consulta)
    q_vec = {t: calcular_tf(tokens_query, t) * idf[t] for t in termos_query}
    norma_q = math.sqrt(sum(v * v for v in q_vec.values()))

    resultados, detalhes = {}, []
    for id_doc, tokens in docs_tokens.items():
        acumulado, produto = 0.0, 0.0
        for t in termos_query:
            tf = calcular_tf(tokens, t)
            tfidf = tf * idf[t]
            acumulado += tfidf
            produto += q_vec[t] * tfidf
            detalhes.append({
                "Documento": id_doc, "Termo": t, "TF": tf,
                "IDF": idf[t], "TF-IDF": tfidf,
            })
        # Norma do vetor do documento considera TODO o vocabulário do documento
        norma_d = math.sqrt(sum(
            (calcular_tf(tokens, t) * idf_vocab[t]) ** 2 for t in set(tokens)
        ))
        cosseno = produto / (norma_q * norma_d) if norma_q > 0 and norma_d > 0 else 0.0
        resultados[id_doc] = {"tfidf": acumulado, "cosseno": cosseno}
    return resultados, detalhes, tabela_idf


# =============================================================================
# FUNÇÕES AUXILIARES DE INTERFACE
# =============================================================================
def carregar_documentos(texto):
    """Converte o texto (1 documento por linha) em {"Doc N": texto}."""
    docs = {}
    for linha in texto.split("\n"):
        linha = re.sub(r"^\s*doc\s*\d+\s*[:\-–]\s*", "", linha, flags=re.IGNORECASE).strip()
        if linha:
            docs[f"Doc {len(docs) + 1}"] = linha
    return docs


def _mostrar_tabela(df, destacar_primeira=False, casas=4):
    """Exibe um DataFrame (opcionalmente com a 1ª linha destacada em verde)."""
    try:
        colunas_num = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
        estilo = df.style.format({c: f"{{:.{casas}f}}" for c in colunas_num})
        if destacar_primeira:
            estilo = estilo.apply(
                lambda linha: [
                    "background-color: #d4edda; color: #155724; font-weight: bold"
                    if linha.name == 0 else ""
                ] * len(linha),
                axis=1,
            )
        st.dataframe(estilo, hide_index=True)
    except Exception:  # fallback caso o Styler não esteja disponível
        st.dataframe(df.round(casas), hide_index=True)


def _resumo(texto, limite=90):
    return texto if len(texto) <= limite else texto[: limite - 1].rstrip() + "…"


# =============================================================================
# APLICAÇÃO STREAMLIT
# =============================================================================
def main():
    st.set_page_config(page_title="AgroSearch", page_icon="🌱", layout="wide")
    st.title("🌱 AgroSearch — Motor de Busca Inteligente")
    st.caption(
        "AgroTech Solutions · protótipo de busca textual em manuais técnicos · "
        "índice invertido e TF-IDF implementados do zero (sem scikit-learn)"
    )

    # ---------------------------- Barra lateral ------------------------------
    with st.sidebar:
        st.header("⚙️ Pipeline (Fase 1)")
        usar_stop = st.checkbox("Remover stopwords", value=True)
        usar_stem = st.checkbox("Aplicar stemming", value=True)
        st.caption("Tokenização e normalização (minúsculas + sem acentos) são sempre aplicadas.")
        st.header("⭐ Bônus")
        usar_cos = st.checkbox("Similaridade de cosseno", value=True)
        st.divider()
        with st.expander("📚 Base de documentos (editável)"):
            texto_docs = st.text_area(
                "Um documento por linha:",
                value="\n".join(DOCUMENTOS_PADRAO),
                height=320,
            )

    documentos = carregar_documentos(texto_docs)
    if not documentos:
        st.error("Informe ao menos um documento na barra lateral.")
        return

    # Pré-processa toda a base com as configurações atuais dos checkboxes
    proc = {i: preprocessar(t, usar_stop, usar_stem) for i, t in documentos.items()}
    docs_tokens = {i: p["finais"] for i, p in proc.items()}
    indice = construir_indice_invertido(docs_tokens)

    # A consulta é a mesma para as 3 abas
    consulta = st.text_input(
        "🔎 Consulta do técnico de campo:",
        value="lagartas na soja",
        help="Exemplos: 'lagartas na soja', 'irrigação', 'controle biológico', 'milho'",
    )

    aba1, aba2, aba3 = st.tabs([
        "1️⃣ Pré-processamento", "2️⃣ Índice Invertido", "3️⃣ Busca e Ranking TF-IDF"
    ])

    # ------------------------------ ABA 1 ------------------------------------
    with aba1:
        st.subheader("Fase 1 — Pipeline de pré-processamento")
        vocab_bruto = {t for p in proc.values() for t in p["tokens"]}
        vocab_norm = {t for p in proc.values() for t in p["normalizados"]}
        vocab_final = set(indice.keys())

        c1, c2, c3 = st.columns(3)
        c1.metric("Vocabulário bruto (tokens únicos)", len(vocab_bruto))
        c2.metric("Após normalização", len(vocab_norm), delta=len(vocab_norm) - len(vocab_bruto),
                  delta_color="inverse")
        c3.metric("Vocabulário final (índice)", len(vocab_final),
                  delta=len(vocab_final) - len(vocab_bruto), delta_color="inverse")
        st.caption(
            "Ligue/desligue *stopwords* e *stemming* na barra lateral e observe o vocabulário mudar. "
            "As stopwords reduzem o tamanho do vocabulário; o stemming troca as palavras pelos radicais "
            "e agrupa variações (ex.: *lagarta*/*lagartas*). Na base padrão as palavras já aparecem na "
            "mesma forma, então teste a consulta **lagarta** (singular) com e sem stemming na aba 3."
        )

        escolhido = st.selectbox("Documento para inspecionar:", list(documentos.keys()))
        p = proc[escolhido]
        st.markdown(f"**Texto original:** {documentos[escolhido]}")

        with st.expander("1. Tokenização", expanded=True):
            st.write(p["tokens"])
        with st.expander("2. Normalização (minúsculas e sem acentos)"):
            st.write(p["normalizados"])
        with st.expander("3. Remoção de stopwords" + ("" if usar_stop else " (desativada)")):
            if usar_stop:
                removidas = [t for t in p["normalizados"] if t in STOPWORDS_PT]
                st.write(p["sem_stopwords"])
                st.caption(f"Removidas: {removidas}")
            else:
                st.caption("Etapa desativada: os tokens seguem inalterados.")
                st.write(p["sem_stopwords"])
        with st.expander("4. Stemming (redução à raiz)" + ("" if usar_stem else " (desativado)")):
            if usar_stem:
                st.write(p["finais"])
                st.dataframe(
                    pd.DataFrame({"Palavra": p["sem_stopwords"], "Radical": p["finais"]}),
                    hide_index=True,
                )
            else:
                st.caption("Etapa desativada: os tokens seguem inalterados.")
                st.write(p["finais"])

        with st.expander("📖 Vocabulário final da coleção"):
            st.write(sorted(vocab_final))
        with st.expander("🧾 Tokens finais de todos os documentos"):
            st.dataframe(
                pd.DataFrame({
                    "Documento": list(docs_tokens.keys()),
                    "Nº de tokens": [len(t) for t in docs_tokens.values()],
                    "Tokens finais": [", ".join(t) for t in docs_tokens.values()],
                }),
                hide_index=True,
            )
        with st.expander("🛑 Lista de stopwords utilizada"):
            st.write(sorted(STOPWORDS_PT))

    # ------------------------------ ABA 2 ------------------------------------
    with aba2:
        st.subheader("Fase 2 — Índice invertido (Termo → Documentos)")
        st.markdown(
            "Construído a partir dos **tokens pré-processados**. O **DF** (document frequency) "
            "é simplesmente o tamanho da lista de documentos de cada termo."
        )
        df_indice = pd.DataFrame([
            {"Termo": t, "DF": len(p), "Documentos": ", ".join(p)} for t, p in indice.items()
        ])
        st.dataframe(df_indice, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            with st.expander("Ver índice invertido em JSON", expanded=False):
                st.json(indice)
        with c2:
            with st.expander("Ver índice direto (Doc → Termos) para comparar", expanded=False):
                st.json(docs_tokens)

    # ------------------------------ ABA 3 ------------------------------------
    with aba3:
        st.subheader("Fase 3 — Busca e ranqueamento")
        q = preprocessar(consulta, usar_stop, usar_stem)
        tokens_query = q["finais"]
        st.markdown(f"**Consulta:** `{consulta}` → **pré-processada:** `{tokens_query}`")

        if not tokens_query:
            st.info("Digite uma consulta. (Se ela só tiver stopwords, todos os termos foram descartados.)")
        else:
            resultados, detalhes, tabela_idf = ranquear(tokens_query, docs_tokens, indice)

            st.markdown("**1) Termos da consulta no índice invertido**")
            _mostrar_tabela(pd.DataFrame(tabela_idf))

            criterio = "TF-IDF acumulado"
            if usar_cos:
                criterio = st.radio(
                    "Ordenar o ranking por:", ["TF-IDF acumulado", "Similaridade de cosseno"],
                    horizontal=True,
                )
            chave = "tfidf" if criterio == "TF-IDF acumulado" else "cosseno"

            ordem = sorted(
                resultados.items(),
                key=lambda kv: (-kv[1][chave], int(kv[0].split()[-1])),
            )
            linhas = []
            for pos, (id_doc, r) in enumerate(ordem, start=1):
                linha = {"Posição": pos, "Documento": id_doc, "TF-IDF acumulado": r["tfidf"]}
                if usar_cos:
                    linha["Cosseno"] = r["cosseno"]
                linha["Trecho"] = _resumo(documentos[id_doc])
                linhas.append(linha)

            st.markdown(f"**2) Ranking (maior → menor {criterio})**")
            melhor = ordem[0][1][chave]
            _mostrar_tabela(pd.DataFrame(linhas), destacar_primeira=melhor > 0)

            if melhor > 0:
                empatados = [i for i, r in ordem if abs(r[chave] - melhor) < 1e-12]
                vencedor = empatados[0]
                st.success(f"🏆 **Documento vencedor: {vencedor}** ({criterio} = {melhor:.4f})")
                st.markdown(f"> {documentos[vencedor]}")
                if len(empatados) > 1:
                    st.caption(f"Empate técnico entre: {', '.join(empatados)} "
                               "(desempate pela ordem dos documentos).")
            else:
                st.warning("Nenhum documento contém os termos da consulta (score = 0 para todos).")

            with st.expander("🔬 Detalhamento: TF, IDF e TF-IDF por documento e termo"):
                _mostrar_tabela(pd.DataFrame(detalhes), casas=4)

            with st.expander("📐 Fórmulas utilizadas"):
                st.latex(r"TF(t,d)=\frac{f(t,d)}{|d|}\qquad IDF(t)=\ln\frac{N}{DF(t)}")
                st.latex(r"TFIDF(t,d)=TF(t,d)\cdot IDF(t)\qquad "
                         r"Score(q,d)=\sum_{t\in q}TFIDF(t,d)")
                st.latex(r"\cos(\vec q,\vec d)=\frac{\vec q\cdot\vec d}"
                         r"{\lVert\vec q\rVert\,\lVert\vec d\rVert}")
                st.caption(
                    "f(t,d): ocorrências do termo no documento; |d|: nº de tokens do documento "
                    "após o pré-processamento; N: nº de documentos; DF(t): nº de documentos "
                    "com o termo (tamanho da lista no índice invertido). Termos que aparecem "
                    "em todos os documentos têm IDF = 0. O vetor da consulta usa o TF "
                    "calculado sobre os tokens da própria consulta."
                )


if __name__ == "__main__":
    main()
