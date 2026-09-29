# Ouvidoria Inteligente — Triagem Semântica de Manifestações Cidadãs
# Entrega 4 — Buscador Semântico + App Streamlit

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    TEM_SENTENCE_TRANSFORMERS = True
except ImportError:
    TEM_SENTENCE_TRANSFORMERS = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
    TEM_LANGCHAIN = True
except ImportError:
    TEM_LANGCHAIN = False


st.set_page_config(page_title="Ouvidoria Inteligente", page_icon="🏛️", layout="wide")

MODELOS_DISPONIVEIS = [
    "paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
]

CATEGORIAS_CORES = {
    "infraestrutura": "#2563eb",
    "saúde": "#dc2626",
    "segurança": "#7c3aed",
    "educação": "#059669",
    "meio ambiente": "#65a30d",
}



# CARREGAMENTO DE DADOS E MODELO (com cache)

@st.cache_data(show_spinner=False)
def carregar_manifestacoes():
    with open("manifestacoes.json", encoding="utf-8") as f:
        dados = json.load(f)
    return pd.DataFrame(dados)


@st.cache_resource(show_spinner="Carregando modelo de embeddings...")
def carregar_modelo(nome_modelo):
    if not TEM_SENTENCE_TRANSFORMERS:
        return None
    return SentenceTransformer(nome_modelo)


@st.cache_data(show_spinner="Gerando embeddings da base...")
def gerar_embeddings(textos, nome_modelo):
    modelo = carregar_modelo(nome_modelo)
    if modelo is None:
        return None
    return modelo.encode(textos, normalize_embeddings=True)


def cor_por_score(score):
    if score > 0.7:
        return "🟢"
    elif score > 0.5:
        return "🟡"
    return "🔴"



# SIDEBAR

st.title("🏛️ Ouvidoria Inteligente")
st.caption(
    "Protótipo de triagem semântica de manifestações cidadãs — combina BoW/TF-IDF, "
    "embeddings, detecção de duplicatas e chunking em uma interface única."
)

if not TEM_SENTENCE_TRANSFORMERS:
    st.error(
        "⚠️ A biblioteca `sentence-transformers` não está instalada. Rode "
        "`pip install -r requirements.txt` para habilitar a busca semântica, a "
        "visualização do espaço vetorial e o chunking com embeddings."
    )
    st.stop()

st.sidebar.header("⚙️ Configurações")
nome_modelo = st.sidebar.selectbox("Modelo de Embedding", MODELOS_DISPONIVEIS)
top_k = st.sidebar.slider("Top-K resultados (Busca Semântica)", 1, 10, 5)

df = carregar_manifestacoes()
textos = df["texto"].tolist()
ids = df["id"].tolist()

embeddings = gerar_embeddings(tuple(textos), nome_modelo)
modelo = carregar_modelo(nome_modelo)

st.sidebar.markdown("---")
st.sidebar.caption(f"📊 Base carregada: **{len(df)} manifestações**")
st.sidebar.caption(f"🧮 Dimensão dos embeddings: **{embeddings.shape[1] if embeddings is not None else '—'}**")



# ABAS

tab_busca, tab_base, tab_espaco, tab_chunking = st.tabs(
    ["🔍 Busca Semântica", "📋 Base Completa", "🌐 Espaço Vetorial", "🧩 Chunking"]
)


# ABA 1 — BUSCA SEMÂNTICA

with tab_busca:
    st.subheader("Buscar manifestações semanticamente similares")
    st.caption(
        "Digite uma descrição livre — o sistema retorna as manifestações mais "
        "próximas semanticamente, mesmo que usem palavras diferentes."
    )

    query = st.text_input(
        "Descrição da manifestação:",
        value="rua cheia de buracos que danifica os carros",
    )

    if st.button("🔍 Buscar", type="primary") and query.strip():
        q_emb = modelo.encode([query], normalize_embeddings=True)
        sims = cosine_similarity(q_emb, embeddings)[0]
        ranking = np.argsort(sims)[::-1][:top_k]

        st.markdown(f"### 🏆 Top-{top_k} manifestações mais similares")
        for pos, idx in enumerate(ranking, 1):
            score = float(sims[idx])
            row = df.iloc[idx]
            emoji = cor_por_score(score)
            with st.expander(f"{emoji} #{pos} — {row['id']} — score: {score:.3f} — {row['categoria_oficial']}"):
                st.write(f"**Data:** {row['data']}")
                st.write(f"**Categoria oficial:** {row['categoria_oficial']}")
                st.info(row["texto"])

        st.caption("🟢 similaridade > 0.7 · 🟡 entre 0.5 e 0.7 · 🔴 abaixo de 0.5")


# ABA 2 — BASE COMPLETA

with tab_base:
    st.subheader("Todas as manifestações da ouvidoria")
    st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("---")
    if st.button("📐 Gerar Matriz de Similaridade Completa"):
        with st.spinner("Calculando similaridades..."):
            matriz = cosine_similarity(embeddings)
            df_matriz = pd.DataFrame(matriz, index=ids, columns=ids)

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(matriz, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(ids)))
        ax.set_yticks(range(len(ids)))
        ax.set_xticklabels(ids, rotation=90, fontsize=6)
        ax.set_yticklabels(ids, fontsize=6)
        plt.colorbar(im, ax=ax, label="Similaridade de cosseno")
        plt.title(f"Matriz de Similaridade — {len(ids)} manifestações")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("#### Pares com maior similaridade (excluindo a diagonal)")
        pares = []
        n = len(ids)
        for i in range(n):
            for j in range(i + 1, n):
                pares.append((ids[i], ids[j], matriz[i, j]))
        pares.sort(key=lambda p: -p[2])
        df_pares = pd.DataFrame(pares[:15], columns=["ID 1", "ID 2", "Similaridade"])
        st.dataframe(df_pares, width="stretch", hide_index=True)


# ABA 3 — ESPAÇO VETORIAL

with tab_espaco:
    st.subheader("Visualização 2D do Espaço Semântico")
    metodo_reducao = st.radio("Método de redução de dimensionalidade:", ["PCA", "t-SNE"], horizontal=True)

    if metodo_reducao == "PCA":
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(embeddings)
        var_explicada = reducer.explained_variance_ratio_.sum()
        st.caption(f"Variância explicada pelas 2 componentes: {var_explicada:.1%}")
    else:
        perp = min(30, max(2, len(df) - 1))
        reducer = TSNE(n_components=2, perplexity=perp, random_state=42)
        coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 7))
    for categoria, cor in CATEGORIAS_CORES.items():
        mask = df["categoria_oficial"] == categoria
        ax.scatter(coords[mask, 0], coords[mask, 1], c=cor, label=categoria,
                   s=130, edgecolor="black", zorder=3, alpha=0.85)
    for i, row in df.reset_index(drop=True).iterrows():
        ax.annotate(row["id"], (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(4, 4), fontsize=6)
    ax.legend(title="Categoria oficial", loc="best", fontsize=8)
    ax.set_title(f"Manifestações no espaço semântico ({metodo_reducao}), coloridas por categoria oficial")
    ax.set_xlabel("Dimensão 1"); ax.set_ylabel("Dimensão 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("#### 🧠 Os clusters semânticos coincidem com as categorias oficiais?")
    st.write(
        "Observe se manifestações da mesma cor (mesma categoria oficial atribuída "
        "manualmente) tendem a ficar próximas no espaço. Quando isso acontece, é "
        "um bom sinal de que a categorização manual está alinhada com o conteúdo "
        "semântico real dos textos. Divergências — pontos da mesma cor espalhados, "
        "ou cores diferentes muito próximas — podem indicar manifestações "
        "mal categorizadas manualmente, ou temas que cruzam mais de uma categoria "
        "oficial (ex.: uma reclamação sobre iluminação pública que afeta tanto a "
        "segurança quanto a infraestrutura)."
    )


# ABA 4 — CHUNKING

with tab_chunking:
    st.subheader("Chunking de Manifestações Longas")

    if not TEM_LANGCHAIN:
        st.error(
            "⚠️ A biblioteca `langchain-text-splitters` não está instalada. Rode "
            "`pip install -r requirements.txt` para habilitar esta aba."
        )
    else:
        manifestacoes_longas = df[df["texto"].str.len() > 300]
        opcoes = ["(colar texto manualmente)"] + manifestacoes_longas["id"].tolist()
        escolha = st.selectbox("Escolha uma manifestação longa da base ou cole seu próprio texto:", opcoes)

        if escolha == "(colar texto manualmente)":
            texto_para_chunk = st.text_area("Cole o texto da manifestação:", height=200)
        else:
            texto_para_chunk = df[df["id"] == escolha]["texto"].values[0]
            st.text_area("Texto selecionado:", value=texto_para_chunk, height=150, disabled=True)

        col1, col2, col3 = st.columns(3)
        estrategia = col1.selectbox("Estratégia", ["RecursiveCharacter", "Fixed-Size (Character)"])
        chunk_size = col2.slider("Chunk Size (caracteres)", 50, 800, 300, 25)
        chunk_overlap = col3.slider("Overlap (caracteres)", 0, 200, 50, 10)

        if st.button("🧩 Gerar Chunks") and texto_para_chunk and texto_para_chunk.strip():
            if estrategia == "RecursiveCharacter":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
            else:
                splitter = CharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=" ",
                )
            chunks = splitter.split_text(texto_para_chunk)
            st.success(f"✅ {len(chunks)} chunks gerados com a estratégia **{estrategia}**.")

            for i, c in enumerate(chunks):
                with st.expander(f"Chunk {i + 1} ({len(c)} caracteres)"):
                    st.write(c)

            emb_chunks = modelo.encode(chunks, normalize_embeddings=True)
            st.caption(f"Embeddings gerados: {emb_chunks.shape}")

            if len(chunks) > 1:
                st.markdown("#### Matriz de Similaridade entre Chunks")
                sim_chunks = cosine_similarity(emb_chunks)
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(sim_chunks, cmap="Purples", vmin=0, vmax=1)
                labels = [f"C{i+1}" for i in range(len(chunks))]
                ax.set_xticks(range(len(chunks))); ax.set_yticks(range(len(chunks)))
                ax.set_xticklabels(labels, fontsize=8); ax.set_yticklabels(labels, fontsize=8)
                for i in range(len(chunks)):
                    for j in range(len(chunks)):
                        ax.text(j, i, f"{sim_chunks[i,j]:.2f}", ha="center", va="center",
                                fontsize=7, color="white" if sim_chunks[i, j] > 0.6 else "black")
                plt.colorbar(im, ax=ax)
                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("#### Espaço 2D dos Chunks (PCA)")
                coords_c = PCA(n_components=2).fit_transform(emb_chunks)
                fig2, ax2 = plt.subplots(figsize=(7, 5))
                ax2.scatter(coords_c[:, 0], coords_c[:, 1], c=range(len(chunks)), cmap="viridis", s=140)
                for i, (x, y) in enumerate(coords_c):
                    ax2.annotate(f"C{i+1}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
                ax2.set_xlabel("Dimensão 1"); ax2.set_ylabel("Dimensão 2")
                ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig2)
            else:
                st.info("Apenas 1 chunk foi gerado — aumente o texto ou reduza o chunk_size para comparar múltiplos chunks.")

st.sidebar.markdown("---")
st.sidebar.caption("Ouvidoria Inteligente — Desafio Integrador (UNIPÊ)")
