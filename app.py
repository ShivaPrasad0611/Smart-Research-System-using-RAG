"""
app.py
------
Streamlit UI for the Smart Research Assistant.

Run with:
    streamlit run app.py

Provides:
    - Sidebar configuration (embedding/LLM provider, vector store, top-k)
    - Document upload (PDF, TXT, MD, DOCX) and/or a web URL
    - A chat interface with multi-turn memory
    - Expandable source snippets for every answer
    - Reliability/evaluation scores per answer
"""

import os
import tempfile
import streamlit as st

# On Streamlit Cloud, this picks up secrets and bridges them into env vars
# so config.py's os.getenv() calls work. Locally, if no secrets.toml exists,
# this safely does nothing -- local testers' .env file still works as normal.
try:
    for key, value in st.secrets.items():
        os.environ[key] = str(value)
except FileNotFoundError:
    pass


from config import settings
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="Smart Research Assistant", page_icon="🔎", layout="wide")


# ----------------------------------------------------------------------
# Sidebar: configuration
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

embedding_provider = st.sidebar.selectbox(
    "Embedding provider", ["huggingface", "openai"],
    index=0 if settings.embedding_provider == "huggingface" else 1,
    help="huggingface = free & local. openai = needs an API key, often higher quality.",
)
llm_provider = st.sidebar.selectbox(
    "LLM provider", ["groq", "huggingface", "openai"],
    index=["groq", "huggingface", "openai"].index(settings.llm_provider)
    if settings.llm_provider in ("groq", "huggingface", "openai") else 0,
    help="groq = free, very fast hosted API (default). huggingface = free local model, slower. openai = paid hosted API.",
)
vector_store_type = st.sidebar.selectbox(
    "Vector store", ["faiss", "chroma"],
    index=0 if settings.vector_store == "faiss" else 1,
    help="FAISS: fast, in-memory, great for prototyping. Chroma: persistent, production-like.",
)
top_k = st.sidebar.slider("Chunks to retrieve (top-k)", min_value=1, max_value=10, value=settings.top_k)
search_type = st.sidebar.selectbox(
    "Retrieval strategy", ["mmr", "similarity"],
    index=0 if settings.search_type == "mmr" else 1,
    help="mmr: diversifies results to avoid redundant chunks. similarity: plain nearest-neighbor.",
)
enable_rerank = st.sidebar.checkbox(
    "Rerank with cross-encoder", value=settings.enable_rerank,
    help="Over-fetches candidates and reorders them with BAAI/bge-reranker-base for better precision. Slower per query.",
)
run_eval = st.sidebar.checkbox("Show evaluation metrics", value=True)

if llm_provider == "openai" or embedding_provider == "openai":
    key_input = st.sidebar.text_input(
        "OpenAI API key", type="password", value=settings.openai_api_key,
        help="Only needed if you selected 'openai' above. Stored only for this session.",
    )
    if key_input:
        os.environ["OPENAI_API_KEY"] = key_input
        settings.openai_api_key = key_input

if llm_provider == "groq":
    groq_key_input = st.sidebar.text_input(
        "Groq API key", type="password", value=settings.groq_api_key,
        help="Free key from https://console.groq.com. Stored only for this session.",
    )
    if groq_key_input:
        os.environ["GROQ_API_KEY"] = groq_key_input
        settings.groq_api_key = groq_key_input

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset conversation & documents"):
    for k in ("pipeline", "messages"):
        st.session_state.pop(k, None)
    st.rerun()


# ----------------------------------------------------------------------
# Session state: build (or reuse) the pipeline
# ----------------------------------------------------------------------
def get_pipeline() -> RAGPipeline:
    cfg_key = (embedding_provider, llm_provider, vector_store_type, top_k, search_type, enable_rerank)
    if "pipeline" not in st.session_state or st.session_state.get("cfg_key") != cfg_key:
        st.session_state.pipeline = RAGPipeline(
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            vector_store_type=vector_store_type,
            top_k=top_k,
            search_type=search_type,
            enable_rerank=enable_rerank,
        )
        st.session_state.cfg_key = cfg_key
    return st.session_state.pipeline


if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------------------------------------------------
# Main layout
# ----------------------------------------------------------------------
st.title("🔎 Smart Research Assistant")
st.caption("Ask questions about your own documents — answers are grounded in retrieved context, not guesswork.")

tab_ingest, tab_chat = st.tabs(["📄 Documents", "💬 Chat"])

with tab_ingest:
    st.subheader("1. Add your documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, MD, or DOCX files", type=["pdf", "txt", "md", "docx"], accept_multiple_files=True
    )
    url = st.text_input("...or add a web page URL")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Ingest documents", type="primary", disabled=not uploaded_files):
            pipeline = get_pipeline()
            os.makedirs(settings.upload_dir, exist_ok=True)
            saved_paths = []
            for f in uploaded_files:
                path = os.path.join(settings.upload_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                saved_paths.append(path)
            with st.spinner("Chunking, embedding, and indexing your documents..."):
                n = pipeline.ingest_files(saved_paths)
            st.success(f"Indexed {n} chunks from {len(saved_paths)} file(s). Switch to the Chat tab to ask questions.")

    with col2:
        if st.button("🌐 Ingest URL", disabled=not url):
            pipeline = get_pipeline()
            with st.spinner(f"Fetching and indexing {url}..."):
                n = pipeline.ingest_url(url)
            st.success(f"Indexed {n} chunks from the URL.")

    if "pipeline" in st.session_state and st.session_state.pipeline.num_chunks > 0:
        st.info(f"📚 Currently indexed: {st.session_state.pipeline.num_chunks} chunks.")

with tab_chat:
    st.subheader("2. Ask questions")

    has_docs = "pipeline" in st.session_state and st.session_state.pipeline.num_chunks > 0
    if not has_docs:
        st.warning("Upload and ingest at least one document in the 'Documents' tab first.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Sources used"):
                    for i, s in enumerate(msg["sources"], 1):
                        st.markdown(f"**{i}. {s['source']}**")
                        st.text(s["snippet"])
                if msg.get("evaluation"):
                    ev = msg["evaluation"]
                    st.caption(
                        f"Reliability ({ev.get('method', 'n/a')}) — "
                        f"Faithfulness: {ev.get('faithfulness', 'n/a')} · "
                        f"Relevance: {ev.get('answer_relevance', 'n/a')} · "
                        f"Context precision: {ev.get('context_precision', 'n/a')}"
                    )

    question = st.chat_input("Ask a question about your documents...", disabled=not has_docs)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        pipeline = get_pipeline()
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating an answer..."):
                try:
                    result = pipeline.query(question, use_memory=True, run_evaluation=run_eval)
                    st.markdown(result.answer)
                    sources = [
                        {"source": d.metadata.get("source", "unknown"), "snippet": d.page_content[:500]}
                        for d in result.sources
                    ]
                    with st.expander("📎 Sources used"):
                        for i, s in enumerate(sources, 1):
                            st.markdown(f"**{i}. {s['source']}**")
                            st.text(s["snippet"])
                    if result.evaluation:
                        ev = result.evaluation
                        st.caption(
                            f"Reliability ({ev.get('method', 'n/a')}) — "
                            f"Faithfulness: {ev.get('faithfulness', 'n/a')} · "
                            f"Relevance: {ev.get('answer_relevance', 'n/a')} · "
                            f"Context precision: {ev.get('context_precision', 'n/a')}"
                        )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result.answer,
                            "sources": sources,
                            "evaluation": result.evaluation,
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Something went wrong: {e}")
