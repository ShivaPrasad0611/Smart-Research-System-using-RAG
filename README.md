# 🔎 Smart Research Assistant (RAG-Based Knowledge System)

A Retrieval-Augmented Generation (RAG) AI assistant that lets you upload documents
(PDFs, text, Word docs, or a web page) and ask natural-language questions about
them. Every answer is grounded in retrieved source text — not a generic LLM
guess — and comes with a reliability score.

Runs on **free defaults out of the box**: local HuggingFace embeddings (no
key needed) + Groq for generation (free, but needs a free API key from
console.groq.com). Switch `LLM_PROVIDER=huggingface` for a fully offline
setup with no keys at all (downloads a ~3GB local model, slower on CPU), or
upgrade to **OpenAI** for a paid, high-quality alternative.

---

## How it works (pipeline)

```
 ┌──────────────┐   ┌────────────────┐   ┌────────────────┐   ┌───────────────┐   ┌──────────────┐
 │ 1. Ingestion │──▶│ 2. Chunking &  │──▶│ 3. Retrieval   │──▶│ 4. Generation │──▶│ 5. Evaluation│
 │ Load PDFs,   │   │    Embedding   │   │ MMR/similarity │   │ LLM answers   │   │ Faithfulness,│
 │ TXT, DOCX,   │   │ Store vectors  │   │ search + optional│  │ using ONLY    │   │ relevance &  │
 │ or a URL     │   │ in FAISS/Chroma│   │ cross-encoder rerank│ retrieved text│  │ context prec.│
 └──────────────┘   └────────────────┘   └────────────────┘   └───────────────┘   └──────────────┘
```

1. **Data Ingestion** — documents are loaded and split into overlapping chunks
   (`src/document_loader.py`, `src/text_splitter.py`). Files are hashed and
   tracked in a manifest so re-uploading an unchanged file skips re-embedding.
2. **Storage** — chunks are embedded and stored in FAISS (fast, local) or
   ChromaDB (persistent) (`src/embeddings.py`, `src/vector_store.py`).
3. **Query Processing** — the user's question is embedded and relevant chunks
   are retrieved, either via plain similarity search or MMR (Max Marginal
   Relevance, which reduces redundant/near-duplicate chunks). Optionally, a
   wider candidate pool is then re-scored by a cross-encoder reranker for
   higher precision (`RAGPipeline.retrieve`, `src/reranker.py`).
4. **Answer Generation** — the question + retrieved context is passed to an
   LLM (local HuggingFace model, OpenAI, or Groq), which must answer using
   only that context (`src/llm.py`, `src/rag_pipeline.py`).
5. **Evaluation** — every answer is scored for faithfulness, relevance, and
   context precision, either with RAGAS (LLM-judged) or a fast offline cosine-
   similarity heuristic (`src/evaluation.py`).

## Tech stack

| Layer | Choice |
|---|---|
| Orchestration | LangChain |
| Vector store | FAISS (default) or ChromaDB (switchable) |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5` (default, free, local) or OpenAI |
| LLM | Local HuggingFace `Qwen/Qwen2.5-1.5B-Instruct` (default, free), or OpenAI `gpt-4o-mini`, or Groq `llama-3.3-70b-versatile` (free, very fast) |
| Retrieval | MMR search by default; optional cross-encoder reranking (`BAAI/bge-reranker-base`) |
| Evaluation | RAGAS metrics, or a built-in offline heuristic |
| UI | Streamlit (primary), Gradio-compatible design |

## Project structure

```
smart-research-assistant/
├── app.py                  # Streamlit UI — main entry point
├── config.py                # All settings, read from .env
├── requirements.txt
├── .env.example              # Copy to .env and customize
├── src/
│   ├── document_loader.py    # Step 1: load PDF/TXT/DOCX/web
│   ├── text_splitter.py       # Step 1b: chunking
│   ├── embeddings.py           # Step 2: embedding models
│   ├── vector_store.py          # Step 2: FAISS / ChromaDB + MMR retriever
│   ├── reranker.py               # Step 3b: optional cross-encoder reranking
│   ├── llm.py                     # Step 4: answer generation (HF/OpenAI/Groq)
│   ├── memory.py                   # Multi-turn conversation memory
│   ├── rag_pipeline.py              # Ties it all together + ingestion caching
│   └── evaluation.py                 # Step 5: RAGAS / heuristic scoring
├── data/
│   ├── uploads/                       # Uploaded files land here
│   └── vectorstore/                    # Persisted vector index + ingest manifest
└── tests/
    └── test_pipeline.py                 # Offline unit tests (no downloads needed)
```

## Setup

```bash
cd smart-research-assistant
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
cp .env.example .env       # defaults work with zero configuration
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

> **First run note:** with the default settings (embeddings on `huggingface`,
> LLM on `groq`), the first query downloads only the small embedding model
> (`BAAI/bge-small-en-v1.5`, ~130MB) -- the 3GB local LLM is *not* downloaded
> unless you switch `LLM_PROVIDER=huggingface`. You'll need a free Groq API
> key (console.groq.com) for the default LLM provider to respond.

## Choosing an LLM provider

Three options, set via `LLM_PROVIDER` in `.env` (or the sidebar dropdown):

| Provider | Cost | Speed | Setup |
|---|---|---|---|
| `huggingface` (default) | Free | Slow-medium, runs on your CPU | Nothing to configure |
| `groq` | Free, no credit card | Very fast (hosted LPU hardware) | Free key at https://console.groq.com |
| `openai` | Paid | Fast | Key at https://platform.openai.com |

**Groq** is usually the best upgrade path if local inference feels slow: it's
free, needs no billing setup, and is dramatically faster since it runs on
Groq's own hardware instead of your machine.

```
LLM_PROVIDER=groq
GROQ_API_KEY=your-free-key-here
```

or paste the key into the sidebar after selecting `groq` -- no restart needed.

For OpenAI instead:
```
EMBEDDING_PROVIDER=openai   # optional, embeddings can stay on huggingface
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

### Local HuggingFace model options (free, no HF token required)

The default, `Qwen/Qwen2.5-1.5B-Instruct`, is a good balance of quality and
CPU speed with no gated license or Hugging Face token needed. Swap it via
`HF_LLM_MODEL` in `.env` -- the code auto-detects whether a model is causal
(Qwen, Phi) or seq2seq (flan-t5) and handles both:

| Model | Params | Quality | CPU speed | HF token needed? |
|---|---|---|---|---|
| `google/flan-t5-small` | 80M | Poor | Very fast | No |
| `google/flan-t5-large` | 780M | Decent | Medium | No |
| `Qwen/Qwen2.5-1.5B-Instruct` (default) | 1.5B | Good | Medium | No |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | Very good | Slower | No |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | Very good | Slow on CPU | No |

Avoid Llama and Gemma family models unless you're prepared to accept their
license on Hugging Face and pass a token -- they're free but gated, which
contradicts "no token required."

## Retrieval quality: MMR and reranking

Two knobs control how context is selected before it reaches the LLM:

- **`SEARCH_TYPE=mmr`** (default) -- pulls `FETCH_K` candidates and picks a
  diverse `TOP_K` subset, reducing near-duplicate chunks. Set to `similarity`
  for plain nearest-neighbor search instead.
- **`ENABLE_RERANK=true`** (off by default) -- retrieves a wider candidate
  pool (`RERANK_CANDIDATE_K`) and re-scores each one against the question
  with a cross-encoder (`BAAI/bge-reranker-base`), keeping only the best
  `TOP_K`. This is usually the single biggest accuracy lever after the
  embedding model itself, but adds an extra model download and some latency.

Both are also toggleable live from the Streamlit sidebar.

## Example usage

1. Upload your company's HR policy PDF in the **Documents** tab and click
   **Ingest documents**.
2. Switch to the **Chat** tab and ask: *"What is the maternity leave policy?"*
3. You'll get:
   - A concise, grounded answer (bullet points and short quotes where useful)
   - An expandable list of the exact source passages used
   - Reliability scores (faithfulness / relevance / context precision)

Ask a follow-up like *"What about for adoptive parents?"* -- the assistant
remembers the last few turns of conversation.

Re-uploading the same, unchanged file is fast -- it's skipped via a
content-hash manifest instead of being re-embedded.

## Running tests

```bash
pytest tests/ -v
```

The tests use a small deterministic fake embedding model, so they run with
no internet access and no API keys -- useful for CI. Covers chunking,
conversation memory, FAISS retrieval, MMR retrieval, and ingestion caching.

## Full configuration reference

Everything is a config flag in `.env` (or the sidebar dropdowns where noted):

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_PROVIDER` | `huggingface` | `huggingface` or `openai` |
| `LLM_PROVIDER` | `huggingface` | `huggingface`, `openai`, or `groq` (sidebar) |
| `HF_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | any HF sentence-transformers model |
| `HF_LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | any causal or seq2seq HF model |
| `DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `OPENAI_LLM_MODEL` | `gpt-4o-mini` | |
| `GROQ_LLM_MODEL` | `llama-3.3-70b-versatile` | free Groq model |
| `VECTOR_STORE` | `faiss` | `faiss` or `chroma` (sidebar) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `400` / `80` | smaller chunks improve retrieval precision |
| `TOP_K` | `5` | chunks handed to the LLM as context (sidebar) |
| `SEARCH_TYPE` | `mmr` | `mmr` or `similarity` |
| `FETCH_K` | `20` | MMR candidate pool size |
| `ENABLE_RERANK` | `false` | cross-encoder reranking on/off (sidebar) |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | |
| `RERANK_CANDIDATE_K` | `20` | candidates fed into the reranker |
| `MAX_NEW_TOKENS` | `200` | local model generation cap |
| `ENABLE_RAGAS` | `false` | LLM-judged eval vs. offline heuristic |

## Learning outcomes

Working through this codebase covers:
- How LLMs integrate with external knowledge via retrieval
- How vector databases and embeddings work in practice
- MMR and cross-encoder reranking as retrieval-quality levers
- Prompt engineering for grounded, hallucination-resistant answers
- Retrieval-based AI system design and modular pipeline architecture
- Why evaluation (faithfulness, relevance, context precision) matters for
  trustworthy AI systems

## Future enhancements

- Voice-based interaction
- Multi-language support
- Cloud deployment (AWS/GCP)
- Domain-specific assistants (legal, healthcare, finance)
- User authentication for multi-user/enterprise use
- Gradio alternative front-end (the pipeline in `src/` is UI-agnostic, so a
  `gradio_app.py` can reuse `RAGPipeline` directly)
- Token-level response streaming
- Quantized GGUF models via llama.cpp/Ollama for faster local CPU inference

## Troubleshooting

| Problem | Fix |
|---|---|
| `OPENAI_API_KEY is not set` error | Add a key to `.env`/sidebar, or switch that provider back to `huggingface` |
| `GROQ_API_KEY is not set` error | Add a free key from console.groq.com to `.env`/sidebar |
| Slow first response | Normal -- local model download/load happens once |
| Local model responses are slow | Expected on CPU for 1.5B+ models; switch `LLM_PROVIDER=groq` for free, fast hosted inference |
| Poor answer quality with local model | Try a larger `HF_LLM_MODEL`, enable reranking, or switch to `openai`/`groq` |
| `No documents have been ingested yet` | Upload + click "Ingest documents" in the Documents tab first |