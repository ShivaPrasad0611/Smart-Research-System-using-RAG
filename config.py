"""
config.py
---------
Central configuration for the Smart Research Assistant.

All tunable settings live here so the rest of the codebase never hardcodes
provider names, model names, or paths. Values are read from environment
variables (via a .env file) with sensible defaults, so the project runs
out-of-the-box in a fully local/offline mode and can be upgraded to
OpenAI-backed generation just by setting a few env vars.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()  # loads variables from a .env file in the project root, if present


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # ---- Providers -------------------------------------------------
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")

    # ---- Model names -------------------------------------------------
    hf_embedding_model: str = os.getenv(
        "HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
    )
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    # Experimental: Groq's embeddings endpoint isn't officially confirmed stable.
    groq_embedding_model: str = os.getenv("GROQ_EMBEDDING_MODEL", "nomic-embed-text-v1_5")

    # Device for local HF models: "cpu", "cuda", or "auto" (auto-detects a GPU
    # if torch reports one available, otherwise falls back to cpu).
    device: str = os.getenv("DEVICE", "auto")

    hf_llm_model: str = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    openai_llm_model: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

    # Groq -- free, very fast hosted inference (LPU hardware), OpenAI-compatible
    # API, no credit card required. Get a key at https://console.groq.com
    groq_llm_model: str = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    # ---- API keys -------------------------------------------------
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    huggingfacehub_api_token: str = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

    # ---- Vector store -------------------------------------------------
    # "faiss" (fast, in-memory/local file, great for prototyping)
    # "chroma" (persistent, better for longer-lived / production-like use)
    vector_store: str = os.getenv("VECTOR_STORE", "faiss")
    vector_store_dir: str = os.getenv("VECTOR_STORE_DIR", "data/vectorstore")

    # ---- Chunking -------------------------------------------------
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))

    # ---- Retrieval -------------------------------------------------
    # Final number of chunks handed to the LLM as context.
    top_k: int = int(os.getenv("TOP_K", "5"))
    # "similarity" (plain nearest-neighbor) or "mmr" (Max Marginal Relevance,
    # trades a little top-1 precision for less redundant/duplicate context).
    search_type: str = os.getenv("SEARCH_TYPE", "mmr")
    # How many candidates MMR pulls from the vector store before diversifying
    # down to top_k. Only used when search_type == "mmr".
    fetch_k: int = int(os.getenv("FETCH_K", "20"))

    # ---- Reranking -------------------------------------------------
    # If enabled, retrieve more candidates than top_k, then rerank them with
    # a cross-encoder and keep only the best top_k. Usually the single
    # biggest accuracy lever after the embedding model itself.
    enable_rerank: bool = _get_bool("ENABLE_RERANK", False)
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    # Candidate pool size fed into the reranker (independent of fetch_k).
    rerank_candidate_k: int = int(os.getenv("RERANK_CANDIDATE_K", "20"))

    # ---- Generation limits -------------------------------------------------
    # Kept modest so the free local model answers quickly and concisely
    # instead of rambling. Raise this if you switch to OpenAI and want
    # longer answers.
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "200"))

    # ---- Evaluation -------------------------------------------------
    # RAGAS evaluation uses an LLM judge (best with OpenAI). If disabled,
    # or if no OpenAI key is present, a lightweight heuristic evaluator
    # (cosine-similarity based) is used instead so the app always works.
    enable_ragas: bool = _get_bool("ENABLE_RAGAS", False)

    # ---- Paths -------------------------------------------------
    upload_dir: str = os.getenv("UPLOAD_DIR", "data/uploads")


settings = Settings()
