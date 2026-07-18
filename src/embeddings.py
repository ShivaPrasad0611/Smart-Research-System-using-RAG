"""
embeddings.py
-------------
Provides an embeddings model based on config.settings.embedding_provider.

- "huggingface": runs locally via sentence-transformers. Free, offline-capable
  (after the first model download), no API key needed. Default choice.
- "openai": uses OpenAI's embedding API. Higher quality on some benchmarks,
  requires OPENAI_API_KEY.
"""

from __future__ import annotations

from config import settings


def _resolve_device(requested: str) -> str:
    """Turn 'auto' into 'cuda' or 'cpu' depending on what's actually available,
    so the same config works unmodified on a laptop or a GPU box."""
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - torch not importable yet, etc.
        return "cpu"


def get_embeddings(provider: str | None = None):
    provider = (provider or settings.embedding_provider).lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.openai_api_key:
            raise ValueError(
                "EMBEDDING_PROVIDER is 'openai' but OPENAI_API_KEY is not set. "
                "Add it to your .env file, or switch EMBEDDING_PROVIDER to 'huggingface'."
            )
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        device = _resolve_device(settings.device)
        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": True}

        if "bge" in settings.hf_embedding_model.lower():
            # bge models are trained with a retrieval instruction prefix on
            # the query side only ("Represent this sentence for searching
            # relevant passages: ..."); adding it measurably improves
            # retrieval quality. The installed langchain_huggingface version
            # doesn't expose a query_instruction constructor kwarg on
            # HuggingFaceEmbeddings (that's pydantic-validated and rejects
            # unknown fields), so we subclass instead and only override
            # embed_query - embed_documents (used for indexing) is untouched.
            class _BgeEmbeddings(HuggingFaceEmbeddings):
                def embed_query(self, text: str):
                    prefix = "Represent this sentence for searching relevant passages: "
                    return super().embed_query(prefix + text)

            return _BgeEmbeddings(
                model_name=settings.hf_embedding_model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )

        return HuggingFaceEmbeddings(
            model_name=settings.hf_embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    raise ValueError(f"Unknown embedding provider: '{provider}'. Use 'huggingface' or 'openai'.")
