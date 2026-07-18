"""
reranker.py
-----------
Optional reranking stage that sits between the vector store's initial
(cheap, approximate) retrieval and the LLM.

Bi-encoder retrieval (what FAISS/Chroma do) embeds the query and each chunk
independently, so it's fast but misses fine-grained query/chunk interactions.
A cross-encoder reranker feeds the (query, chunk) pair into one model
together, which is much slower per pair but far more accurate at judging
relevance - so we only run it over a small shortlist (rerank_candidate_k
candidates), not the whole index.

This is usually the single highest-leverage accuracy improvement available
after picking a good embedding model, because it fixes ranking mistakes the
initial retrieval already made.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from config import settings

_model_cache: dict = {}


def _get_cross_encoder(model_name: str):
    """Cross-encoders are slow to load, so keep one instance per process."""
    if model_name not in _model_cache:
        from sentence_transformers import CrossEncoder

        from src.embeddings import _resolve_device

        device = _resolve_device(settings.device)
        _model_cache[model_name] = CrossEncoder(model_name, device=device)
    return _model_cache[model_name]


def rerank(
    query: str,
    documents: List[Document],
    top_k: int,
    model_name: str | None = None,
) -> List[Document]:
    """Rerank `documents` by relevance to `query`, returning the best `top_k`.

    Falls back to the original order (truncated to top_k) if the reranker
    model can't be loaded (e.g. offline / dependency missing), so the app
    never breaks just because reranking is unavailable.
    """
    if not documents:
        return documents

    model_name = model_name or settings.reranker_model

    try:
        cross_encoder = _get_cross_encoder(model_name)
    except Exception as e:  # noqa: BLE001 - degrade gracefully, don't crash the query
        print(f"[reranker] Falling back to unranked order ({e})")
        return documents[:top_k]

    pairs = [(query, doc.page_content) for doc in documents]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]
