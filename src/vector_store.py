"""
vector_store.py
----------------
Step 2 of the RAG pipeline: Storage.

Wraps two interchangeable vector store backends behind one simple API:

- FAISS: fast, in-memory, saved to a local folder. Great for prototyping
  and small-to-medium document collections.
- ChromaDB: persistent by design, a bit more suited to long-lived,
  production-like deployments.

Switching backends is a one-line config change (VECTOR_STORE=faiss|chroma).
"""

from __future__ import annotations

import os
from typing import List

from langchain_core.documents import Document

from config import settings


def build_vector_store(chunks: List[Document], embeddings, store_type: str | None = None):
    store_type = (store_type or settings.vector_store).lower()

    if store_type == "faiss":
        from langchain_community.vectorstores import FAISS

        return FAISS.from_documents(chunks, embeddings)

    if store_type == "chroma":
        from langchain_community.vectorstores import Chroma

        persist_dir = settings.vector_store_dir
        os.makedirs(persist_dir, exist_ok=True)
        return Chroma.from_documents(
            chunks, embeddings, persist_directory=persist_dir, collection_name="research_assistant"
        )

    raise ValueError(f"Unknown vector store type: '{store_type}'. Use 'faiss' or 'chroma'.")


def save_vector_store(store, store_type: str | None = None, path: str | None = None) -> None:
    """FAISS needs an explicit save; Chroma persists automatically."""
    store_type = (store_type or settings.vector_store).lower()
    if store_type == "faiss":
        path = path or settings.vector_store_dir
        os.makedirs(path, exist_ok=True)
        store.save_local(path)
    # Chroma with persist_directory set already writes to disk on every add.


def load_vector_store(embeddings, store_type: str | None = None, path: str | None = None):
    """Reload a previously-saved vector store (skips re-embedding documents)."""
    store_type = (store_type or settings.vector_store).lower()
    path = path or settings.vector_store_dir

    if store_type == "faiss":
        from langchain_community.vectorstores import FAISS

        if not os.path.exists(path):
            return None
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

    if store_type == "chroma":
        from langchain_community.vectorstores import Chroma

        if not os.path.exists(path):
            return None
        return Chroma(
            persist_directory=path, embedding_function=embeddings, collection_name="research_assistant"
        )

    raise ValueError(f"Unknown vector store type: '{store_type}'.")
