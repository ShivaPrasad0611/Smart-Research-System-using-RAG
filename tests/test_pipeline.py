"""
test_pipeline.py
-----------------
Offline unit tests. Uses a small deterministic fake embedding model
(hash-based) instead of downloading a real sentence-transformers model,
so these tests run anywhere with no network access and no API keys.

Run with:
    pytest tests/test_pipeline.py -v
"""

import hashlib
import os
import sys

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.text_splitter import split_documents
from src.memory import ConversationMemory
from src.vector_store import build_vector_store


STOPWORDS = {"the", "a", "an", "is", "are", "of", "what", "to", "within", "every"}


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words style embedding, no downloads required."""

    DIM = 512

    def _embed(self, text: str):
        vec = np.zeros(self.DIM)
        for word in text.lower().strip("?.").split():
            word = word.strip(".,?")
            if word in STOPWORDS or not word:
                continue
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % self.DIM] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def test_split_documents_creates_overlapping_chunks():
    long_text = "Paragraph one about leave policy. " * 50
    docs = [Document(page_content=long_text, metadata={"source": "policy.txt"})]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=40)

    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.metadata["chunk_id"] == i
        assert c.metadata["source"] == "policy.txt"


def test_conversation_memory_bounds_history():
    mem = ConversationMemory(max_turns=2)
    mem.add("q1", "a1")
    mem.add("q2", "a2")
    mem.add("q3", "a3")

    assert len(mem.history) == 2
    assert mem.history[0] == ("q2", "a2")
    assert "q3" in mem.as_prompt_block()
    assert "q1" not in mem.as_prompt_block()


def test_faiss_vector_store_retrieves_relevant_chunk():
    docs = [
        Document(page_content="The maternity leave policy grants 26 weeks of paid leave.", metadata={"source": "hr.txt"}),
        Document(page_content="The office WiFi password is changed every quarter.", metadata={"source": "it.txt"}),
        Document(page_content="Employees can claim travel reimbursement within 30 days.", metadata={"source": "finance.txt"}),
    ]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=0)
    store = build_vector_store(chunks, FakeEmbeddings(), store_type="faiss")

    results = store.similarity_search("What is the maternity leave policy?", k=1)
    assert len(results) == 1
    assert results[0].metadata["source"] == "hr.txt"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
