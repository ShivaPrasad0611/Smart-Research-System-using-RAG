"""
rag_pipeline.py
----------------
Ties every stage together into one `RAGPipeline` class:

    1. ingest(paths)        -> load, chunk, embed, store
    2. query(question)      -> retrieve, generate, (optionally) evaluate

This is the main object the Streamlit app (or any other UI / script)
talks to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from config import settings
from src.document_loader import load_many, load_web
from src.text_splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import build_vector_store, save_vector_store, load_vector_store
from src.llm import get_llm
from src.memory import ConversationMemory
from src.evaluation import evaluate_response
from src.reranker import rerank


SYSTEM_PROMPT = """You are a careful research assistant. Answer the user's \
question using ONLY the information in the provided context. If the answer \
is not contained in the context, say you don't have enough information — \
do not make anything up.

{history}

Context:
{context}

Question: {question}

Answer in 2-3 concise sentences, then note which source(s) you used. Do not \
copy long passages from the context verbatim."""


@dataclass
class RAGResult:
    answer: str
    sources: List[Document]
    evaluation: dict = field(default_factory=dict)


def _strip_echoed_prompt(answer: str, formatted_prompt: str) -> str:
    """Some local pipelines ignore return_full_text and echo the whole input
    prompt back before the actual answer. This strips that echo so the UI
    never shows the raw prompt to the user, regardless of model/version quirks.
    """
    answer = answer.strip()
    if formatted_prompt.strip() and formatted_prompt.strip() in answer:
        answer = answer.split(formatted_prompt.strip(), 1)[-1].strip()
    # Also guard against a partial echo ending right at the last instruction line.
    anchor = "copy long passages from the context verbatim."
    if anchor in answer:
        answer = answer.split(anchor, 1)[-1].strip()
    return answer if answer else "I don't have enough information to answer that from the provided context."


class RAGPipeline:
    def __init__(
        self,
        embedding_provider: Optional[str] = None,
        llm_provider: Optional[str] = None,
        vector_store_type: Optional[str] = None,
        top_k: Optional[int] = None,
        search_type: Optional[str] = None,
        enable_rerank: Optional[bool] = None,
    ):
        self.embedding_provider = embedding_provider or settings.embedding_provider
        self.llm_provider = llm_provider or settings.llm_provider
        self.vector_store_type = vector_store_type or settings.vector_store
        self.top_k = top_k or settings.top_k
        # "similarity" or "mmr" (Max Marginal Relevance - reduces redundant/
        # near-duplicate chunks in the returned context).
        self.search_type = search_type or settings.search_type
        self.enable_rerank = settings.enable_rerank if enable_rerank is None else enable_rerank

        self.embeddings = get_embeddings(self.embedding_provider)
        self._llm = None  # lazily created on first query (avoids loading a local
        # model until it's actually needed)

        self.vector_store = None
        self.memory = ConversationMemory()
        self.num_chunks = 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_files(self, paths: List[str]) -> int:
        """Load, chunk, embed, and index a batch of files. Returns chunk count."""
        docs = load_many(paths)
        return self._ingest_documents(docs)

    def ingest_url(self, url: str) -> int:
        docs = load_web(url)
        return self._ingest_documents(docs)

    def _ingest_documents(self, docs: List[Document]) -> int:
        if not docs:
            return 0
        chunks = split_documents(docs)
        if self.vector_store is None:
            self.vector_store = build_vector_store(chunks, self.embeddings, self.vector_store_type)
        else:
            self.vector_store.add_documents(chunks)
        self.num_chunks += len(chunks)
        return len(chunks)

    def persist(self) -> None:
        if self.vector_store is not None:
            save_vector_store(self.vector_store, self.vector_store_type)

    def load_persisted(self) -> bool:
        store = load_vector_store(self.embeddings, self.vector_store_type)
        if store is not None:
            self.vector_store = store
            return True
        return False

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm(self.llm_provider)
        return self._llm

    def retrieve(self, question: str) -> List[Document]:
        if self.vector_store is None:
            raise ValueError("No documents have been ingested yet.")

        # When reranking is on, over-fetch a candidate pool and let the
        # cross-encoder pick the best top_k, rather than trusting the vector
        # store's initial (bi-encoder) ranking as final.
        candidate_k = settings.rerank_candidate_k if self.enable_rerank else self.top_k

        if self.search_type == "mmr":
            fetch_k = max(settings.fetch_k, candidate_k)
            candidates = self.vector_store.max_marginal_relevance_search(
                question, k=candidate_k, fetch_k=fetch_k
            )
        else:
            candidates = self.vector_store.similarity_search(question, k=candidate_k)

        if self.enable_rerank:
            return rerank(question, candidates, top_k=self.top_k)
        return candidates[: self.top_k]

    def query(self, question: str, use_memory: bool = True, run_evaluation: bool = False) -> RAGResult:
        sources = self.retrieve(question)
        context = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in sources
        )
        history = self.memory.as_prompt_block() if use_memory else ""

        prompt = PromptTemplate.from_template(SYSTEM_PROMPT)
        formatted_prompt = prompt.format(history=history, context=context, question=question)

        raw = self.llm.invoke(formatted_prompt)
        answer = raw.content if hasattr(raw, "content") else str(raw)
        answer = _strip_echoed_prompt(answer, formatted_prompt)

        if use_memory:
            self.memory.add(question, answer)

        evaluation = {}
        if run_evaluation:
            evaluation = evaluate_response(
                question=question,
                answer=answer,
                contexts=[d.page_content for d in sources],
                embeddings=self.embeddings,
            )

        return RAGResult(answer=answer, sources=sources, evaluation=evaluation)
