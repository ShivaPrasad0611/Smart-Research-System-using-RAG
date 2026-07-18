"""
evaluation.py
-------------
Step 5 of the RAG pipeline: Evaluation.

Two modes:

1. RAGAS metrics (faithfulness, answer relevancy, context precision) -
   the "proper" way, but it uses an LLM judge under the hood, so it's
   slower and, in the default config, needs an OpenAI key. Enable with
   ENABLE_RAGAS=true in .env.

2. A lightweight heuristic evaluator (default) - uses the embedding model
   that's already loaded to compute cosine-similarity-based proxies for
   the same three questions RAGAS asks:
     - Faithfulness  : how much does the answer overlap with the retrieved
                       context (checks it isn't "hallucinating")?
     - Relevance     : how similar is the answer to the original question?
     - Context Prec. : how similar is the retrieved context to the question?

   This keeps the app fully functional offline / without any paid API,
   while still giving the user a meaningful reliability signal.
"""

from __future__ import annotations

from typing import List

import numpy as np

from config import settings


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _heuristic_eval(question: str, answer: str, contexts: List[str], embeddings) -> dict:
    q_vec = embeddings.embed_query(question)
    a_vec = embeddings.embed_query(answer)
    ctx_vecs = [embeddings.embed_query(c) for c in contexts] if contexts else []

    relevance = _cosine(q_vec, a_vec)
    faithfulness = (
        max(_cosine(a_vec, c) for c in ctx_vecs) if ctx_vecs else 0.0
    )
    context_precision = (
        float(np.mean([_cosine(q_vec, c) for c in ctx_vecs])) if ctx_vecs else 0.0
    )

    return {
        "method": "heuristic (cosine similarity proxy)",
        "faithfulness": round(faithfulness, 3),
        "answer_relevance": round(relevance, 3),
        "context_precision": round(context_precision, 3),
    }


def _ragas_eval(question: str, answer: str, contexts: List[str]) -> dict:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision

    dataset = Dataset.from_dict(
        {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
    )
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
    df = result.to_pandas()
    row = df.iloc[0]
    return {
        "method": "ragas (LLM-judged)",
        "faithfulness": round(float(row.get("faithfulness", 0.0)), 3),
        "answer_relevance": round(float(row.get("answer_relevancy", 0.0)), 3),
        "context_precision": round(float(row.get("context_precision", 0.0)), 3),
    }


def evaluate_response(question: str, answer: str, contexts: List[str], embeddings) -> dict:
    if settings.enable_ragas:
        try:
            return _ragas_eval(question, answer, contexts)
        except Exception as e:  # noqa: BLE001 - always fall back rather than break the UI
            fallback = _heuristic_eval(question, answer, contexts, embeddings)
            fallback["note"] = f"RAGAS evaluation failed ({e}); showing heuristic scores instead."
            return fallback
    return _heuristic_eval(question, answer, contexts, embeddings)
