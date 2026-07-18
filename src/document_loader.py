"""
document_loader.py
-------------------
Step 1 of the RAG pipeline: Data Ingestion.

Responsible for turning raw user input (uploaded files or a web URL) into
a list of LangChain `Document` objects with clean text + metadata
(source filename, page number, etc.), ready to be chunked.
"""

from __future__ import annotations

import os
from typing import List

from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".docx")


def load_pdf(path: str) -> List[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(path)
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = os.path.basename(path)
    return docs


def load_text(path: str) -> List[Document]:
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = os.path.basename(path)
    return docs


def load_docx(path: str) -> List[Document]:
    from langchain_community.document_loaders import Docx2txtLoader

    loader = Docx2txtLoader(path)
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = os.path.basename(path)
    return docs


def load_web(url: str) -> List[Document]:
    from langchain_community.document_loaders import WebBaseLoader

    loader = WebBaseLoader(url)
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = url
    return docs


def load_file(path: str) -> List[Document]:
    """Dispatch to the right loader based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext in (".txt", ".md"):
        return load_text(path)
    if ext == ".docx":
        return load_docx(path)
    raise ValueError(
        f"Unsupported file type: '{ext}'. Supported types: {SUPPORTED_EXTENSIONS}"
    )


def load_many(paths: List[str]) -> List[Document]:
    """Load multiple files (any mix of supported types) into one document list."""
    all_docs: List[Document] = []
    errors: List[str] = []
    for p in paths:
        try:
            all_docs.extend(load_file(p))
        except Exception as e:  # noqa: BLE001 - we want to surface all load errors, not crash the batch
            errors.append(f"{os.path.basename(p)}: {e}")
    if errors:
        # Non-fatal: surface which files failed so the UI can warn the user,
        # while still returning whatever loaded successfully.
        print("[document_loader] Some files failed to load:\n  " + "\n  ".join(errors))
    return all_docs
