"""
retriever.py — Vector Store & Retrieval Module
=================================================
Responsibility:
  1. Build a vector store (FAISS or Chroma) from embedded chunks.
  2. Expose a top-k similarity retriever for use in the RAG chain.
  3. Persist FAISS indexes to disk keyed by file MD5 hash to avoid
     re-embedding the same file on every page reload.
  4. Return relevance (cosine similarity) scores alongside retrieved chunks
     so the UI can display High / Medium / Low confidence badges.
  5. (Optional) Hybrid search: combine dense vector + BM25 keyword retrieval
     via EnsembleRetriever when ENABLE_HYBRID_SEARCH=True in config.

All config values (VECTOR_STORE_TYPE, TOP_K, thresholds, etc.) are read
from config.py — nothing is hardcoded here.
"""

from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

import config


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_vector_store(
    chunks: List[Document],
    embeddings: Embeddings,
    file_hash: str | None = None,
) -> VectorStore:
    """
    Embed chunks and store them in FAISS or Chroma (per config.VECTOR_STORE_TYPE).

    For FAISS: if file_hash is provided and a saved index exists on disk for
    that hash, the saved index is loaded instead of re-embedding — saving API
    calls and time.

    Args:
        chunks:     Output of chunker.split_documents().
        embeddings: Output of embeddings.get_embedding_model().
        file_hash:  MD5 digest of the source PDF bytes (optional).
                    Used for FAISS persistence keying.

    Returns:
        Populated VectorStore ready for similarity search.
    """
    store_type = config.VECTOR_STORE_TYPE.lower()

    if store_type == "faiss":
        return _build_faiss(chunks, embeddings, file_hash)
    elif store_type == "chroma":
        return _build_chroma(chunks, embeddings, file_hash)
    else:
        raise ValueError(
            f"Unsupported VECTOR_STORE_TYPE: '{store_type}'. "
            "Choose 'faiss' or 'chroma'."
        )


def get_retriever(
    vector_store: VectorStore,
    chunks: List[Document] | None = None,
    top_k: int = config.TOP_K,
):
    """
    Return a LangChain retriever. If ENABLE_HYBRID_SEARCH is True AND chunks
    are provided, returns an EnsembleRetriever combining dense + BM25.
    Otherwise returns a standard similarity retriever.

    Args:
        vector_store: Populated VectorStore from build_vector_store().
        chunks:       Original chunk list (needed for BM25 if hybrid is on).
        top_k:        Number of chunks to return per query.

    Returns:
        LangChain BaseRetriever (compatible with LCEL chains).
    """
    # Standard dense retriever
    dense_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    # If hybrid search is enabled and we have chunks, combine with BM25
    if config.ENABLE_HYBRID_SEARCH and chunks:
        return _build_hybrid_retriever(dense_retriever, chunks, top_k)

    return dense_retriever


def retrieve_with_scores(
    vector_store: VectorStore,
    query: str,
    top_k: int = config.TOP_K,
) -> List[Tuple[Document, float]]:
    """
    Perform similarity search and return (Document, score) pairs.

    The score is a cosine similarity in [0.0, 1.0] — higher is more relevant.
    This is used by pipeline.py to attach relevance badges and enforce the
    minimum confidence threshold.

    Args:
        vector_store: Populated VectorStore.
        query:        User's question string.
        top_k:        Number of results to return.

    Returns:
        List of (Document, float) tuples, sorted by descending score.
    """
    return vector_store.similarity_search_with_relevance_scores(
        query=query,
        k=top_k,
    )


def score_label(score: float) -> str:
    """
    Convert a cosine similarity score to a human-readable badge label.

    Thresholds are defined in config.py so they can be tuned without
    touching this module.

    Args:
        score: Cosine similarity float in [0.0, 1.0].

    Returns:
        "🟢 High", "⚡ Medium", or "🔴 Low".
    """
    if score >= config.HIGH_SCORE_THRESHOLD:
        return "🟢 High"
    elif score >= config.MID_SCORE_THRESHOLD:
        return "⚡ Medium"
    return "🔴 Low"


def save_vector_store(vector_store: VectorStore, file_hash: str) -> None:
    """
    Persist a FAISS vector store to disk. Chroma persists automatically.

    Args:
        vector_store: The FAISS VectorStore to save.
        file_hash:    MD5 hash used as the directory name.
    """
    if config.VECTOR_STORE_TYPE.lower() != "faiss":
        return  # Chroma auto-persists

    index_path = _get_store_path(file_hash)
    index_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_path))


def load_vector_store(file_hash: str, embeddings: Embeddings) -> VectorStore | None:
    """
    Load a previously persisted FAISS vector store from disk.

    Args:
        file_hash:  MD5 hash of the original file(s).
        embeddings: Embedding model (must match what was used to build the index).

    Returns:
        VectorStore if found on disk, None otherwise.
    """
    from langchain_community.vectorstores import FAISS

    index_path = _get_store_path(file_hash)

    if not index_path.exists():
        return None

    try:
        return FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        # Dimension mismatch, corrupt index, etc. — caller should rebuild.
        return None


def store_exists(file_hash: str) -> bool:
    """
    Check if a persisted vector store exists for this file hash.

    Args:
        file_hash: MD5 hash string.

    Returns:
        True if a cached index directory exists on disk.
    """
    return _get_store_path(file_hash).exists()


# ─────────────────────────────────────────────────────────────────────────────
# Private builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_faiss(
    chunks: List[Document],
    embeddings: Embeddings,
    file_hash: str | None,
) -> VectorStore:
    """
    Build or load a FAISS index.

    FAISS (Facebook AI Similarity Search) stores vectors in memory for fast
    approximate nearest-neighbour search.  We serialise it to disk so the
    same file is never embedded twice.

    Index path: <VECTOR_STORE_DIR>/<file_hash>/
    """
    from langchain_community.vectorstores import FAISS

    # Try loading a cached index first
    if file_hash:
        cached = load_vector_store(file_hash, embeddings)
        if cached is not None:
            return cached

    # Build fresh index from chunks
    store = FAISS.from_documents(chunks, embeddings)

    # Persist if we have a hash to key it by
    if file_hash:
        save_vector_store(store, file_hash)

    return store


def _build_chroma(
    chunks: List[Document],
    embeddings: Embeddings,
    file_hash: str | None,
) -> VectorStore:
    """
    Build a Chroma vector store with persistence.

    Chroma stores vectors on disk automatically via its built-in SQLite
    backend. The persist_directory is derived from VECTOR_STORE_DIR + the
    file hash (or a default "chroma_db" subfolder).
    """
    from langchain_community.vectorstores import Chroma

    # Use file_hash as collection name for cache separation, or a default
    collection_name = f"rag_{file_hash}" if file_hash else "rag_default"
    persist_dir = str(Path(config.VECTOR_STORE_DIR) / "chroma_db")

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )


def _build_hybrid_retriever(dense_retriever, chunks: List[Document], top_k: int):
    """
    Combine dense vector retriever + BM25 keyword retriever using
    LangChain's EnsembleRetriever.

    Why hybrid?
      Dense retrieval is great for semantic similarity but can miss exact
      keyword matches. BM25 catches those. The ensemble balances both.

    Args:
        dense_retriever: The FAISS/Chroma similarity retriever.
        chunks:          Original document chunks (BM25 needs raw text).
        top_k:           Number of results from each retriever.

    Returns:
        EnsembleRetriever that merges results from both retrievers.
    """
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers import EnsembleRetriever

    # BM25 retriever over the raw chunk texts
    bm25_retriever = BM25Retriever.from_documents(chunks, k=top_k)

    # Combine with configurable weights
    weights = config.HYBRID_SEARCH_WEIGHTS

    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=weights,
    )


def _get_store_path(file_hash: str) -> Path:
    """
    Return the directory path for a persisted vector store keyed by file hash.

    Structure: <VECTOR_STORE_DIR>/<file_hash>/
    """
    return Path(config.VECTOR_STORE_DIR) / file_hash