"""
embeddings.py — Embedding Model Factory
=========================================
Responsibility:
  Return a LangChain-compatible Embeddings object based on the provider
  configured in config.py (EMBEDDING_PROVIDER).

Supported providers:
  "openai"      → OpenAI text-embedding-3-small (requires OPENAI_API_KEY)
  "huggingface" → sentence-transformers/all-MiniLM-L6-v2 (runs locally, free)

Why a factory?
  The rest of the codebase never needs to know WHICH embedding model is active.
  retriever.py just calls get_embedding_model() and gets back an object with
  .embed_documents() and .embed_query() methods. Swapping providers is a single
  config change — zero code changes required.

HuggingFace as fallback:
  If you don't have an OpenAI key or want to avoid API costs during development,
  set EMBEDDING_PROVIDER="huggingface" in your .env file. The all-MiniLM-L6-v2
  model runs entirely on your CPU, produces 384-dim vectors, and is surprisingly
  good for a free local model.
"""

from langchain_core.embeddings import Embeddings

import config


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_embedding_model() -> Embeddings:
    """
    Factory function: return the configured embedding model.

    Reads EMBEDDING_PROVIDER from config.py and instantiates the
    corresponding LangChain Embeddings object.

    Returns:
        A LangChain Embeddings object (OpenAIEmbeddings or HuggingFaceEmbeddings).

    Raises:
        ValueError: If the API key is missing for the selected provider,
                    or if the provider string is not recognized.
    """
    provider = config.EMBEDDING_PROVIDER.lower().strip()

    if provider == "openai":
        return _get_openai_embeddings()
    elif provider == "huggingface":
        return _get_huggingface_embeddings()
    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER: '{provider}'. "
            f"Supported values: 'openai', 'huggingface'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Private factory methods
# ─────────────────────────────────────────────────────────────────────────────

def _get_openai_embeddings() -> Embeddings:
    """
    Instantiate OpenAI embeddings (text-embedding-3-small by default).

    Why text-embedding-3-small?
      It's OpenAI's latest cost-efficient embedding model with 1536 dimensions.
      Good balance of quality and speed for RAG applications.

    Returns:
        OpenAIEmbeddings instance.

    Raises:
        ValueError: If OPENAI_API_KEY is not set in the environment.
    """
    if not config.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is required when EMBEDDING_PROVIDER='openai'.\n"
            "Fix: Add OPENAI_API_KEY=sk-... to your .env file,\n"
            "  or switch to free local embeddings by setting EMBEDDING_PROVIDER=huggingface"
        )

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=config.OPENAI_EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )


def _get_huggingface_embeddings() -> Embeddings:
    """
    Instantiate HuggingFace embeddings (all-MiniLM-L6-v2 by default).

    Runs entirely locally on CPU — no API key needed.
    First call downloads the model (~80MB), subsequent calls use the cache.

    Why all-MiniLM-L6-v2?
      - 384-dim vectors (compact, fast similarity search)
      - Trained on 1B+ sentence pairs
      - Top performer among small models on MTEB benchmark

    Returns:
        HuggingFaceEmbeddings instance.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.HF_EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},       # Force CPU to avoid CUDA issues
        encode_kwargs={"normalize_embeddings": True},  # L2-normalize for cosine sim
    )
