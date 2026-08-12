"""Tests for src/retriever.py"""

from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document

import config
from src.retriever import score_label, store_exists, _get_store_path


# ── score_label ─────────────────────────────────────────────────────────────

def test_score_label_high():
    assert "High" in score_label(0.80)
    assert "High" in score_label(0.75)  # boundary


def test_score_label_medium():
    assert "Medium" in score_label(0.60)
    assert "Medium" in score_label(0.50)  # boundary


def test_score_label_low():
    assert "Low" in score_label(0.30)
    assert "Low" in score_label(0.0)
    assert "Low" in score_label(0.49)


# ── store_exists ────────────────────────────────────────────────────────────

def test_store_exists_false(tmp_path):
    with patch.object(config, "VECTOR_STORE_DIR", str(tmp_path)):
        assert store_exists("nonexistent_hash") is False


def test_store_exists_true(tmp_path):
    hash_dir = tmp_path / "abc123"
    hash_dir.mkdir()
    with patch.object(config, "VECTOR_STORE_DIR", str(tmp_path)):
        assert store_exists("abc123") is True


# ── _get_store_path ─────────────────────────────────────────────────────────

def test_get_store_path():
    with patch.object(config, "VECTOR_STORE_DIR", "/data/vectors"):
        path = _get_store_path("hash123")
        assert str(path).replace("\\", "/") == "/data/vectors/hash123"


# ── retrieve_with_scores ────────────────────────────────────────────────────

def test_retrieve_with_scores_delegates_to_vector_store():
    mock_store = MagicMock()
    doc = Document(page_content="test", metadata={})
    mock_store.similarity_search_with_relevance_scores.return_value = [(doc, 0.85)]

    from src.retriever import retrieve_with_scores
    results = retrieve_with_scores(mock_store, "query", top_k=3)

    mock_store.similarity_search_with_relevance_scores.assert_called_once_with(
        query="query", k=3
    )
    assert len(results) == 1
    assert results[0][1] == 0.85


# ── rerank ──────────────────────────────────────────────────────────────────

def test_rerank_empty_returns_empty():
    from src.retriever import rerank
    result = rerank("question", [], top_n=3)
    assert result == []


def test_rerank_reorders_by_cross_encoder_score():
    import sys

    # Create a mock sentence_transformers module so the import inside rerank works
    mock_st_module = MagicMock()
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.2, 0.95]
    mock_st_module.CrossEncoder.return_value = mock_model
    sys.modules["sentence_transformers"] = mock_st_module

    try:
        from src.retriever import rerank

        doc_a = Document(page_content="low relevance", metadata={})
        doc_b = Document(page_content="high relevance", metadata={})

        scored_results = [(doc_a, 0.9), (doc_b, 0.5)]

        reranked = rerank("question", scored_results, top_n=2)

        assert len(reranked) == 2
        # doc_b should now be first (higher cross-encoder score)
        assert reranked[0][0].page_content == "high relevance"
        assert reranked[1][0].page_content == "low relevance"
    finally:
        sys.modules.pop("sentence_transformers", None)
