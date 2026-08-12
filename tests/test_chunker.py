"""Tests for src/chunker.py"""

import pytest
from langchain_core.documents import Document

from src.chunker import get_chunk_stats, split_documents


# ── split_documents ─────────────────────────────────────────────────────────

def test_split_documents_empty_raises():
    with pytest.raises(ValueError, match="Cannot split an empty document list"):
        split_documents([])


def test_split_documents_basic():
    # Create a document long enough to be split
    long_text = "This is a sentence. " * 100  # ~2000 chars
    docs = [Document(page_content=long_text, metadata={"source": "test.pdf", "page": 0})]

    chunks = split_documents(docs, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    # Each chunk should have enriched metadata
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i
        assert "chunk_char_count" in chunk.metadata
        assert chunk.metadata["source"] == "test.pdf"


def test_split_documents_preserves_metadata():
    docs = [
        Document(
            page_content="Some text. " * 50,
            metadata={"source": "report.pdf", "page": 3, "file_hash": "abc123"},
        )
    ]
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)

    for chunk in chunks:
        assert chunk.metadata["source"] == "report.pdf"
        assert chunk.metadata["page"] == 3
        assert chunk.metadata["file_hash"] == "abc123"


def test_split_documents_short_text_single_chunk():
    docs = [Document(page_content="Short text", metadata={"source": "s.pdf"})]
    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)

    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text"
    assert chunks[0].metadata["chunk_index"] == 0


# ── get_chunk_stats ─────────────────────────────────────────────────────────

def test_get_chunk_stats_empty():
    stats = get_chunk_stats([])
    assert stats["total_chunks"] == 0
    assert stats["avg_chunk_size"] == 0


def test_get_chunk_stats_with_chunks():
    chunks = [
        Document(page_content="a" * 100, metadata={}),
        Document(page_content="b" * 200, metadata={}),
        Document(page_content="c" * 300, metadata={}),
    ]
    stats = get_chunk_stats(chunks)

    assert stats["total_chunks"] == 3
    assert stats["avg_chunk_size"] == 200
    assert stats["min_chunk_size"] == 100
    assert stats["max_chunk_size"] == 300
    assert stats["total_chars"] == 600
