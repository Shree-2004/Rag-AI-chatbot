"""Tests for src/pipeline.py"""

from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document

import config
from src.pipeline import ask, _format_context, _format_history, _count_tokens, _truncate_history_by_tokens


# ── _format_context ─────────────────────────────────────────────────────────

def test_format_context_single_chunk():
    doc = Document(
        page_content="The answer is 42.",
        metadata={"source": "guide.pdf", "page": 5},
    )
    result = _format_context([(doc, 0.87)])

    assert "guide.pdf" in result
    assert "Page: 5" in result
    assert "0.87" in result
    assert "The answer is 42." in result


def test_format_context_multiple_chunks():
    docs = [
        (Document(page_content="First", metadata={"source": "a.pdf", "page": 1}), 0.9),
        (Document(page_content="Second", metadata={"source": "b.pdf", "page": 2}), 0.7),
    ]
    result = _format_context(docs)

    assert "Chunk 1" in result
    assert "Chunk 2" in result
    assert "---" in result  # separator


# ── _format_history ─────────────────────────────────────────────────────────

def test_format_history_empty():
    result = _format_history([])
    assert result == "No previous conversation."


def test_format_history_with_messages():
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = _format_history(history)

    assert "User: Hello" in result
    assert "Assistant: Hi there" in result


def test_format_history_respects_memory_window():
    history = [
        {"role": "user", "content": f"Q{i}"}
        for i in range(20)
    ]
    with patch.object(config, "MAX_HISTORY_TOKENS", None):
        result = _format_history(history)

    # With MEMORY_WINDOW=5, should keep last 10 messages (5 turns * 2)
    lines = result.strip().split("\n")
    assert len(lines) == config.MEMORY_WINDOW * 2


# ── _count_tokens ───────────────────────────────────────────────────────────

def test_count_tokens_returns_positive_int():
    count = _count_tokens("Hello, world!")
    assert isinstance(count, int)
    assert count > 0


def test_count_tokens_empty_string():
    count = _count_tokens("")
    assert count == 0


# ── _truncate_history_by_tokens ─────────────────────────────────────────────

def test_truncate_history_within_budget():
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    result = _truncate_history_by_tokens(messages.copy(), max_tokens=1000)
    assert len(result) == 2


def test_truncate_history_drops_oldest():
    messages = [
        {"role": "user", "content": "word " * 500},       # ~500 tokens
        {"role": "assistant", "content": "word " * 500},   # ~500 tokens
        {"role": "user", "content": "short"},              # ~1 token
    ]
    result = _truncate_history_by_tokens(messages.copy(), max_tokens=50)
    # Should drop the first two long messages, keep only "short"
    assert len(result) == 1
    assert result[0]["content"] == "short"


def test_truncate_history_empty_if_all_exceed():
    messages = [
        {"role": "user", "content": "word " * 1000},
    ]
    result = _truncate_history_by_tokens(messages.copy(), max_tokens=5)
    assert result == []


def test_format_history_with_token_truncation():
    history = [
        {"role": "user", "content": "question " * 200},
        {"role": "assistant", "content": "answer " * 200},
        {"role": "user", "content": "short follow-up"},
    ]
    with patch.object(config, "MAX_HISTORY_TOKENS", 50):
        result = _format_history(history)

    assert "short follow-up" in result


# ── ask (integration with mocks) ───────────────────────────────────────────

@patch("src.pipeline.retrieve_with_scores")
def test_ask_low_confidence_returns_fallback(mock_retrieve):
    doc = Document(page_content="irrelevant", metadata={})
    mock_retrieve.return_value = [(doc, 0.1)]  # Below RELEVANCE_THRESHOLD

    mock_llm = MagicMock()
    mock_store = MagicMock()

    stream, sources = ask("What is X?", mock_store, mock_llm, [])

    # Should get fallback message, no sources
    tokens = list(stream)
    assert any("don't have enough information" in t for t in tokens)
    assert sources == []
    # LLM should NOT be called
    mock_llm.stream.assert_not_called()


@patch("src.pipeline.retrieve_with_scores")
def test_ask_high_confidence_streams_llm(mock_retrieve):
    doc = Document(
        page_content="Python is great.",
        metadata={"source": "doc.pdf", "page": 1, "chunk_index": 0},
    )
    mock_retrieve.return_value = [(doc, 0.85)]

    mock_chunk = MagicMock()
    mock_chunk.content = "Answer token"
    mock_llm = MagicMock()
    mock_llm.stream.return_value = [mock_chunk]

    mock_store = MagicMock()

    stream, sources = ask("What is Python?", mock_store, mock_llm, [])

    tokens = list(stream)
    assert len(tokens) > 0
    assert len(sources) == 1
    assert sources[0]["source"] == "doc.pdf"
    assert sources[0]["score"] == 0.85


@patch("src.pipeline.retrieve_with_scores")
def test_ask_no_results_returns_fallback(mock_retrieve):
    mock_retrieve.return_value = []

    mock_llm = MagicMock()
    mock_store = MagicMock()

    stream, sources = ask("Random question?", mock_store, mock_llm, [])

    tokens = list(stream)
    assert any("don't have enough information" in t for t in tokens)
    assert sources == []
