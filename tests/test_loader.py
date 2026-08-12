"""Tests for src/loader.py"""

import hashlib
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.loader import (
    get_document_stats,
    load_file_from_bytes,
    load_multiple_pdfs,
    load_pdf_from_bytes,
    load_pdf_from_path,
    SUPPORTED_EXTENSIONS,
)


# ── load_pdf_from_path ──────────────────────────────────────────────────────

def test_load_pdf_from_path_file_not_found():
    with pytest.raises(FileNotFoundError, match="PDF not found"):
        load_pdf_from_path("/nonexistent/file.pdf")


def test_load_pdf_from_path_not_a_pdf(tmp_path):
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("hello")
    with pytest.raises(ValueError, match="Expected a .pdf file"):
        load_pdf_from_path(str(txt_file))


def test_load_pdf_from_path_success(tmp_path):
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake content")

    mock_doc = Document(page_content="Page 1 text", metadata={"page": 0})

    with patch("langchain_community.document_loaders.PyPDFLoader") as mock_cls:
        mock_cls.return_value.load.return_value = [mock_doc]
        docs = load_pdf_from_path(str(pdf_file))

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "sample.pdf"
    assert "file_hash" in docs[0].metadata


# ── load_file_from_bytes ────────────────────────────────────────────────────

def test_load_file_from_bytes_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_file_from_bytes(b"data", "file.xyz")


def test_load_file_from_bytes_pdf():
    raw = b"%PDF-1.4 fake content"
    expected_hash = hashlib.md5(raw).hexdigest()
    mock_doc = Document(page_content="Byte page", metadata={"page": 0})

    with patch("langchain_community.document_loaders.PyPDFLoader") as mock_cls:
        mock_cls.return_value.load.return_value = [mock_doc]
        docs = load_file_from_bytes(raw, "upload.pdf")

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "upload.pdf"
    assert docs[0].metadata["file_hash"] == expected_hash


def test_load_file_from_bytes_txt():
    raw = b"Hello world text content"
    mock_doc = Document(page_content="Hello world text content", metadata={})

    with patch("langchain_community.document_loaders.TextLoader") as mock_cls:
        mock_cls.return_value.load.return_value = [mock_doc]
        docs = load_file_from_bytes(raw, "notes.txt")

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "notes.txt"
    assert "file_hash" in docs[0].metadata


def test_load_file_from_bytes_csv():
    raw = b"name,age\nAlice,30\nBob,25"
    mock_doc = Document(page_content="name: Alice\nage: 30", metadata={})

    with patch("langchain_community.document_loaders.CSVLoader") as mock_cls:
        mock_cls.return_value.load.return_value = [mock_doc]
        docs = load_file_from_bytes(raw, "data.csv")

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "data.csv"


def test_load_file_from_bytes_docx():
    raw = b"fake docx bytes"
    mock_doc = Document(page_content="Document text", metadata={})

    with patch("langchain_community.document_loaders.Docx2txtLoader") as mock_cls:
        mock_cls.return_value.load.return_value = [mock_doc]
        docs = load_file_from_bytes(raw, "report.docx")

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "report.docx"


# ── load_pdf_from_bytes (backwards compat wrapper) ──────────────────────────

def test_load_pdf_from_bytes_delegates():
    raw = b"%PDF-1.4 fake"
    mock_doc = Document(page_content="page", metadata={})

    with patch("langchain_community.document_loaders.PyPDFLoader") as mock_cls:
        mock_cls.return_value.load.return_value = [mock_doc]
        docs = load_pdf_from_bytes(raw, "test.pdf")

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "test.pdf"


# ── SUPPORTED_EXTENSIONS ────────────────────────────────────────────────────

def test_supported_extensions():
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".docx" in SUPPORTED_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert ".csv" in SUPPORTED_EXTENSIONS


# ── load_multiple_pdfs ──────────────────────────────────────────────────────

@patch("src.loader.load_file_from_bytes")
def test_load_multiple_pdfs_combined_hash(mock_load):
    mock_load.return_value = [
        Document(page_content="text", metadata={"source": "a.pdf"})
    ]

    file_a = MagicMock()
    file_a.read.return_value = b"content_a"
    file_a.name = "a.pdf"

    file_b = MagicMock()
    file_b.read.return_value = b"content_b"
    file_b.name = "b.pdf"

    docs, combined_hash = load_multiple_pdfs([file_a, file_b])

    assert len(docs) == 2
    assert isinstance(combined_hash, str)
    assert len(combined_hash) == 32  # MD5 hex digest length


@patch("src.loader.load_file_from_bytes")
def test_load_multiple_pdfs_order_independent_hash(mock_load):
    """Uploading {A, B} should produce the same hash as {B, A}."""
    mock_load.return_value = [
        Document(page_content="text", metadata={})
    ]

    file_a = MagicMock()
    file_a.read.return_value = b"content_a"
    file_a.name = "a.pdf"

    file_b = MagicMock()
    file_b.read.return_value = b"content_b"
    file_b.name = "b.pdf"

    _, hash_ab = load_multiple_pdfs([file_a, file_b])

    # Reset mocks for second call
    file_a.read.return_value = b"content_a"
    file_b.read.return_value = b"content_b"

    _, hash_ba = load_multiple_pdfs([file_b, file_a])

    assert hash_ab == hash_ba


# ── get_document_stats ──────────────────────────────────────────────────────

def test_get_document_stats_empty():
    stats = get_document_stats([])
    assert stats["pages"] == 0
    assert stats["total_chars"] == 0
    assert stats["num_files"] == 0


def test_get_document_stats_with_documents():
    docs = [
        Document(page_content="Hello world", metadata={"source": "a.pdf"}),
        Document(page_content="Test content here", metadata={"source": "a.pdf"}),
        Document(page_content="Another file", metadata={"source": "b.pdf"}),
    ]
    stats = get_document_stats(docs)

    assert stats["pages"] == 3
    assert stats["total_chars"] == len("Hello world") + len("Test content here") + len("Another file")
    assert stats["num_files"] == 2
    assert stats["avg_chars_per_page"] == stats["total_chars"] // 3
