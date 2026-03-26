"""
loader.py — PDF Ingestion Module
==================================
Responsibility:
  Convert PDF files into LangChain Document objects, whether from a file path
  on disk or from raw bytes (Streamlit uploads). Each Document's metadata is
  enriched with: source filename, page number, and an MD5 file hash (used for
  vector store caching so we don't re-index the same file twice).

Key design decisions:
  - Bytes-based loading writes to a temp file, loads via PyPDFLoader, then
    cleans up. This avoids keeping large PDFs in memory.
  - MD5 hash is computed on the raw bytes, NOT on extracted text, so it's
    fast and deterministic regardless of how the PDF parser tokenizes content.
  - Multi-PDF support merges all documents into a single list while preserving
    per-file metadata, enabling the retriever to show which file a chunk came from.
"""

import hashlib
import os
import tempfile
from typing import Dict, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_pdf_from_path(file_path: str) -> List[Document]:
    """
    Load a PDF from a file path on disk.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of LangChain Document objects (one per page).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if not file_path.lower().endswith(".pdf"):
        raise ValueError(f"Expected a .pdf file, got: {file_path}")

    # Compute MD5 hash from the raw file bytes
    file_hash = _compute_file_hash(file_path)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Enrich every page's metadata with source filename and hash
    filename = os.path.basename(file_path)
    for doc in documents:
        doc.metadata["source"] = filename
        doc.metadata["file_hash"] = file_hash

    return documents


def load_pdf_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Load a PDF from raw bytes — used for Streamlit UploadedFile objects.

    Strategy: write bytes to a temp file → load with PyPDFLoader → cleanup.
    This avoids holding the entire parsed result AND the raw bytes in memory
    at the same time.

    Args:
        file_bytes: Raw bytes of the uploaded PDF.
        filename:   Original filename (for metadata tracking).

    Returns:
        List of LangChain Document objects (one per page).
    """
    # Compute MD5 hash directly from the bytes
    file_hash = hashlib.md5(file_bytes).hexdigest()

    # Write to a secure temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Tag every page with the original filename and hash
        for doc in documents:
            doc.metadata["source"] = filename
            doc.metadata["file_hash"] = file_hash

        return documents

    finally:
        # Always clean up the temp file, even if loading fails
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_multiple_pdfs(uploaded_files: list) -> tuple[List[Document], str]:
    """
    Load multiple PDFs from Streamlit UploadedFile objects and merge them
    into a single document list. Also computes a combined hash for caching.

    Why a combined hash?
      If the user uploads the same set of files again, we can skip re-indexing
      by checking this single hash against the persisted vector store.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.

    Returns:
        Tuple of (merged_documents, combined_md5_hash).
    """
    all_documents: List[Document] = []
    hash_parts: list[str] = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name

        # Load pages from this PDF
        docs = load_pdf_from_bytes(file_bytes, filename)
        all_documents.extend(docs)

        # Collect individual file hashes for the combined hash
        file_hash = hashlib.md5(file_bytes).hexdigest()
        hash_parts.append(file_hash)

    # Combined hash: sort individual hashes so order doesn't matter,
    # then hash the concatenation. This means uploading {A.pdf, B.pdf}
    # produces the same combined hash as {B.pdf, A.pdf}.
    hash_parts.sort()
    combined_hash = hashlib.md5("".join(hash_parts).encode()).hexdigest()

    return all_documents, combined_hash


def get_document_stats(documents: List[Document]) -> Dict[str, int]:
    """
    Compute basic statistics about loaded documents for sidebar display.

    Args:
        documents: List of loaded Document objects.

    Returns:
        Dict with keys: pages, total_chars, avg_chars_per_page, num_files.
    """
    if not documents:
        return {
            "pages": 0,
            "total_chars": 0,
            "avg_chars_per_page": 0,
            "num_files": 0,
        }

    page_count = len(documents)
    total_chars = sum(len(doc.page_content) for doc in documents)
    avg_chars = total_chars // page_count if page_count > 0 else 0

    # Count unique source filenames
    unique_files = set(doc.metadata.get("source", "unknown") for doc in documents)

    return {
        "pages": page_count,
        "total_chars": total_chars,
        "avg_chars_per_page": avg_chars,
        "num_files": len(unique_files),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_file_hash(file_path: str) -> str:
    """
    Compute the MD5 hash of a file by reading it in 8KB chunks.

    Why MD5?
      It's fast, deterministic, and we only use it as a cache key —
      not for cryptographic security.

    Args:
        file_path: Path to the file.

    Returns:
        Hex digest string (32 characters).
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()
