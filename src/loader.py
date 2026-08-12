"""
loader.py — Document Ingestion Module
========================================
Responsibility:
  Convert uploaded files (PDF, DOCX, TXT, CSV) into LangChain Document objects.
  Each Document's metadata is enriched with: source filename, page number, and
  an MD5 file hash (used for vector store caching).

Supported file types:
  .pdf  — via PyPDFLoader
  .docx — via Docx2txtLoader
  .txt  — via TextLoader
  .csv  — via CSVLoader
"""

import hashlib
import logging
import os
import tempfile
from typing import Dict, List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}


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

    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Enrich every page's metadata with source filename and hash
    filename = os.path.basename(file_path)
    for doc in documents:
        doc.metadata["source"] = filename
        doc.metadata["file_hash"] = file_hash

    logger.info("Loaded %d pages from '%s' (hash: %s)", len(documents), filename, file_hash)
    return documents


def load_pdf_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Load a PDF from raw bytes — used for Streamlit UploadedFile objects.

    Args:
        file_bytes: Raw bytes of the uploaded PDF.
        filename:   Original filename (for metadata tracking).

    Returns:
        List of LangChain Document objects (one per page).
    """
    return load_file_from_bytes(file_bytes, filename)


def load_file_from_bytes(file_bytes: bytes, filename: str, file_hash: str | None = None) -> List[Document]:
    """
    Load a document from raw bytes, dispatching to the correct loader
    based on file extension.

    Supported types: .pdf, .docx, .txt, .csv

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename (determines which loader to use).
        file_hash:  Pre-computed MD5 hash. Computed from file_bytes if not provided.

    Returns:
        List of LangChain Document objects.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if file_hash is None:
        file_hash = hashlib.md5(file_bytes).hexdigest()

    # Write to a temp file so LangChain loaders can read from disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = _get_loader_for_extension(ext, tmp_path)
        documents = loader.load()

        for doc in documents:
            doc.metadata["source"] = filename
            doc.metadata["file_hash"] = file_hash

        logger.info("Loaded %d document(s) from '%s' [%s] (hash: %s)", len(documents), filename, ext, file_hash)
        return documents

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_multiple_pdfs(uploaded_files: list) -> tuple[List[Document], str]:
    """
    Load multiple files from Streamlit UploadedFile objects and merge them
    into a single document list. Also computes a combined hash for caching.

    Despite the name (kept for backwards compatibility), this now supports
    all file types: PDF, DOCX, TXT, CSV.

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

        docs = load_file_from_bytes(file_bytes, filename)
        all_documents.extend(docs)

        file_hash = hashlib.md5(file_bytes).hexdigest()
        hash_parts.append(file_hash)

    # Sort so upload order doesn't affect the hash
    hash_parts.sort()
    combined_hash = hashlib.md5("".join(hash_parts).encode()).hexdigest()

    logger.info(
        "Loaded %d file(s), %d total documents (combined hash: %s)",
        len(uploaded_files), len(all_documents), combined_hash,
    )
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

def _get_loader_for_extension(ext: str, file_path: str):
    """
    Return the appropriate LangChain document loader for a file extension.

    Args:
        ext:       Lowercase file extension (e.g. ".pdf").
        file_path: Path to the temp file on disk.

    Returns:
        A LangChain BaseLoader instance.
    """
    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(file_path)
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(file_path)
    elif ext == ".txt":
        from langchain_community.document_loaders import TextLoader
        return TextLoader(file_path, encoding="utf-8")
    elif ext == ".csv":
        from langchain_community.document_loaders import CSVLoader
        return CSVLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"No loader available for extension: '{ext}'")


def _compute_file_hash(file_path: str) -> str:
    """
    Compute the MD5 hash of a file by reading it in 8KB chunks.

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
