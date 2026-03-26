"""
chunker.py — Document Splitting Module
========================================
Responsibility:
  Take a list of LangChain Document objects (one per PDF page) and split them
  into smaller, overlapping chunks suitable for embedding and retrieval.

Why overlap?
  A question might span a paragraph boundary. Overlap ensures that no information
  is lost at chunk edges. For example, with chunk_size=1000 and overlap=200, the
  last 200 characters of chunk N are repeated at the start of chunk N+1.

Why RecursiveCharacterTextSplitter?
  It tries the most semantically meaningful split first (paragraph break "\n\n"),
  then falls back to newline, sentence, word, and finally character-level splits.
  This preserves paragraph structure whenever possible.

Metadata enrichment:
  Every chunk inherits the parent Document's metadata (source, page, file_hash)
  AND gets additional fields: chunk_index (global position) and chunk_char_count.
  This allows the UI to display "Chunk 14 from report.pdf, page 3".
"""

from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Split documents into overlapping chunks with enriched metadata.

    Args:
        documents:     List of Document objects (typically from loader.py).
        chunk_size:    Max characters per chunk. Defaults to config.CHUNK_SIZE.
        chunk_overlap: Overlap between chunks. Defaults to config.CHUNK_OVERLAP.

    Returns:
        List of chunked Document objects with original + new metadata.

    Raises:
        ValueError: If documents list is empty.
    """
    if not documents:
        raise ValueError("Cannot split an empty document list.")

    # Fall back to config values if not explicitly provided
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    # Initialize the splitter with our separator priority
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=config.SEPARATORS,
        length_function=len,            # character count, not token count
        is_separator_regex=False,
    )

    # Split all documents at once — LangChain preserves parent metadata
    raw_chunks = splitter.split_documents(documents)

    # Enrich each chunk with positional and size metadata
    enriched_chunks = _enrich_chunk_metadata(raw_chunks)

    return enriched_chunks


def get_chunk_stats(chunks: List[Document]) -> Dict[str, int]:
    """
    Compute statistics about the chunks for sidebar display.

    Args:
        chunks: List of chunked Document objects.

    Returns:
        Dict with keys: total_chunks, avg_chunk_size, min_chunk_size,
        max_chunk_size, total_chars.
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
            "total_chars": 0,
        }

    sizes = [len(chunk.page_content) for chunk in chunks]

    return {
        "total_chunks": len(sizes),
        "avg_chunk_size": sum(sizes) // len(sizes),
        "min_chunk_size": min(sizes),
        "max_chunk_size": max(sizes),
        "total_chars": sum(sizes),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_chunk_metadata(chunks: List[Document]) -> List[Document]:
    """
    Add chunk_index and chunk_char_count to every chunk's metadata.

    Why?
      - chunk_index: Allows the UI to show "Chunk 14 of 200" and helps with
        debugging retrieval results.
      - chunk_char_count: Quick filter — if a chunk is suspiciously small,
        it might be a header or footer artifact.

    Args:
        chunks: Raw chunks from the splitter (already have inherited metadata).

    Returns:
        The same list, mutated in place for efficiency, with new metadata fields.
    """
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_char_count"] = len(chunk.page_content)

    return chunks
