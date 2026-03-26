"""
src/ — Core RAG Engine Package
================================
This package contains all the modular components for the RAG pipeline:

  loader.py      → PDF ingestion into LangChain Documents
  chunker.py     → Recursive text splitting with metadata
  embeddings.py  → Embedding model factory (OpenAI / HuggingFace)
  retriever.py   → Vector store build, persist, load, and search
  llm.py         → LLM factory (OpenAI / Gemini)
  pipeline.py    → End-to-end RAG chain with memory and scoring

All modules read their configuration from config.py (project root).
"""
