"""
config.py — Single Source of Truth
===================================
Every tunable parameter for the entire RAG chatbot lives here.
No other module should hardcode model names, chunk sizes, or thresholds.

Usage:
    from config import LLM_PROVIDER, CHUNK_SIZE, TOP_K   # import what you need

Environment variables are loaded from a .env file via python-dotenv.
"""

import os
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Load .env file (must be called before any os.getenv for API keys)
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LLM CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Which LLM provider to use: "openai" | "gemini"
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")

# OpenAI LLM settings
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL: str = "gpt-4o-mini"

# Google Gemini LLM settings
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL: str = "gemini-1.5-flash"

# Shared LLM tuning
LLM_TEMPERATURE: float = 0.2       # Low = factual; high = creative
LLM_MAX_TOKENS: int = 1024         # Max output tokens per response
STREAMING: bool = True              # Stream tokens to UI in real time


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EMBEDDING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Which embedding provider to use: "openai" | "huggingface"
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai")

# OpenAI embeddings (requires OPENAI_API_KEY)
OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

# HuggingFace embeddings (runs locally, no API key needed — great fallback)
HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. VECTOR STORE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Which vector store: "faiss" | "chroma"
VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "faiss")

# Directory where persisted indexes are saved (auto-created)
VECTOR_STORE_DIR: str = os.path.join(os.path.dirname(__file__), "vector_store")

# Number of top-k chunks to retrieve per query
TOP_K: int = 5


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TEXT CHUNKING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum characters per chunk
CHUNK_SIZE: int = 1000

# Overlap between consecutive chunks (preserves context at boundaries)
CHUNK_OVERLAP: int = 200

# Separator priority for RecursiveCharacterTextSplitter
# Tries paragraph breaks first, then newlines, then sentences, etc.
SEPARATORS: list[str] = ["\n\n", "\n", ". ", " ", ""]


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CONVERSATION MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

# Number of recent conversation turns to keep in memory buffer
# Higher = better context for follow-ups, but uses more tokens
MEMORY_WINDOW: int = 5


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RELEVANCE / CONFIDENCE SCORING
# ═══════════════════════════════════════════════════════════════════════════════

# If ALL retrieved chunks score below this, the bot says "I don't know"
RELEVANCE_THRESHOLD: float = 0.30

# Thresholds for UI badges:  🟢 High | ⚡ Medium | 🔴 Low
HIGH_SCORE_THRESHOLD: float = 0.75
MID_SCORE_THRESHOLD: float = 0.50


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ADVANCED FEATURE FLAGS
# ═══════════════════════════════════════════════════════════════════════════════

# Hybrid Search: combine dense vector retrieval + BM25 keyword retrieval
# Requires: rank-bm25 (already in requirements.txt)
ENABLE_HYBRID_SEARCH: bool = False

# Ensemble weights when hybrid search is ON: [dense_weight, bm25_weight]
HYBRID_SEARCH_WEIGHTS: list[float] = [0.5, 0.5]

# Reranking: after initial retrieval, re-score with a cross-encoder model
# Requires: sentence-transformers (already in requirements.txt)
ENABLE_RERANKING: bool = False
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_N: int = 3             # Keep top N after reranking


# ═══════════════════════════════════════════════════════════════════════════════
# 8. UI CONFIGURATION (Streamlit)
# ═══════════════════════════════════════════════════════════════════════════════

APP_TITLE: str = "🧠 RAG AI Chatbot"
APP_ICON: str = "🤖"
APP_LAYOUT: str = "wide"            # "centered" | "wide"


# ═══════════════════════════════════════════════════════════════════════════════
# 9. RAG PROMPT TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════

RAG_PROMPT_TEMPLATE: str = """You are a helpful, accurate assistant. Answer the user's question strictly based on the provided context.

RULES:
- Only use information from the context below.
- If the context does not contain enough information, say "I don't have enough information in the provided documents to answer that."
- Cite the source filename and page number when referencing specific information.
- Be concise but thorough.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

USER QUESTION:
{question}

ANSWER:"""

