"""
app.py — Streamlit UI for the RAG AI Chatbot
==============================================
Responsibility:
  Provide a polished, production-quality chat interface that ties together
  every module in the src/ package.

Layout:
  SIDEBAR                          MAIN AREA
  ┌──────────────┐                ┌──────────────────────────────┐
  │ 📁 PDF Upload │                │ 💬 Chat messages (scrollable) │
  │ ⚙️  Providers  │                │                              │
  │ 📊 Stats      │                │ 🤖 Streamed answer           │
  │ 💾 Export     │                │ ▼ Source chunks (expandable) │
  │ 🗑️  Clear      │                │ [Ask a question...]          │
  └──────────────┘                └──────────────────────────────┘

Session state keys:
  vector_store   — persisted FAISS/Chroma VectorStore across questions
  chat_history   — List[{role, content, sources}]
  file_hash      — MD5 hash of currently indexed file(s)
  doc_stats      — page/char stats from loader
  chunk_stats    — chunk count/size stats from chunker
  chunks         — raw chunk list (needed for hybrid search BM25)
"""

import json
import datetime
from typing import List

import streamlit as st

import config
from src.loader import load_multiple_pdfs, get_document_stats
from src.chunker import split_documents, get_chunk_stats
from src.embeddings import get_embedding_model
from src.retriever import build_vector_store, store_exists, load_vector_store
from src.llm import get_llm
from src.pipeline import ask


# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.APP_LAYOUT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS for a polished look
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Chat message styling */
    .stChatMessage { border-radius: 12px; }

    /* Source chunk badges */
    .score-high { color: #22c55e; font-weight: 700; }
    .score-med  { color: #f59e0b; font-weight: 700; }
    .score-low  { color: #ef4444; font-weight: 700; }

    /* Sidebar stats cards */
    .stat-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        color: #e2e8f0;
    }
    .stat-card h4 { margin: 0 0 4px 0; font-size: 0.85rem; color: #94a3b8; }
    .stat-card p  { margin: 0; font-size: 1.3rem; font-weight: 700; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions (defined BEFORE use — Streamlit runs top-to-bottom)
# ─────────────────────────────────────────────────────────────────────────────

def _init_session_state() -> None:
    """Set default values for all session state keys on first load."""
    defaults = {
        "vector_store": None,
        "chat_history": [],
        "file_hash": None,
        "doc_stats": None,
        "chunk_stats": None,
        "chunks": None,
        "indexed": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _render_stat_card(label: str, value) -> None:
    """Render a styled stat card in the sidebar."""
    st.markdown(
        f'<div class="stat-card"><h4>{label}</h4><p>{value}</p></div>',
        unsafe_allow_html=True,
    )


def _render_sources(sources: List[dict]) -> None:
    """
    Render source chunks in an expandable accordion with relevance badges.
    Each source shows: content preview, filename, page number, relevance score.
    """
    with st.expander(f"📚 Sources ({len(sources)} chunks retrieved)", expanded=False):
        for i, src in enumerate(sources, 1):
            score = src.get("score", 0)
            label = src.get("label", "Low")

            # Color-coded badge
            if "High" in label:
                badge_class = "score-high"
            elif "Med" in label:
                badge_class = "score-med"
            else:
                badge_class = "score-low"

            st.markdown(
                f"**Chunk {i}** &nbsp; "
                f"<span class='{badge_class}'>{label}</span> &nbsp; "
                f"Score: `{score}` &nbsp; "
                f"📄 `{src.get('source', '?')}` &nbsp; "
                f"Page: `{src.get('page', '?')}`",
                unsafe_allow_html=True,
            )
            st.caption(
                src.get("content", "")[:500]
                + ("..." if len(src.get("content", "")) > 500 else "")
            )
            if i < len(sources):
                st.divider()


def _index_documents(uploaded_files: list) -> None:
    """
    Load, chunk, embed, and store uploaded PDFs. Updates session state.

    If the same files were already indexed (same MD5 hash), loads the cached
    vector store from disk instead of re-embedding.
    """
    with st.spinner("📄 Loading PDFs..."):
        documents, combined_hash = load_multiple_pdfs(uploaded_files)
        doc_stats = get_document_stats(documents)
        st.session_state.doc_stats = doc_stats

    # Check if we already have this exact set of files indexed in memory
    if st.session_state.file_hash == combined_hash and st.session_state.vector_store:
        st.success("✅ These files are already indexed! Ready to chat.")
        return

    with st.spinner("✂️ Splitting into chunks..."):
        chunks = split_documents(documents)
        chunk_stats = get_chunk_stats(chunks)
        st.session_state.chunk_stats = chunk_stats
        st.session_state.chunks = chunks

    with st.spinner("🧠 Building embeddings & vector store..."):
        embeddings = get_embedding_model()

        # Try loading cached vector store from disk
        if store_exists(combined_hash):
            vector_store = load_vector_store(combined_hash, embeddings)
            if vector_store:
                st.session_state.vector_store = vector_store
                st.session_state.file_hash = combined_hash
                st.session_state.indexed = True
                st.success(
                    f"✅ Loaded cached index! {chunk_stats['total_chunks']} chunks ready."
                )
                return

        # Build fresh vector store
        vector_store = build_vector_store(
            chunks, embeddings, file_hash=combined_hash
        )
        st.session_state.vector_store = vector_store
        st.session_state.file_hash = combined_hash
        st.session_state.indexed = True

    st.success(
        f"✅ Indexed {doc_stats['num_files']} file(s) → "
        f"{chunk_stats['total_chunks']} chunks. Ready to chat!"
    )


def _build_export_data() -> list:
    """Build JSON-serializable export data from chat history."""
    export = []
    for msg in st.session_state.chat_history:
        entry = {"role": msg["role"], "content": msg["content"]}
        if msg.get("sources"):
            entry["sources"] = [
                {
                    "source": s.get("source"),
                    "page": s.get("page"),
                    "score": s.get("score"),
                    "label": s.get("label"),
                }
                for s in msg["sources"]
            ]
        export.append(entry)
    return export


def _build_export_txt() -> str:
    """Build a plain-text export of the chat history."""
    lines = [
        f"{'=' * 60}",
        f"  {config.APP_TITLE} — Chat Export",
        f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"{'=' * 60}\n",
    ]
    for msg in st.session_state.chat_history:
        role = "YOU" if msg["role"] == "user" else "BOT"
        lines.append(f"[{role}]")
        lines.append(msg["content"])
        if msg.get("sources"):
            lines.append("  Sources:")
            for s in msg["sources"]:
                lines.append(
                    f"    - {s.get('source', '?')} | Page {s.get('page', '?')} | "
                    f"Score: {s.get('score', '?')} ({s.get('label', '?')})"
                )
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Initialize session state
# ─────────────────────────────────────────────────────────────────────────────
_init_session_state()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{config.APP_ICON} {config.APP_TITLE}")
    st.caption("Upload PDFs → Ask questions → Get cited answers")

    st.divider()

    # ── Provider selectors ──────────────────────────────────────────────────
    st.subheader("⚙️ Model Settings")

    llm_provider = st.selectbox(
        "LLM Provider",
        options=["openai", "gemini"],
        index=0 if config.LLM_PROVIDER == "openai" else 1,
        help="Switch between OpenAI GPT-4o-mini and Google Gemini",
    )
    config.LLM_PROVIDER = llm_provider

    embed_provider = st.selectbox(
        "Embedding Provider",
        options=["openai", "huggingface"],
        index=0 if config.EMBEDDING_PROVIDER == "openai" else 1,
        help="HuggingFace runs locally (free, no API key needed)",
    )
    config.EMBEDDING_PROVIDER = embed_provider

    st.divider()

    # ── PDF uploader ────────────────────────────────────────────────────────
    st.subheader("📁 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload your documents to start asking questions",
    )

    # ── Index button ────────────────────────────────────────────────────────
    if uploaded_files:
        if st.button("🚀 Index Documents", type="primary", use_container_width=True):
            _index_documents(uploaded_files)

    # ── Stats display ───────────────────────────────────────────────────────
    if st.session_state.doc_stats:
        st.divider()
        st.subheader("📊 Document Stats")
        stats = st.session_state.doc_stats
        _render_stat_card("📄 Files", stats["num_files"])
        _render_stat_card("📃 Pages", stats["pages"])
        _render_stat_card("🔤 Characters", f"{stats['total_chars']:,}")

    if st.session_state.chunk_stats:
        c_stats = st.session_state.chunk_stats
        _render_stat_card("🧩 Chunks", c_stats["total_chunks"])
        _render_stat_card("📏 Avg Chunk Size", f"{c_stats['avg_chunk_size']} chars")

    # ── Export & Clear ──────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.divider()
        st.subheader("💾 Export & Clear")

        col1, col2 = st.columns(2)
        with col1:
            export_data = _build_export_data()
            st.download_button(
                "📥 JSON",
                data=json.dumps(export_data, indent=2, ensure_ascii=False),
                file_name=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col2:
            export_txt = _build_export_txt()
            st.download_button(
                "📄 TXT",
                data=export_txt,
                file_name=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA — Chat Interface
# ─────────────────────────────────────────────────────────────────────────────
st.header("💬 Chat with your Documents")

if not st.session_state.indexed:
    st.info(
        "👈 Upload PDF files in the sidebar and click "
        "**Index Documents** to get started."
    )

# ── Render existing chat history ────────────────────────────────────────────
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            _render_sources(message["sources"])

# ── Chat input ──────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask a question about your documents...",
    disabled=not st.session_state.indexed,
):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        try:
            llm = get_llm(streaming=config.STREAMING)

            answer_stream, source_chunks = ask(
                question=prompt,
                vector_store=st.session_state.vector_store,
                llm=llm,
                chat_history=st.session_state.chat_history,
            )

            # Stream the response token by token
            full_response = st.write_stream(answer_stream)

            # Render source chunks below the answer
            if source_chunks:
                _render_sources(source_chunks)

            # Save to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_response,
                "sources": source_chunks,
            })

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg,
                "sources": [],
            })
