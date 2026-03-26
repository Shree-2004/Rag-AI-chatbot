"""
src/pipeline.py
─────────────────────────────────────────────────────────────────────────────
Responsibility: Orchestrate the full RAG chain:

  1. Receive user question + chat history.
  2. Retrieve top-k relevant chunks (with cosine similarity scores).
  3. Check minimum confidence — return "I don't know" if all scores are low.
  4. Format context from retrieved chunks.
  5. Fill the prompt template ({context}, {chat_history}, {question}).
  6. Stream the LLM response token by token.
  7. Return the answer text + annotated source chunks for the UI.

Why a separate pipeline module?
  app.py stays thin (UI only). All RAG logic lives here and can be tested
  independently from Streamlit.
─────────────────────────────────────────────────────────────────────────────
"""

from typing import Generator, List, Tuple

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore

import config
from src.retriever import retrieve_with_scores, score_label


# ─────────────────────────────────────────────────────────────────────────────
# Prompt template (defined once, reused across calls)
# ─────────────────────────────────────────────────────────────────────────────
_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=config.RAG_PROMPT_TEMPLATE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def ask(
    question: str,
    vector_store: VectorStore,
    llm: BaseChatModel,
    chat_history: List[dict],
) -> Tuple[Generator[str, None, None], List[dict]]:
    """
    Run the full RAG pipeline for a single user question.

    Args:
        question:     The user's current question string.
        vector_store: Populated VectorStore from retriever.build_vector_store().
        llm:          LLM instance from llm.get_llm(streaming=True).
        chat_history: List of {"role": "user"|"assistant", "content": str}
                      dicts representing previous turns.

    Returns:
        A tuple of:
          - answer_stream: A generator yielding string tokens for st.write_stream.
          - source_chunks: List of dicts with keys:
              "content", "source", "page", "chunk_index", "score", "label"
    """
    # ── Step 1: Retrieve relevant chunks with cosine similarity scores ──────
    scored_results: List[Tuple[Document, float]] = retrieve_with_scores(
        vector_store, question, top_k=config.TOP_K
    )

    # ── Step 2: Confidence gate ──────────────────────────────────────────────
    # If the best match is below minimum confidence, skip the LLM and return
    # a canned "I don't know" response to avoid hallucination.
    if scored_results:
        best_score = max(score for _, score in scored_results)
    else:
        best_score = 0.0

    if best_score < config.RELEVANCE_THRESHOLD:
        def _low_confidence_stream():
            yield "I don't have enough information in the provided documents to answer that."
        return _low_confidence_stream(), []

    # ── Step 3: Format context from retrieved chunks ─────────────────────────
    context_text = _format_context(scored_results)

    # ── Step 4: Format chat history for the prompt ───────────────────────────
    history_text = _format_history(chat_history)

    # ── Step 5: Build the filled prompt ──────────────────────────────────────
    filled_prompt = _PROMPT.format(
        context=context_text,
        chat_history=history_text,
        question=question,
    )

    # ── Step 6: Build annotated source chunk list for the UI ─────────────────
    source_chunks = [
        {
            "content":     doc.page_content,
            "source":      doc.metadata.get("source", "unknown"),
            "page":        doc.metadata.get("page", "?"),
            "chunk_index": doc.metadata.get("chunk_index", "?"),
            "score":       round(score, 3),
            "label":       score_label(score),
        }
        for doc, score in scored_results
    ]

    # ── Step 7: Stream LLM response ──────────────────────────────────────────
    answer_stream = _stream_response(llm, filled_prompt)

    return answer_stream, source_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stream_response(llm: BaseChatModel, prompt: str) -> Generator[str, None, None]:
    """
    Yield LLM response tokens one by one using LangChain's .stream() API.

    Streamlit's st.write_stream() consumes this generator and renders each
    token as it arrives — no waiting for the full response.

    Args:
        llm:    LangChain chat model with streaming=True.
        prompt: The fully formatted prompt string.

    Yields:
        Individual token strings.
    """
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        if chunk.content:
            yield chunk.content


def _format_context(scored_results: List[Tuple[Document, float]]) -> str:
    """
    Format retrieved chunks into a single context string for the prompt.

    Each chunk is labelled with its source filename and page number so the
    LLM can reference them and the user can verify answers.

    Args:
        scored_results: List of (Document, score) pairs from retrieve_with_scores.

    Returns:
        Multi-line context string.
    """
    parts = []
    for i, (doc, score) in enumerate(scored_results, start=1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        parts.append(
            f"[Chunk {i} | Source: {source} | Page: {page} | Relevance: {score:.2f}]\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


def _format_history(chat_history: List[dict]) -> str:
    """
    Convert the chat history list into a plain-text block for the prompt.

    Keeps the last MEMORY_WINDOW_SIZE turns (user + assistant pairs).
    Each turn is formatted as "User: ...\nAssistant: ..." for clarity.

    Args:
        chat_history: List of {"role": "user"|"assistant", "content": str}.

    Returns:
        Formatted history string, or "No previous conversation." if empty.
    """
    if not chat_history:
        return "No previous conversation."

    # Take only the last N messages (2 messages = 1 turn)
    window = chat_history[-(config.MEMORY_WINDOW * 2):]

    lines = []
    for msg in window:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")

    return "\n".join(lines)