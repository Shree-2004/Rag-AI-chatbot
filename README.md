# 🧠 RAG AI Chatbot

A **production-ready Retrieval-Augmented Generation (RAG) chatbot** that lets you upload PDF documents and ask questions with cited, context-grounded answers.

Built with **LangChain**, **FAISS/Chroma**, **OpenAI/Gemini**, and **Streamlit**.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     USER  (Streamlit UI — app.py)                   │
│  Upload PDFs │ Select Provider │ Ask Questions │ Export Chat        │
└──────┬────────────────┬──────────────────┬──────────────────────────┘
       │                │                  │
       ▼                │                  ▼
┌──────────────┐        │         ┌──────────────────┐
│  loader.py   │        │         │   pipeline.py    │
│  PDF → Docs  │        │         │   RAG Chain:     │
│  + MD5 hash  │        │         │   1. Retrieve    │
└──────┬───────┘        │         │   2. Score       │
       │                │         │   3. Prompt      │
       ▼                │         │   4. Stream LLM  │
┌──────────────┐        │         └────┬───────┬─────┘
│  chunker.py  │        │              │       │
│  Split + tag │        │              │       │
│  metadata    │        │              ▼       ▼
└──────┬───────┘        │     ┌────────────┐ ┌──────────┐
       │                │     │retriever.py│ │  llm.py  │
       ▼                │     │FAISS/Chroma│ │ OpenAI / │
┌──────────────┐        │     │+ BM25      │ │ Gemini   │
│embeddings.py │        │     │+ Persist   │ └──────────┘
│OpenAI / HF   │────────┘     └────────────┘
└──────────────┘
       │
       └──► All settings from config.py (single source of truth)
```

---

## 📁 Project Structure

```
RAG AI CHATBOT/
├── src/
│   ├── __init__.py       # Package docstring
│   ├── loader.py         # PDF → LangChain Documents + MD5 hashing
│   ├── chunker.py        # RecursiveCharacterTextSplitter + metadata
│   ├── embeddings.py     # Embedding factory (OpenAI / HuggingFace)
│   ├── retriever.py      # Vector store build, persist, load, hybrid search
│   ├── llm.py            # LLM factory (OpenAI / Gemini) + streaming
│   └── pipeline.py       # Full RAG chain: retrieve → score → prompt → stream
├── app.py                # Streamlit UI
├── config.py             # All settings (models, chunk size, thresholds, flags)
├── requirements.txt      # Pinned dependencies
├── .env.example          # API key template
├── .env                  # Your actual keys (gitignored)
├── vector_store/         # Auto-created: persisted FAISS/Chroma indexes
└── README.md             # This file
```

---

## 🚀 Quick Start

### 1. Clone & enter the project

```bash
git clone https://github.com/your-username/rag-ai-chatbot.git
cd rag-ai-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your API key(s):

```env
# Only need the key for your chosen provider
OPENAI_API_KEY=sk-your-key-here        # If using OpenAI
GOOGLE_API_KEY=your-key-here            # If using Gemini
```

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## ⚙️ Configuration

All settings live in **`config.py`**. Key options:

| Setting | Default | Options |
|---|---|---|
| `LLM_PROVIDER` | `"openai"` | `"openai"`, `"gemini"` |
| `OPENAI_MODEL` | `"gpt-4o-mini"` | Any OpenAI chat model |
| `GEMINI_MODEL` | `"gemini-1.5-flash"` | Any Gemini model |
| `EMBEDDING_PROVIDER` | `"openai"` | `"openai"`, `"huggingface"` |
| `VECTOR_STORE_TYPE` | `"faiss"` | `"faiss"`, `"chroma"` |
| `CHUNK_SIZE` | `1000` | Integer (chars) |
| `CHUNK_OVERLAP` | `200` | Integer (chars) |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `MEMORY_WINDOW` | `5` | Conversation turns to remember |
| `RELEVANCE_THRESHOLD` | `0.30` | Below this → "I don't know" |
| `ENABLE_HYBRID_SEARCH` | `False` | Dense + BM25 keyword search |
| `ENABLE_RERANKING` | `False` | Cross-encoder reranking |
| `STREAMING` | `True` | Real-time token streaming |

You can also change LLM and embedding providers **at runtime** from the sidebar dropdown — no restart needed.

---

## 🔄 Swapping Providers

### OpenAI → Gemini

```env
# In .env:
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-google-api-key
```

Or select "gemini" from the sidebar dropdown at runtime.

### FAISS → Chroma

```env
# In .env:
VECTOR_STORE_TYPE=chroma
```

Chroma persists automatically via its SQLite backend. No code changes needed.

### OpenAI Embeddings → HuggingFace (Free, Local)

```env
# In .env:
EMBEDDING_PROVIDER=huggingface
```

Uses `all-MiniLM-L6-v2` locally on CPU — no API key required. Great for development and testing.

---

## ⚡ Advanced Features

### 1. Multi-PDF Support
Upload multiple PDFs simultaneously. Each chunk's metadata tracks which file it came from, so source citations show the correct filename.

### 2. Conversation Memory
The chatbot remembers the last 5 turns (configurable via `MEMORY_WINDOW`). Follow-up questions like *"What did it say about that?"* work correctly because the chat history is injected into the prompt.

### 3. Confidence Scoring
Every retrieved chunk comes with a cosine similarity score and a badge:
- 🟢 **High** (≥ 0.75)
- ⚡ **Medium** (≥ 0.50)
- 🔴 **Low** (< 0.50)

If ALL chunks score below `RELEVANCE_THRESHOLD` (0.30), the bot responds with *"I don't have enough information…"* instead of hallucinating.

### 4. Hybrid Search (Optional)
Set `ENABLE_HYBRID_SEARCH=True` in config.py to combine:
- **Dense vector search** (semantic similarity via FAISS/Chroma)
- **BM25 keyword search** (exact term matching)
- Merged via `EnsembleRetriever` with configurable weights

### 5. Cross-Encoder Reranking (Optional)
Set `ENABLE_RERANKING=True` to re-score the initial top-k results with a cross-encoder model (`ms-marco-MiniLM-L-6-v2`) for higher precision.

### 6. Streaming Responses
Tokens stream to the UI in real time via `st.write_stream()` — no waiting for the full response.

### 7. Export Chat History
Download the full Q&A session as **JSON** or **TXT** from the sidebar.

### 8. Vector Store Persistence
FAISS indexes are saved to disk using an MD5 hash of the uploaded file(s). Re-uploading the same files loads the cached index instantly — no re-embedding.

---

## 🧩 How Features Interact

```
Upload PDF(s)
    │
    ├─ loader.py computes MD5 hash per file + combined hash
    │
    ├─ If combined_hash matches cached index → load from disk (instant ✅)
    │
    ├─ Otherwise:
    │   ├─ chunker.py splits into overlapping chunks (metadata preserved)
    │   ├─ embeddings.py embeds each chunk (OpenAI API or local HuggingFace)
    │   ├─ retriever.py stores in FAISS/Chroma + saves to disk
    │   └─ If ENABLE_HYBRID_SEARCH: also builds BM25 index from raw text
    │
    └─ User asks a question
        │
        ├─ pipeline.py retrieves top-k chunks with cosine scores
        ├─ Confidence gate: if best score < 0.30 → "I don't know"
        ├─ If ENABLE_RERANKING: cross-encoder re-scores and reorders
        ├─ Chat history (last 5 turns) + context + question → prompt template
        ├─ LLM streams response token by token → Streamlit renders live
        └─ Source chunks displayed with relevance badges in expandable accordion
```

---

## 🔮 Extending the Project

### Add Web Scraping as a Data Source
Create a `src/web_loader.py` using `WebBaseLoader` from LangChain:
```python
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://example.com/article")
docs = loader.load()
```
Feed the docs into the same `chunker.py → retriever.py` pipeline.

### Add SQL Database Retrieval
Use LangChain's `SQLDatabaseChain` to query databases alongside document retrieval:
```python
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
```

### Add More File Types
Add loaders for DOCX, TXT, CSV, etc. in `loader.py`:
```python
from langchain_community.document_loaders import Docx2txtLoader, CSVLoader, TextLoader
```

### Deploy to Production
- **Docker**: Add a `Dockerfile` with `streamlit run app.py`
- **Cloud**: Deploy to Streamlit Cloud, GCP Cloud Run, or AWS ECS
- **Auth**: Add `streamlit-authenticator` for user login

---

## 📄 License

MIT License — use freely for personal and commercial projects.

---

## 🤝 Contributing

1. Fork it
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Built with ❤️ using LangChain, FAISS, OpenAI, Google Gemini & Streamlit**
"# Rag-AI-chatbot" 
