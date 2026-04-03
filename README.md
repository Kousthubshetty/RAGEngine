# RAG API

A production-ready Retrieval-Augmented Generation API built with FastAPI, LangChain, ChromaDB, and Claude.

## Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)
- (Optional) Docker & Docker Compose

## Quick Start (Local)

### 1. Clone & enter the project

```bash
cd learn-RAG
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This will install FastAPI, LangChain, ChromaDB, sentence-transformers, and all other required packages. The first install downloads the `all-MiniLM-L6-v2` embedding model (~80MB).

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

All other settings have sensible defaults. See [Configuration](#configuration) for the full list.

### 5. Add documents to the knowledge base

Create subfolders inside `knowledge/` for each knowledge base. Each subfolder becomes a separately queryable collection:

```
knowledge/
├── faq/
│   ├── general.txt
│   └── pricing.md
├── mahabharata/
│   ├── chapter1.txt
│   └── chapter2.pdf
└── sample/
    ├── sample.txt
    └── sample.md
```

Supported formats:

| Format | Extension | Notes |
|--------|-----------|-------|
| Plain text | `.txt` | Simple text files |
| Markdown | `.md` | Requires `unstructured` (included) |
| PDF | `.pdf` | Requires `poppler-utils` system package for some PDFs |
| JSON | `.json` | Loaded via jq-style parsing |

Sample files are included in `knowledge/sample/` to get started.

### 6. Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

On first startup, the embedding model is loaded into memory. This takes a few seconds.

### 7. Load the knowledge base

```bash
# Refresh all knowledge bases
curl -X POST http://localhost:8000/refresh-knowledge

# Refresh a specific knowledge base
curl -X POST http://localhost:8000/refresh-knowledge \
  -H 'Content-Type: application/json' \
  -d '{"knowledge": "sample"}'
```

Expected response:

```json
{
  "status": "success",
  "knowledge": "sample",
  "doc_count": 2,
  "chunk_count": 3,
  "processing_time": 1.25
}
```

### 8. List available knowledge bases

```bash
curl http://localhost:8000/knowledge
```

```json
{"knowledge_bases": ["faq", "mahabharata", "sample"]}
```

### 9. Ask questions

```bash
# Query a specific knowledge base
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is RAG?", "knowledge": "sample"}'

# Streaming response (Server-Sent Events)
curl -N -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is RAG?", "knowledge": "sample", "stream": true}'

# Health check
curl http://localhost:8000/health
```

## Running with Docker

### Build and run

```bash
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY

docker compose up --build
```

The Docker setup:
- Uses Python 3.12-slim base image
- Installs `poppler-utils` for PDF support
- Pre-downloads the embedding model at build time (faster startup)
- Mounts `knowledge/` as read-only
- Persists ChromaDB data in a named Docker volume

### Stop

```bash
docker compose down
```

To also remove the persisted vector DB volume:

```bash
docker compose down -v
```

## API Reference

### `GET /health`

Returns server status.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok"}
```

### `GET /knowledge`

Lists all available knowledge bases (subfolders in `knowledge/`).

```bash
curl http://localhost:8000/knowledge
```

```json
{"knowledge_bases": ["faq", "mahabharata", "sample"]}
```

### `POST /refresh-knowledge`

Reloads documents into the vector store. Pass `knowledge` to refresh a single knowledge base, or omit to refresh all.

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `knowledge` | string | `null` | Specific knowledge base to refresh. Omit to refresh all. |

```bash
# Refresh all
curl -X POST http://localhost:8000/refresh-knowledge

# Refresh specific
curl -X POST http://localhost:8000/refresh-knowledge \
  -H 'Content-Type: application/json' \
  -d '{"knowledge": "faq"}'
```

```json
{
  "status": "success",
  "knowledge": "faq",
  "doc_count": 2,
  "chunk_count": 5,
  "processing_time": 2.48
}
```

### `POST /ask`

Ask a question against a specific knowledge base.

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string (1-2000 chars) | *required* | The question to ask |
| `knowledge` | string | *required* | Knowledge base to query (e.g. `"faq"`, `"mahabharata"`) |
| `top_k` | integer (1-20) | `4` | Number of document chunks to retrieve |
| `stream` | boolean | `false` | Enable Server-Sent Events streaming |

**Standard request:**

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What are the benefits of RAG?", "knowledge": "sample", "top_k": 4}'
```

**Standard response:**

```json
{
  "answer": "The benefits of RAG include...",
  "sources": [
    {
      "content": "chunk text...",
      "metadata": {"source": "knowledge/sample/sample.txt"}
    }
  ]
}
```

**Streaming request:**

```bash
curl -N -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is RAG?", "knowledge": "sample", "stream": true}'
```

**Streaming response (SSE):**

```
data: {"token": "RAG"}
data: {"token": " stands"}
data: {"token": " for"}
...
data: {"done": true, "sources": [...]}
```

## Configuration

All settings are configured via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *required* | Your Anthropic API key |
| `MODEL_NAME` | `claude-sonnet-4-20250514` | Claude model to use |
| `CHUNK_SIZE` | `1000` | Max characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `TOP_K` | `4` | Default number of chunks to retrieve |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage directory |
| `KNOWLEDGE_DIR` | `./knowledge` | Directory containing source documents |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model for embeddings |
| `RATE_LIMIT_ASK` | `10/minute` | Rate limit for `/ask` endpoint |
| `RATE_LIMIT_REFRESH` | `2/minute` | Rate limit for `/refresh-knowledge` endpoint |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Architecture

```
User → FastAPI → RAG Chain → Claude (LLM)
                    ↓
              ChromaDB (vector search)
                    ↓
              knowledge/ (documents)
```

- **FastAPI** — Async web framework with automatic OpenAPI docs at `/docs`
- **LangChain (LCEL)** — Chains retrieval + prompt + LLM with native streaming
- **ChromaDB** — In-process vector store with disk persistence (no external DB needed)
- **all-MiniLM-L6-v2** — Local embedding model via sentence-transformers (free, no API costs)
- **Claude** — LLM for answer generation via Anthropic API

## Project Structure

```
app/
├── main.py                 # FastAPI app, lifespan, middleware registration
├── config.py               # Pydantic Settings from .env
├── dependencies.py         # Service singletons (vectorstore, RAG service)
├── models/
│   ├── requests.py         # AskRequest schema
│   └── responses.py        # AskResponse, RefreshResponse, SourceDocument
├── routers/
│   ├── knowledge.py        # POST /refresh-knowledge
│   └── query.py            # POST /ask
├── services/
│   ├── document_loader.py  # Multi-format doc loading + text splitting
│   ├── vectorstore.py      # ChromaDB operations + TTL cache
│   └── rag_chain.py        # LCEL chain, query, stream_query
└── middleware/
    ├── error_handler.py    # Global exception → JSON responses
    └── rate_limiter.py     # SlowAPI rate limiting
knowledge/                  # Knowledge bases (each subfolder = one knowledge base)
chroma_db/                  # Persisted vector DB (auto-created)
```

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `POST /ask` | 10 requests/minute per IP |
| `POST /refresh-knowledge` | 2 requests/minute per IP |

Exceeding the limit returns `429 Too Many Requests`.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Ensure virtual environment is activated: `source .venv/bin/activate` |
| Slow first startup | The embedding model is being downloaded/loaded — this is normal |
| PDF loading fails | Install `poppler-utils`: `brew install poppler` (macOS) or `apt-get install poppler-utils` (Linux) |
| `ANTHROPIC_API_KEY` error | Check your `.env` file has a valid key |
| 0 chunks after refresh | Check `knowledge/` folder has supported files and check server logs for errors |





#### RUN LOCALLY                              
                                                                                
  # List knowledge bases
  curl http://localhost:8000/knowledge

  # Ask against specific knowledge
  curl -X POST http://localhost:8000/ask \
    -H 'Content-Type: application/json' \
    -d '{"question": "What is RAG?", "knowledge": "sample"}'

  # Refresh one knowledge base
  curl -X POST http://localhost:8000/refresh-knowledge \
    -H 'Content-Type: application/json' \
    -d '{"knowledge": "faq"}'

  # Refresh all
  curl -X POST http://localhost:8000/refresh-knowledge