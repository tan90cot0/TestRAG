# Mini RAG System

## Overview

A **Retrieval-Augmented Generation (RAG)** pipeline over 100 synthetic emails: load → chunk → embed (sentence-transformers, padded to 1536) → store in **Qdrant** → retrieve (with optional payload filters) → generate with **Mistral** API. Built without LangChain/LlamaIndex.

## Requirements

- **Python 3.10+**
- **Qdrant**: URL and API key in `.env` (see below).
- **Mistral API key** for generation in `.env`.

Create a virtualenv and install dependencies:

```bash
cd /path/to/TestRAG
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set:

- `MISTRAL_API_KEY` — for generation
- `QDRANT_URL` — e.g. `https://xxx.europe-west3-0.gcp.cloud.qdrant.io:6333`
- `QDRANT_API_KEY` — your Qdrant API key

## Quick start

1. **Build the index** (from the repo root):

```bash
python cli.py index
```

This reads all `emails/email_*.txt`, chunks them, embeds with sentence-transformers (padded to 1536), and upserts into the Qdrant collection.

2. **Ask a question**:

```bash
python cli.py ask "What did Helen Powell ask Nico about?"
```

3. **Optional filters** (exact match on payload):

```bash
python cli.py ask "Summarize the email" --subject "Meeting Request"
```

4. **Run tests** (unit + e2e; e2e require QDRANT_URL in `.env`):

```bash
python cli.py eval
# or: pytest tests/ -v
```

See **DESIGN.md** for design choices, tradeoffs, and quality evaluation.

## Configuration (environment)

| Variable | Default | Description |
|----------|---------|-------------|
| `MISTRAL_API_KEY` | (required for ask) | Mistral API key |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | (optional) | Qdrant API key (for cloud) |
| `QDRANT_COLLECTION_NAME` | `email_chunks` | Collection name |
| `QDRANT_VECTOR_SIZE` | `1536` | Vector dimension (embeddings padded to this) |
| `EMAILS_DIR` | `./emails` | Directory of email `.txt` files |
| `EMBEDDING_MODEL` | `all-mpnet-base-v2` | sentence-transformers model |
| `MISTRAL_MODEL` | `mistral-small-latest` | Mistral chat model |
| `TOP_K` | `5` | Number of chunks to retrieve |

Run: `python cli.py eval` or `pytest tests/ -v`. See **DESIGN.md** §4 for the full evaluation approach.

## How to run (exact commands)

```bash
cd /path/to/TestRAG
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Copy .env.example to .env and set MISTRAL_API_KEY, QDRANT_URL, QDRANT_API_KEY
python cli.py index
python cli.py ask "What did Helen Powell ask Nico about?"
```

## Submission

- Public git repository with this submission; share the repository link.
- Do not fork this repository or create pull requests.
