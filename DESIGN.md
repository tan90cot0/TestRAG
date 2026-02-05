# RAG System — Design Document

**Status**: Living document. Vector store: **Qdrant** (embeddings padded to 1536).

---

## 1. Overview

- **Goal**: RAG pipeline over 100 synthetic emails: load → chunk → embed (pad to 1536) → store in **Qdrant** → retrieve (with optional payload filters) → generate with Mistral.
- **Constraints**: No end-to-end RAG frameworks (LangChain, LlamaIndex). Use sentence-transformers, Qdrant client, and Mistral API only.
- **Quality**: End-to-end tests for retrieval and generation; design and tradeoffs documented here and in README.

---

## 2. Architecture

```
emails/*.txt → [Ingest] → ParsedEmail[]
                ↓
         [Chunker] → Chunk[] (text + metadata)
                ↓
    [Embedder] (sentence-transformers) → vectors → pad to 1536
                ↓
    [Qdrant] ← upsert(ids, vectors, payload: text, source_file, subject, from, to)
                ↓
    [Retriever] ← query_vector (1536) + optional payload filter
                ↓
    [Generator] (Mistral) ← prompt(context, query) → answer
```

---

## 3. Design Decisions

### 3.1 Document loading and parsing
- Parse each `emails/email_*.txt` into `subject`, `from_name`, `from_email`, `to_name`, `to_email`, `body`, `source_file`. Line-based/regex.

### 3.2 Chunking strategy
- Paragraph-based (split on `\n\n`). Each chunk has email context prepended; metadata: `source_file`, `subject`, `from`, `to`.

### 3.3 Embedding
- **Model**: sentence-transformers `all-mpnet-base-v2` (768 dims).
- **Padding**: Vectors are padded with zeros to **1536** dimensions for Qdrant (configurable via `QDRANT_VECTOR_SIZE`). Same for index and query.

### 3.4 Vector store (Qdrant)
- **Client**: `qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)` from `.env`.
- **Collection**: Single collection; `VectorParams(size=1536, distance=COSINE)`.
- **Payload** per point: `text`, `source_file`, `subject`, `from`, `to` (all strings). Point IDs: UUID from chunk_id (stable).

### 3.5 Retrieval
- Embed query → pad to 1536 → `client.search(collection_name, query_vector=..., limit=k, query_filter=...)`. Filter: `where` dict converted to Qdrant `Filter(must=[FieldCondition(key=..., match=MatchValue(value=...))])`.

### 3.6 Generation (Mistral)
- Mistral API; system prompt (answer only from context); user message = context + question.

### 3.7 Configuration
- **Env**: `MISTRAL_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_NAME`, `QDRANT_VECTOR_SIZE`, `EMAILS_DIR`, `EMBEDDING_MODEL`, `MISTRAL_MODEL`, `TOP_K`. Load via `python-dotenv` from project root `.env`.

### 3.8 Quality evaluation
- E2E tests: build index in Qdrant (session-scoped), run retrieval and generation tests. Require `QDRANT_URL` in env.

---

## 4. File layout

| Path | Purpose |
|------|--------|
| `rag/ingest.py` | Load and parse emails |
| `rag/chunking.py` | Paragraph chunking + metadata |
| `rag/embedding.py` | sentence-transformers; pad to 1536 |
| `rag/store.py` | Qdrant: create collection, upsert points |
| `rag/retrieve.py` | Query embedding + Qdrant search + filter |
| `rag/generate.py` | Mistral client, prompt, generate |
| `rag/pipeline.py` | Orchestrate index and ask |
| `rag/config.py` | Env config |
| `rag/models.py` | ParsedEmail, Chunk, RetrieveResult |
| `cli.py` | CLI: index, ask, eval |
| `tests/` | Unit + e2e tests |

---

## 5. Implementation status

- **Ingest, chunking**: Unchanged.
- **Embedding**: Pad to 1536 in `embed_texts` / `embed_query`.
- **Store**: Qdrant client, create_collection(vectors_config=1536), upsert with payload.
- **Retrieve**: Qdrant search, `where` dict → Filter.
- **Pipeline**: Collection name only (no persist flag).
- **Tests**: Conftest builds index in Qdrant using env `QDRANT_URL` and `QDRANT_COLLECTION_NAME`.
