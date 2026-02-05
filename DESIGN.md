# RAG System — Design Document

## 1. Overview

This document describes the design of a **Retrieval-Augmented Generation (RAG)** pipeline over 100 email documents: ingest → chunk → embed → store in Qdrant → retrieve → generate with Mistral. It documents **design choices and tradeoffs** and explains the **approach to quality evaluation**.

**Constraints**: No end-to-end RAG frameworks (e.g. LangChain, LlamaIndex). The implementation uses only focused libraries: sentence-transformers, qdrant-client, and the Mistral API.

---

## 2. Architecture

```
emails/*.txt → [Ingest] → ParsedEmail[]
                ↓
         [Chunk] → Chunk[] (text + metadata)
                ↓
    [Embed] (sentence-transformers) → 768-dim vectors → pad to 1536
                ↓
    [Qdrant] ← upsert(ids, vectors, payload)
                ↓
    [Retrieve] ← query vector + optional payload filter → top-k chunks
                ↓
    [Generate] (Mistral) ← prompt(context, query) → answer
```

---

## 3. Design Choices and Tradeoffs

### 3.1 Document loading and parsing

**Choice**: Line-based parsing with regex for `Subject:`, `From:`, `To:` and a blank line to separate header from body.

**Tradeoffs**:
- **Pro**: Simple, no external HTML/email parser; works for the given plain-text format.
- **Con**: Fragile if format changes (e.g. multi-line headers, encoding issues). Acceptable for a fixed synthetic dataset.

### 3.2 Chunking strategy

**Choice**: Paragraph-based chunking: split body on `\n\n`. Each chunk keeps full email context in the text (Subject, From, To) and stores the same in metadata for filtering.

**Tradeoffs**:
- **Pro**: Preserves semantic units; metadata enables filter-by-subject/sender/receiver without re-parsing.
- **Con**: Paragraphs vary in length; very short or long paragraphs may hurt retrieval granularity. Alternative (e.g. fixed token windows) would give more uniform chunks but could split sentences. For 100 emails with moderate body length, paragraph chunking is a good balance.

### 3.3 Embedding

**Choice**: sentence-transformers model `all-mpnet-base-v2` (768 dimensions). Vectors are **padded with zeros to 1536 dimensions** before storing in Qdrant and when embedding the query.

**Tradeoffs**:
- **Pro**: Single strong open-source model; no API cost for embedding; 1536 matches common cloud vector DB expectations (e.g. OpenAI dimension).
- **Con**: Padding does not add information; it increases storage and bandwidth. Alternative would be a native 1536-dim model (e.g. OpenAI) at higher cost and dependency.

### 3.4 Vector store (Qdrant)

**Choice**: Qdrant cloud or local; one collection; cosine distance; payload fields `text`, `source_file`, `subject`, `from`, `to`; keyword payload indexes on those fields for filtering; batched upserts with configurable timeout.

**Tradeoffs**:
- **Pro**: Managed vector DB with filtering; payload indexes allow fast subject/from/to filters; batching avoids timeouts on large upserts.
- **Con**: Requires Qdrant URL (and API key for cloud). Index build is not incremental (collection is recreated on each full index run).

### 3.5 Retrieval

**Choice**: Embed query with the same model (padded to 1536), then `query_points` with optional Qdrant filter built from a simple `where` dict (e.g. `{"subject": "Meeting Request"}`). Return top-k by cosine similarity.

**Tradeoffs**:
- **Pro**: Same embedding space for index and query; filters narrow results by metadata without re-ranking.
- **Con**: No hybrid (keyword + vector) or re-ranker; top-k is fixed per request.

### 3.6 Generation (Mistral)

**Choice**: Mistral chat API with a system prompt that instructs the model to answer only from the provided context and to cite sources when possible. User message = concatenated context chunks + question.

**Tradeoffs**:
- **Pro**: Clear instruction to reduce hallucination; source labels in context support traceability.
- **Con**: Depends on external API and key; no fallback model. Prompt design is minimal; more structured prompts (e.g. strict templates) could improve consistency.

### 3.7 Configuration

**Choice**: All config via environment variables loaded from `.env` (python-dotenv) at import time: API keys, Qdrant URL, collection name, vector size, model names, top-k, optional timeouts and batch sizes.

**Tradeoffs**:
- **Pro**: No code changes for different environments; `.env.example` documents required and optional vars.
- **Con**: No validation at startup beyond “key present or not”; invalid values surface at runtime.

---

## 4. Quality Evaluation

Evaluation is implemented as **unit tests** and **end-to-end (e2e) tests** in `tests/`, run via `python cli.py eval` or `pytest tests/ -v`.

### 4.1 Unit tests

- **Ingest** (`test_ingest.py`): Parsing of valid email content and handling of missing headers; loading a real file from `emails/`.
- **Chunking** (`test_chunking.py`): Single- and multi-paragraph emails produce the expected number of chunks and correct metadata.

These tests do not require Qdrant or Mistral and run quickly.

### 4.2 End-to-end retrieval

- **Index presence**: After building the index (session-scoped fixture), a generic query returns results with expected shape (text, metadata including `subject`).
- **Topic relevance**: Queries about “meetings”, “budget”, and “training” are checked to see if at least one of the top-k results has the expected topic (via subject or text). This is a **qualitative relevance check**: we assert that the right *type* of email appears in the top-k, not a numeric IR metric.
- **Metadata filter**: A filter on `subject` is applied; we assert all returned chunks have that exact subject. This validates that payload indexes and filter translation work.

E2e retrieval tests require `QDRANT_URL` (and `QDRANT_API_KEY` for cloud) in the environment. The fixture builds the index once per test session.

### 4.3 End-to-end generation

- **Prompt structure** (with Mistral mocked): We patch the Mistral client and assert that the model is called with messages that contain the user query and context (including “Question:” and “Context”/“Source”). This checks that the pipeline passes the right inputs to the LLM.
- **Answer shape**: With the same mock, we assert the pipeline returns a non-empty string.
- **Live groundedness** (optional): If `MISTRAL_API_KEY` is set, one test runs a real request and asserts the answer is non-trivial and does not look like a generic “I don’t have access” response when context was provided. This is a light **qualitative check** that the model uses the context.

We do not use a formal metric (e.g. faithfulness or answer correctness) on a labelled set; the focus is on integration and basic behavioural checks.

### 4.4 Pipeline integration

- **No-results fallback**: When retrieval is constrained by a filter that matches no documents, the pipeline returns a fallback message (e.g. “no relevant emails”) and does not call the LLM. The test asserts zero results and that the answer string contains the fallback wording.

### 4.5 Summary of evaluation approach

| Aspect | Approach |
|--------|----------|
| Correctness of ingest/chunking | Unit tests on parsing and chunk counts. |
| Retrieval quality | E2e topic and filter tests; qualitative (expected topic in top-k). |
| Generation quality | Mock-based checks of prompt and response shape; optional live check for groundedness. |
| Robustness | No-results path and fallback message covered by e2e. |

Running `python cli.py eval` (or `pytest tests/ -v`) executes this full suite. E2e tests that need Qdrant or Mistral are skipped or fail clearly if the corresponding environment variables are missing.

---

## 5. File Layout

| Path | Purpose |
|------|--------|
| `rag/ingest.py` | Load and parse email files. |
| `rag/chunking.py` | Paragraph chunking and metadata. |
| `rag/embedding.py` | sentence-transformers; pad to 1536. |
| `rag/store.py` | Qdrant client; create collection; payload indexes; batched upsert. |
| `rag/retrieve.py` | Query embedding; Qdrant query_points; filter translation. |
| `rag/generate.py` | Mistral client; prompt; chat completion. |
| `rag/pipeline.py` | Orchestrate index and ask. |
| `rag/config.py` | Env config (dotenv). |
| `rag/models.py` | ParsedEmail, Chunk, RetrieveResult. |
| `cli.py` | CLI: index, ask, eval. |
| `tests/` | Unit and e2e tests. |

---

## 6. Setup and Run

- Copy `.env.example` to `.env` and set `MISTRAL_API_KEY`, `QDRANT_URL`, and (for cloud) `QDRANT_API_KEY`.
- Install: `pip install -r requirements.txt`.
- Index: `python cli.py index`.
- Query: `python cli.py ask "your question"`; optional `--subject "Meeting Request"` (or other filters).
- Evaluate: `python cli.py eval`.

See **README.md** for full setup and configuration details.
