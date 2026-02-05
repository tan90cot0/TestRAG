# Mini RAG System — Problem Statement & Proposed Solution

This document explains the task, constraints, and a concrete end-to-end plan for building the RAG pipeline **without using** end-to-end frameworks like LangChain or LlamaIndex.

---

## 1. Problem Statement

### 1.1 Goal

Build a **Retrieval-Augmented Generation (RAG)** pipeline that:

1. **Processes** the provided documents (100 synthetic emails).
2. **Retrieves** the most relevant pieces of text for a given user question.
3. **Generates** an answer that is grounded in the retrieved context (and ideally cites or reflects it).

RAG improves over “raw” LLM answers by tying responses to your corpus, reducing hallucination and keeping answers factual and traceable.

### 1.2 What You’re Given

- **Dataset**: 100 synthetic emails in `emails/` (e.g. `email_001.txt` … `email_100.txt`).
- **Email structure** (consistent across files):
  - **Subject** line
  - **From**: sender name and email (from a pool of 200 people)
  - **To**: receiver name and email
  - **Body**: 100+ words, with diverse topics (project updates, meeting requests, budget approvals, technical issues, client feedback, team announcements, deadline extensions, training, vendor proposals, performance reviews).

So the system must:

- **Chunk and process** these email documents.
- **Retrieve** relevant chunks for arbitrary user queries.
- **Generate** accurate, context-based answers.

### 1.3 Requirements (from the README)

| Component        | What you need to do |
|-----------------|----------------------|
| **Document chunking** | Split documents into appropriate chunks; handle different sizes; **explain your chunking strategy**. |
| **Embedding**        | Generate embeddings for chunks; choose an embedding model; store them efficiently. |
| **Retrieval**        | Similarity search over chunks; embed the query; return top‑k relevant results. |
| **Generation**       | Use retrieved context in a prompt; design effective prompts; integrate with an LLM. |

### 1.4 Constraints

- **No end-to-end RAG frameworks**: Do not use LangChain, LlamaIndex, etc. Build the core pipeline yourself or use **individual** libraries (e.g. `sentence-transformers`, `qdrant-client`, `openai`, etc.).
- **Document** design choices and tradeoffs.
- **Explain** how you will evaluate quality (even if only qualitatively or with a small set of test questions).

### 1.5 Time Box

- **75 minutes** total — so the solution should be minimal but complete and well-documented, not over-engineered.

---

## 2. Dataset Summary

- **Format**: Plain text files; structure is consistent (Subject, From, To, body with salutation and sign-off).
- **Size**: 100 emails; each body is 100+ words (~10–20 sentences).
- **Topics**: 10 template topics (from `generate_emails.py`), so multiple emails share the same topic with slight variations.
- **Entities**: 200 unique people (names + emails) as senders/receivers; useful for queries like “Who wrote about budget?” or “Emails involving Anna”.

Implications for the plan:

- Chunking can be at the **email level** (e.g. one chunk per email) or **sub-email** (e.g. by paragraph or fixed-size tokens). Tradeoff: coarser = fewer chunks and simpler; finer = more precise retrieval but more chunks and overlap.
- Metadata (Subject, From, To) should be **preserved** with each chunk so retrieval can use and the generator can cite “which email” or “who said what”.

---

## 3. Proposed End-to-End Solution

High-level flow:

```
[Email .txt files] → Parse & Chunk → [Chunks + metadata]
         ↓
[Chunks] → Embed (model) → [Vector store]
         ↓
[User query] → Embed (same model) → Similarity search → Top‑k chunks
         ↓
[Top‑k chunks + query] → Prompt → LLM → [Answer]
```

Below we spell out each step and alternatives.

---

### 3.1 Document Loading & Parsing

- **Action**: Read each file in `emails/`, parse into structured fields: `subject`, `from`, `to`, `body` (and optionally `filename` / `id`).
- **Why**: Clean structure makes chunking and metadata attachment straightforward; the body is the main content to chunk and embed.
- **Implementation note**: Simple line-based or regex parsing is enough (e.g. split on “Subject:”, “From:”, “To:”, then take the rest as body). No framework required.

---

### 3.2 Chunking Strategy

**Options (with tradeoffs):**

| Strategy              | Pros | Cons |
|-----------------------|------|------|
| **One chunk per email** | Simple, preserves full context, 100 chunks only. | Less precise retrieval; long bodies may exceed ideal context length for embedding/generation. |
| **Fixed-size chunks** (e.g. 256 tokens, 50% overlap) | Standard RAG approach; good for long docs. | Overlap and split sentences; need to attach metadata (subject, from, to) to each chunk. |
| **Semantic chunking** (e.g. by paragraph or sentence group) | Aligns chunks with natural boundaries. | Slightly more logic; paragraphs in these emails are sometimes short. |

**Recommended for 75 minutes**: **One chunk per email**, with the **full text** (subject + from + to + body) in the chunk text so that:

- Embeddings capture the whole email semantics.
- No overlap or boundary issues.
- Metadata can still be stored separately (e.g. `source_file`, `subject`, `from`, `to`) for filtering or display.

If you want slightly finer retrieval: **paragraph-based chunking** (split body by `\n\n`), and prepend subject + from + to to each chunk so every chunk is self-contained. Document this as “we chunk by paragraph and attach email metadata to each chunk.”

**What to document**: Chunk size (e.g. “one chunk per email, ~100–150 words per chunk”), how metadata is attached, and why you chose this over fixed-size or sentence-level chunking.

---

### 3.3 Embedding

- **Action**: Run each chunk through an **embedding model** to get a vector; store (chunk_id, vector, optional metadata) in a vector store.
- **Model choice** (pick one and justify briefly):
  - **OpenAI `text-embedding-3-small`** (or `ada-002`): Good quality, simple API; requires API key and network.
  - **Hugging Face `sentence-transformers`** (e.g. `all-MiniLM-L6-v2` or `all-mpnet-base-v2`): Free, local, good for 100 chunks; no API cost.
- **Recommendation for 75 min**: Use **sentence-transformers** locally to avoid API setup and to keep the pipeline runnable offline. Dimension is typically 384 or 768; store in a small vector DB or in-memory.

**What to document**: Model name, embedding dimension, and why this model (e.g. “lightweight, good for short documents, no API dependency”).

---

### 3.4 Storing Embeddings

- **Options**: In-memory (e.g. numpy array + list of chunks), **Qdrant**, **FAISS**, or a simple dict/list with cosine similarity implemented by hand.
- **Recommendation**: **Qdrant** or an in-memory index (e.g. with **sentence-transformers**’ built-in util or numpy) so that you get “top‑k by similarity” with minimal code. Qdrant is a single library, not a full RAG framework, so it fits the constraint.
- **What to store per chunk**: `id`, `text` (chunk content), `embedding`, and metadata (`source_file`, `subject`, `from`, `to` at least).

---

### 3.5 Retrieval

- **Input**: User query string.
- **Steps**:
  1. Embed the query with the **same** embedding model used for chunks.
  2. Run **similarity search** (cosine or inner product) and get **top‑k** chunks (e.g. k = 3 or 5).
  3. Return the list of chunk texts (and optionally metadata) to the generation step.
- **k**: Start with **k = 3–5**; document that this is a hyperparameter and can be tuned for quality vs. context length.
- **Optional**: Add a simple **metadata filter** (e.g. “only emails from domain X”) if time allows; otherwise skip.

**What to document**: Similarity metric (e.g. cosine), value of k, and how you use metadata in retrieval (if at all).

---

### 3.6 Generation

- **Input**: User query + top‑k retrieved chunk texts (and optionally metadata).
- **Action**: Build a **single prompt** that includes:
  - **System or instruction**: “You answer questions using only the provided context. If the context does not contain the answer, say so.”
  - **Context**: The concatenated chunk texts (and optionally “Source: email_042” etc.).
  - **Question**: The user query.
  - **Output**: One coherent answer (and optionally “Based on email_042, …”).
- **LLM choice**:
  - **OpenAI API** (e.g. `gpt-4o-mini` or `gpt-3.5-turbo`): Easy, good quality; needs API key.
  - **Local model** (e.g. Ollama with `llama3` or `mistral`): No API; good for a self-contained demo.
- **Recommendation**: Use **OpenAI** or **Ollama** and state which one; keep the prompt short and the context clearly delimited (e.g. `Context:\n...\n\nQuestion: ...`).

**What to document**: Prompt structure (instruction + context + question), how you avoid the model ignoring context (instruction design), and which LLM you use.

---

### 3.7 Quality Evaluation

- **Options**:
  - **Manual**: 5–10 diverse questions (by topic and by entity), run the pipeline, and check if the answer is supported by the retrieved emails and is correct.
  - **Simple scoring**: For each question, record (1) whether the right email(s) were in top‑k, and (2) whether the generated answer is correct (binary or 1–3 scale). Report accuracy or % correct.
  - **No automated metrics**: Given 75 minutes, a short “Evaluation” section that describes **what you would check** (retrieval relevance, factual correctness, no hallucination) and 2–3 example Q&A pairs is acceptable.

**Recommended**: Write **5–10 test questions** (e.g. “What did Helen Powell ask Nico about?”, “Which emails mention budget approval?”, “Who requested a deadline extension?”). Run the pipeline once; in the README or PLAN, show:
- The question,
- The retrieved chunks (or their sources),
- The generated answer,
and briefly comment on correctness. State that “quality is evaluated by manual inspection of retrieval and generation on this set.”

---

## 4. Suggested Implementation Order (75 min)

1. **Parsing** (≈10 min): Load all emails; parse into `subject`, `from`, `to`, `body`; attach `filename`/`id`.
2. **Chunking** (≈5 min): Implement chosen strategy (e.g. one chunk per email); build list of chunks + metadata.
3. **Embedding + store** (≈15 min): Initialize embedding model; embed all chunks; put in Qdrant or in-memory index with metadata.
4. **Retrieval** (≈10 min): Embed query; get top‑k; return chunk texts (+ metadata).
5. **Generation** (≈15 min): Build prompt; call LLM; return answer. Optional: simple CLI or script that takes a question and prints the answer.
6. **Documentation & evaluation** (≈20 min): README (how to run, design choices, chunking/embedding/retrieval/generation), 5–10 test questions, and 2–3 example outputs with short commentary.

This leaves a small buffer and keeps the pipeline minimal but complete.

---

## 5. File/Layout Suggestion

- `load_emails.py` or `ingest.py`: Parse emails → list of structured docs.
- `chunk.py` or inside ingest: Chunking logic.
- `embed.py` or `index.py`: Embed chunks and build the vector store (or persist it to disk once).
- `retrieve.py`: Query embedding + top‑k retrieval.
- `generate.py` or `rag.py`: Prompt construction + LLM call; optionally ties load → retrieve → generate for a single query.
- `README.md`: How to run, dependencies, design choices, evaluation approach, and example Q&A.
- `PLAN.md`: This document (problem + proposed solution).

You can merge some of these into fewer files (e.g. `rag.py` = load + chunk + embed + retrieve + generate) to save time.

---

## 6. Dependencies (to document in README)

- Python 3.8+.
- Libraries (examples): `sentence-transformers`, `qdrant-client` (or numpy for in-memory), `openai` or `requests` (for OpenAI API) or whatever Ollama uses. No LangChain, no LlamaIndex.

---

## 7. Summary

| Step        | Recommendation |
|------------|----------------|
| **Parse**  | Subject, From, To, Body + id/filename. |
| **Chunk**  | One chunk per email (full text + metadata); document rationale. |
| **Embed**  | sentence-transformers (e.g. all-MiniLM-L6-v2) or OpenAI; store in Qdrant or in-memory. |
| **Retrieve** | Same model for query; cosine top‑k (e.g. k=3–5). |
| **Generate** | Single prompt: instruction + context + question; OpenAI or Ollama. |
| **Evaluate** | 5–10 manual questions; show retrieval + answer and briefly comment. |

This gives you a clear, end-to-end plan that satisfies the task statement and constraints, stays within 75 minutes, and produces a documentable, runnable RAG pipeline.
