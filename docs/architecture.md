# System Architecture

## Overview

The Regulatory Intelligence Assistant is a Retrieval-Augmented
Generation (RAG) system. Instead of asking an LLM what it "remembers"
about a regulation, the system first retrieves the relevant passages
from an indexed copy of the regulation text, then instructs a locally
hosted LLM to answer *only* from those passages, citing each one. This
grounds every answer in the source text and makes hallucinated
obligations detectable: any claim without a citation is invalid by
construction.

## Components

### 1. Ingestion (`regintel/ingestion/`)

- `download.py` fetches official regulation text (EUR-Lex HTML).
- `parser.py` converts documents into structured `Article` objects.
  Two strategies: EUR-Lex HTML parsing (uses Official Journal CSS
  classes) and a plain-text fallback that recognizes both `Article N`
  and Turkish `Madde N` headings, so KVKK/BDDK texts extracted from
  PDFs use the same pipeline.
- `chunker.py` splits articles into ~1,500-character chunks at
  paragraph boundaries. Chunks never cross article boundaries, and each
  carries metadata: regulation, article number, article title, chapter,
  chunk index, source URL, language. Each chunk's text is prefixed with
  its regulation + article heading so the embedding encodes its context.

### 2. Retrieval (`regintel/retrieval/store.py`)

- **Semantic**: chunks are embedded with
  `paraphrase-multilingual-MiniLM-L12-v2` (multilingual — covers
  English and Turkish) and stored in ChromaDB (cosine similarity,
  persistent local storage).
- **Keyword**: a BM25 index over the same chunks. Regulatory queries
  are full of exact tokens — article numbers, defined terms — where
  lexical search beats embeddings.
- **Fusion**: Reciprocal Rank Fusion (RRF) merges the two rankings
  without needing score normalization.
- **Metadata filtering**: any search can be restricted to a single
  regulation, which also powers comparison mode.

### 3. Generation (`regintel/generation/`)

- `llm.py` defines a minimal `LLM` interface with two implementations:
  `OllamaLLM` (local inference server, default `qwen2.5:7b-instruct`)
  and `EchoLLM` (no-LLM fallback returning raw retrieved context, used
  in tests and early development). Swapping models or backends touches
  one file.
- `prompts.py` holds the system prompt (the compliance guardrail:
  answer only from sources, cite everything, refuse when the sources
  don't cover the question, note that output is information rather than
  legal advice) and the answer/comparison templates.
- `rag.py` orchestrates: retrieve → format numbered sources → generate
  → return `RagResponse` (answer + the exact chunks it was based on).

  **Comparison mode** retrieves separately per regulation with metadata
  filters. A single unfiltered search for "How do DORA and NIS2 differ
  on X?" typically returns chunks from only one framework, causing the
  LLM to fabricate the other side; per-regulation retrieval guarantees
  both sides are present in context.

### 4. Interfaces

- `regintel/api/main.py` — FastAPI: `POST /ask`, `POST /compare`,
  `GET /search`, `GET /regulations`, `GET /health`. OpenAPI docs served
  at `/docs`.
- `ui/app.py` — Streamlit chat interface with mode selection
  (ask / compare), per-regulation filtering, and expandable source
  panels under every answer.

## Data flow

```
question ──> hybrid retrieval (top-k chunks, optional regulation filter)
        ──> prompt: system guardrails + numbered sources + question
        ──> Ollama (localhost)
        ──> answer with [n] citations + source list
```

## Confidentiality

All persistent state (ChromaDB, BM25 corpus, processed JSON) lives
under `data/` on local disk. All inference (embeddings, LLM) runs on
localhost. Internet is needed only for the one-time download of
regulation texts and the embedding model — both can be fetched on a
separate machine and copied into an air-gapped environment.

## Known limitations / future work

- Annexes and recitals are not yet parsed as first-class citable units.
- No re-ranker; a cross-encoder re-ranking stage is the highest-value
  retrieval upgrade if evaluation shows misses.
- Evaluation harness (gold Q&A set, retrieval hit-rate, citation
  accuracy) is the next deliverable.
