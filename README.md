# Regulatory Intelligence Assistant

Source-backed regulatory Q&A for cybersecurity governance and compliance
teams. Ask questions about DORA (and later NIS2, KVKK, BDDK, ISO 27001)
in natural language; get answers grounded in the actual regulation text,
with citations to the specific articles. Runs **fully locally** — no
data leaves the machine.

*([Türkçe README](README.tr.md))*

## Architecture

```
EUR-Lex / PDFs ──> ingestion (parse into articles, chunk with metadata)
                        │
                        ▼
              ChromaDB (embeddings) + BM25 keyword index
                        │  hybrid retrieval (RRF fusion)
                        ▼
              RAG pipeline ──> local LLM via Ollama
                        │
                        ▼
          FastAPI backend  +  Streamlit chat UI
```

Key design decisions:

- **Article-aware chunking** — chunks never cross article boundaries, so
  every answer can cite "DORA, Article 17" accurately.
- **Hybrid retrieval** — BM25 catches exact terms ("Article 5",
  "madde 12"); embeddings catch conceptual questions; Reciprocal Rank
  Fusion combines both.
- **Comparison mode** — retrieval runs separately per regulation
  (metadata-filtered) so both frameworks are represented when comparing.
- **Multilingual embeddings** — Turkish regulations (KVKK, BDDK) work
  without re-architecting. The parser already understands "Madde N".
- **Swappable LLM layer** — Ollama by default; `EchoLLM` fallback lets
  you develop and test retrieval before any LLM is installed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com) and pull a model:

```bash
ollama pull qwen2.5:7b-instruct
```

(Qwen 2.5 handles both English and Turkish legal text well. Any Ollama
model works — change `OLLAMA_MODEL` in `regintel/config.py`.)

## Usage

```bash
# 1. Download regulation source text (needs internet once)
python -m regintel.ingestion.download DORA

# 2. Parse + index (first run downloads the embedding model, ~500MB)
python scripts/ingest.py

# 3a. Ask from the command line
python scripts/ask.py "What are the deadlines for reporting major ICT incidents?"
python scripts/ask.py --compare DORA NIS2 "incident reporting deadlines"

# 3b. Or launch the chat UI
streamlit run ui/app.py

# 3c. Or run the API
uvicorn regintel.api.main:app --reload   # docs at http://localhost:8000/docs
```

The chat UI's own interface is in Turkish; the LLM answers in whichever
language the question was asked — English questions get English
answers, Turkish questions get Turkish answers.
```

## Adding a regulation

1. Add an entry to `REGULATIONS` in `regintel/config.py`.
2. Drop the source as `data/raw/<ID>.html` (EUR-Lex) or `<ID>.txt`
   (plain text extracted from a PDF — Turkish "Madde" headings are
   supported).
3. Run `python scripts/ingest.py <ID>`.

## Tests

```bash
pytest tests/ -v
```

Tests run offline (no embeddings, no LLM) using a synthetic
mini-regulation, so they verify parsing, chunking, metadata, filtering,
and comparison retrieval anywhere.

## Data confidentiality

Every component — embeddings (sentence-transformers), vector store
(ChromaDB), keyword index (BM25), LLM (Ollama) — runs on localhost.
The only network access needed is the one-time download of regulation
texts and the embedding model, both of which can be done outside the
production environment and copied in.

## Team deployment (bank server + SSO)

For a shared server where your team connects with their own
credentials rather than everyone running it locally:

1. **Ollama stays local to the server.** `OLLAMA_BASE_URL` in
   `regintel/config.py` already points at `localhost:11434` — leave it
   that way. Ollama has no auth of its own, so it must never be
   reachable from the network, only from the Streamlit process on the
   same machine.
2. **Set up login.** Streamlit's built-in auth (`st.login`/`st.logout`,
   used in `ui/app.py`) delegates to your bank's identity provider over
   OIDC — no passwords stored by this app. Copy
   `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and
   fill in the `client_id`/`client_secret`/`server_metadata_url` your
   IT team issues for an internal app registration. Without this file,
   the app runs with **no login at all** — fine for your own laptop,
   not for the shared server.
3. **Put TLS in front of it.** Streamlit itself doesn't terminate
   HTTPS. Run it behind an internal reverse proxy (nginx/Caddy) with
   the bank's internal certificate, and point `redirect_uri` in
   `secrets.toml` at that HTTPS URL.
4. Every question/answer is still logged locally to
   `data/chats/*.jsonl`, now tagged with the authenticated user's
   identity — this is your audit trail of who asked what.
