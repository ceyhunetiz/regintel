# Presentation Brief — RegIntel (paste into Gemini)

You are creating a slide deck about a software project. Audience: a bank's
cybersecurity governance and compliance team (non-academic, business-minded
but technically literate). Keep it balanced — explain the approach clearly
without heavy jargon. Sober, professional tone. No emojis. ~12–14 slides.
Use only the facts below; do not invent metrics, article numbers, or features.

## What the project is

RegIntel — an internal "Regulatory Intelligence Assistant." Compliance staff
ask questions about regulations in plain language (English or Turkish) and get
answers grounded in the actual regulation text, each with a citation to the
specific article it came from. It runs entirely on local infrastructure —
no data is sent to any external AI service, which is a hard requirement for a
bank handling confidential governance documents.

## The problem it solves

Regulatory research today is manual and slow: staff read long PDFs
(DORA, GDPR, KVKK, BDDK) to find the one clause that answers a question, and
different reviewers can interpret the same text differently. RegIntel makes
this a natural-language search that returns the exact source text, faster and
more consistently.

## How it works (explain simply)

It uses Retrieval-Augmented Generation (RAG). Instead of trusting an AI to
"remember" regulations (which causes made-up answers), the system:
1. Splits each regulation into article-sized pieces and indexes them.
2. For a question, retrieves the most relevant pieces (hybrid search:
   meaning-based + keyword-based).
3. Feeds only those pieces to a local language model, which must answer
   using only that text and cite each source.
Every claim in an answer links to an expandable panel showing the exact
regulation article it came from — so answers are verifiable, not trusted
blindly.

## Key design choices worth a slide each

- Article-aware chunking: pieces never cross article boundaries, so citations
  point to a real article.
- Hybrid retrieval: keyword search catches exact terms ("Article 5",
  "madde 12"); meaning-based search catches concept questions; results are
  combined.
- Cross-language support: a Turkish question is rewritten into an English
  search query internally, so it can find answers in English-indexed
  regulations (DORA, GDPR). Answers come back in the user's language.
- Comparison mode: can compare how two regulations treat the same topic
  (e.g. DORA vs GDPR on incident/breach notification), retrieving from each
  separately so both are fairly represented.
- Fully local: local language model (Qwen3 8B via Ollama), local vector
  database, local keyword index. Nothing leaves the machine.

## Technology (one slide, plain list)

Python backend (FastAPI), Streamlit web interface, ChromaDB vector database,
BM25 keyword search, multilingual sentence-embeddings, local LLM (Qwen3 8B)
served by Ollama. Regulations currently indexed: DORA and GDPR (from EUR-Lex),
KVKK (from mevzuat.gov.tr); BDDK planned.

## Evaluation results (use these exact numbers)

Tested on 19 curated questions across DORA, GDPR, KVKK, in English and Turkish.
Metric: "Hit@6" = did the system retrieve the correct article among its top 6
results.

- Overall retrieval accuracy: 89.5% (17 of 19)
- DORA: 100%   |   GDPR: 100%   |   KVKK: 60%
- English questions: 100%   |   Turkish questions: 71%
- Turkish questions about English regulations still succeeded — validates the
  cross-language design.

The evaluation also caught a real bug: KVKK scored lower because article
headings in the Turkish PDF were being dropped during processing. After fixing
the parser to keep those headings, KVKK retrieval is expected to rise toward
the level of the EU regulations. Takeaway: accuracy depends on how well
document structure is preserved during ingestion — not on the AI model.

## Value to the bank (closing slides)

- Faster regulatory research; less time reading PDFs.
- More consistent interpretations (every answer tied to source text).
- Verifiable, audit-friendly answers (each claim links to its article).
- No external data sharing — meets confidentiality/compliance requirements.
- Extensible: adding a regulation is a repeatable ingestion step.

## Suggested slide order

1. Title
2. The problem (manual, slow, inconsistent regulatory research)
3. What RegIntel is (one-sentence solution + screenshot placeholder)
4. How RAG works, simply (the 3 steps)
5. Why answers are trustworthy (citations / verifiability)
6. Article-aware chunking + hybrid retrieval
7. Cross-language support
8. Comparison mode
9. Fully local / data confidentiality
10. Technology stack
11. Evaluation results (the numbers table)
12. What the evaluation revealed (the KVKK finding)
13. Value to the bank
14. Roadmap / future work (more regulations, re-ranking, enterprise deployment)

Leave placeholders where a screenshot of the app should go.
