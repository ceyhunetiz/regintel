# Evaluation

## Purpose and approach

The system is evaluated on **retrieval accuracy**: for a given regulatory
question, does the system retrieve the article that actually contains the
answer? Retrieval is the right primary metric for a source-citing RAG
system because it is objective and reproducible — each test question is
paired with the article number(s) known to contain the answer, and the
metric simply checks whether that article appears among the retrieved
results. Unlike judging free-text answer quality, this needs no human
grader and no LLM-as-judge, so the numbers are stable across runs and
defensible to a reviewer.

Answer quality (whether the generated text is correct, complete, and
properly cited) is assessed separately by human review of generated
answers, because wording correctness cannot be scored automatically.

## Test set

A gold set of 19 questions (`tests/eval_set.json`) spans the three
indexed regulations (DORA, GDPR, KVKK), both languages (English and
Turkish), and three difficulty levels. Each item records the question,
the regulation, the article(s) that contain the answer, and the query
language. The set deliberately includes Turkish questions asked against
English-indexed regulations (DORA) to test the cross-language query
rewriting step.

The set is intentionally small and human-curated rather than large and
auto-generated; every question maps to a specific, verifiable article.
It should be extended with real questions from the compliance team and
reviewed by a domain expert before being treated as authoritative.

## Metrics

- **Hit@k** — the fraction of questions for which a correct article
  appears in the top-*k* retrieved chunks (here *k* = 6). This measures
  whether the answer was retrievable at all.
- **MRR (Mean Reciprocal Rank)** — the average of 1/rank of the first
  correct article. It rewards ranking the right article near the top,
  not merely including it.

Both are broken down by regulation and by language.

## Results

Evaluation at top-*k* = 6:

| Metric | Result |
|---|---|
| Overall Hit@6 | **89.5%** (17/19) |
| MRR | **0.62** |

| Regulation | Hit@6 |
|---|---|
| DORA | 100% (8/8) |
| GDPR | 100% (6/6) |
| KVKK | 60% (3/5) |

| Query language | Hit@6 |
|---|---|
| English | 100% (12/12) |
| Turkish | 71% (5/7) |

## Interpretation

**English-language retrieval is effectively solved on this set** (100%),
and MRR shows the correct article is usually ranked first or second.

**Cross-language retrieval works.** Both Turkish-language DORA questions
retrieved the correct English article (ranks 1 and 3). This validates the
query-rewriting design decision: a Turkish question is translated into a
short English search query before retrieval, so it can match an
English-indexed regulation. Without this step, keyword (BM25) matching
against English text would fail entirely.

**KVKK was the weak point (60%), and the evaluation harness pinpointed a
real, fixable cause.** The two misses were "processing conditions"
(expected Madde 5) and "data subject rights" (expected Madde 11).
Investigation ruled out the initial hypothesis of poor PDF text
extraction — the extracted Turkish text is clean and all 33 articles
parse correctly. The actual cause is structural: in the
mevzuat.gov.tr PDF, each article's heading appears on the line
*before* the "MADDE N-" marker (e.g. the heading
"İlgili kişinin hakları" precedes "MADDE 11-"). The parser began
capturing at the "MADDE N-" marker and therefore discarded these
headings, leaving KVKK chunks without a title. This mattered because the
article heading is often an almost verbatim match for the user's question
("İlgili kişinin hakları" vs. the query "İlgili kişinin hakları
nelerdir?"), making it the single strongest retrieval signal. EU
regulations (DORA, GDPR) were unaffected because their EUR-Lex source
places the title *after* the article number, where it was captured
normally — which is why they scored 100%.

The finding generalizes into a useful conclusion for the report:
**retrieval accuracy is gated by how well document structure is
preserved during ingestion.** Native-HTML sources with well-marked
headings scored 100%; the PDF source whose headings were dropped scored
60%. Correct structural parsing, not model quality, was the limiting
factor.

## Remediation

The plain-text parser was updated to recover the heading from the line
immediately preceding a "MADDE N-" marker. After the fix, all inspected
KVKK articles carry their correct heading (Madde 5 →
"Kişisel verilerin işlenme şartları", Madde 11 →
"İlgili kişinin hakları", etc.). Because these headings are prepended to
each chunk before embedding, the two previously failing questions now
have an exact-match signal in the index.

Re-running the evaluation after re-ingesting KVKK is expected to raise
KVKK Hit@6 toward parity with the EU regulations. (Re-ingest with
`python scripts/ingest.py KVKK`, then `python scripts/evaluate.py`.)

## Answer-behavior regression suites

`scripts/evaluate.py` above measures retrieval only. Two further harnesses
cover answer-level behaviour:

- **`scripts/eval_regression.py`** (`tests/regression_set.json`) — citation
  integrity, jurisdiction/instrument confinement, and run-to-run stability,
  run `--runs` times per case.
- **`scripts/eval.py`** (`tests/eval_cases.yaml`) — the 20-case adversarial
  set from `Reponses/Questions_for_Rag.pdf` (false premises, cross-instrument
  contamination traps, citation-integrity stress, Board-decision corpus-scope
  probes), analyzed in `Reponses/RegIntel_Evaluation_Report_2026-08-20.md`.
  Retrieval-side assertions (`must_retrieve_articles`) run against
  `RagPipeline._ask_prompt`/`_compare_prompt` directly, so they exercise
  jurisdiction routing and scenario decomposition without needing a model
  (`--llm echo`); answer-text assertions need a live Ollama model. Dumps a
  full per-case report to `data/eval_runs/<timestamp>.md`.

## Limitations and future work

- The gold set is small (19 items) and author-curated; it should be
  expanded with real compliance-team questions and expert-reviewed.
- Answer-quality scoring is currently manual. A structured rubric
  (factual correctness, completeness, citation validity, appropriate
  refusal) applied to the `--answers` output would make it repeatable.
- A retrieval re-ranking stage (cross-encoder) is the highest-value next
  improvement for questions where the correct article is retrieved but
  ranked low (e.g. GDPR Art. 32 at rank 6).
- Embedding-model input truncation (~128 tokens) means only the start of
  each chunk is embedded; a longer-context embedding model (e.g. BGE-M3)
  is a documented future upgrade.
