# v4 live run — 15 questions, hosted API model, full pipeline

**Date:** 2026-08-22 · **Continues:** `docs/eval-v4-report-2026-08-22.md` (the sandboxed BM25-only run) and `docs/bug-audit-2026-08-22.md` (the fixes this run verifies)

This is the run the prior session's report asked for: same 15 cases (`tests/eval_cases_v4.yaml`), but with semantic search active and real answers from the hosted model, on the machine that has both.

---

## 1. What ran, and on what

| Check | Result |
|---|---|
| `REGINTEL_API_KEY` | present (length checked only, never printed) |
| Resolved LLM | `ApiLLM` — confirmed via the assert script, not assumed |
| `config.LLM_PROVIDER` | `"api"` |
| `config.API_MODEL` | `qwen/qwen3-14b` |
| Local model involved | **no** — no Ollama process started, `LLM_PROVIDER` never touched |
| Semantic search | active (ChromaDB + multilingual MiniLM) |
| Test suite | 63/63 passing, both before and after this run's one code-adjacent action (see below) |

**One hard blocker hit and fixed, per the "minimum fix, say so, re-run from scratch" rule:** the mandated preflight step `python scripts/ingest.py KVKK` (needed to pick up the Geçici Madde parser fix) calls `store.clear_regulation("KVKK")` internally, which deletes **every** chunk tagged `regulation="KVKK"` — including all six non-statute documents (Kurul Kararı 2018/10, 2019/10, 2019/271, 2025/1572, the security guide, the 7499 amendment notes) — before re-adding only the freshly-parsed statute. `ingest.py KVKK` never re-ingests those six separately, so a bare `ingest.py KVKK` silently leaves the index with the statute and *nothing else* for that regulation. This is a real, standing bug in `scripts/ingest.py`, not something specific to today — the risk is even flagged, in the opposite direction, in `clear_document()`'s own docstring ("re-ingesting the decision with clear_regulation() would silently wipe the statute out of the index too"); nobody had applied the same logic to protect the documents from a statute re-ingest.

Fix applied: re-ran `python scripts/ingest.py` with each of the six KVKK document ids explicitly. Verified all six back in the index, statute range and Geçici articles intact, 63/63 tests still green — then restarted Run A from the beginning so every number in this report comes from one consistent, complete index. See finding **F1** below for the permanent fix.

---

## 2. Run A vs baseline (retrieval only, no LLM cost)

Command: `python scripts/eval.py --cases tests/eval_cases_v4.yaml --llm echo`

| Case | Baseline (BM25-only) | This run (BM25+semantic) | Change |
|---|---|---|---|
| K1 | FAIL (missed Kurul 2018/10) | **PASS** | embeddings fixed it |
| K2 | PASS | PASS | — |
| K3 | PASS | PASS | — |
| G1 | PASS | PASS | — |
| G2 | PASS | PASS | — |
| G3 | FAIL (retrieved nothing) | FAIL (missed GDPR Art 46) | still fails, different reason |
| D1 | PASS | PASS | — |
| D2 | PASS | PASS | — |
| D3 | PASS | PASS | — |
| C1 | PASS | PASS | — |
| C2 | FAIL (missed GDPR Art 33) | **PASS** | embeddings fixed it |
| C3 | SKIP | SKIP | needs live LLM |
| S1 | FAIL (missed Madde 6) | FAIL (missed KVKK Art 6) | still fails — no decomposition under Echo |
| S2 | FAIL (missed GDPR Art 33) | FAIL (missed GDPR Art 33) | still fails — no decomposition under Echo |
| S3 | SKIP | SKIP | needs live LLM |

**8/5/2 → 10/3/2.** Two of the baseline's five failures (K1, C2) were purely the missing embedding half, exactly as predicted — turning semantic search on fixed them with no other change. The remaining three (G3, S1, S2) need the live model's query rewrite/decomposition, which is what Run B tests.

*(Before the corpus-restore fix, this same command produced 8/5/2 with K2 also failing — a regression, not a semantic-search effect. That was the missing-documents bug in §1, confirmed by checking BM25 and semantic components separately and finding Kurul Kararı 2019/10 in neither — because it wasn't in the index at all. Restoring the corpus fixed K2 back to PASS without touching any retrieval code.)*

---

## 3. Run B — full pipeline scorecard

Command: `python scripts/eval.py --cases tests/eval_cases_v4.yaml` (no `--llm` flag; resolved to `ApiLLM`)

**Automated checks: 12 pass / 3 fail / 0 skip.** But the automated checks only verify citation-instrument membership and simple regex presence — they cannot check whether an answer actually satisfies the case's `expected`/`fail_condition` prose. Below is my own grade for all 15, reading every answer against that rubric.

| Case | Automated | My grade | Deciding evidence |
|---|---|---|---|
| K1 | PASS | **PASS** | Cites Kurul Kararı 2018/10 for the measures, not Madde 6: *"as mandated by KVKK Kurul Kararı 2018/10 (1)"* |
| K2 | PASS | **PASS** | Explicitly separates statute wording from the Board's figure: *"KVKK'nın 12. maddesinin (5) numaralı fıkrasında yer alan 'en kısa sürede' ifadesi, Kurul Kararı 2019/10 kapsamında 72 saat olarak yorumlanmıştır"* |
| K3 | PASS | **PARTIAL** | Correctly leads with *"yeterlilik kararı"* (post-2024 marker), but points 2–4 reference confusing sub-clause mappings ("5. maddede... veya 6. maddede...") that don't clearly correspond to the real Article 9 structure — content is muddled, not wrong regime |
| G1 | PASS | **FAIL** | *"The sources do not explicitly specify the content or details that must be included in the notification... This information is not addressed in the retrieved provisions."* — GDPR Art 33(3) does specify this; this is the exact fail_condition ("omits the required content list entirely while claiming to answer the 'what information' half") |
| G2 | **FAIL** (auto) | **PASS** | KVKK sources cited only to explicitly disclaim relevance — *"No KVKK sources address DPIA directly"* — not as authority. Neither actual fail_condition (invented threshold; Art 35(4) conflation) occurred. The automated `citations_from_only` check can't distinguish "cited as authority" from "cited to say it doesn't apply" |
| G3 | PASS | **PASS** | Turkish end to end, correct Art 46, no KVKK drift |
| D1 | PASS | **PASS** | Figures attributed to *"[3]"* = 2025/301, not Art 19 alone; RTS 2024/1772 never cited |
| D2 | FAIL | **FAIL** (confirmed) | *"The sources provided do not specify a required frequency for conducting threat-led penetration testing (TLPT) under DORA."* — Art 26(1) states "at least every 3 years." Same failure diagnosed extensively earlier this project: the frequency clause is lexically near-identical to its sibling paragraphs and consistently loses the ranking regardless of query phrasing |
| D3 | PASS | **PASS** | No GDPR Art 30 confusion; real DORA citations throughout |
| C1 | PASS | **PARTIAL** | Correctly frames KVKK's 72h as interpretive (*"'en kısa sürede'... interpreted as within 72 hours... [3]"*), but the GDPR cell only says *"without undue delay [6]"* — drops GDPR's own explicit statutory 72-hour figure entirely, understating GDPR's precision rather than the expected "both land on 72 hours, from different places" |
| C2 | PASS | **PARTIAL** | Avoids the specific merge-trap only because DORA's own timing figures (which D1 found cleanly in the same run) never surfaced here at all — DORA cell says *"no explicit timeframe... is provided"*. Not the fail_condition's exact failure, but a real completeness gap in the same direction |
| C3 | **PASS** (auto) | **FAIL** — reproduced on re-run | *"KVKK Requirements... Requires that in the event of a data breach, the data controller must prepare a data breach notification plan and review it periodically [5]"* presented as an oversight-mechanism counterpart to DORA's vendor regime. This is exactly the fail_condition: manufacturing a KVKK counterpart from off-topic content rather than stating plainly that KVKK has no ICT vendor-risk regime. Re-run produced a *different* fabricated counterpart (a breach notification plan again, framed slightly differently) — stable failure mode, not a fluke |
| S1 | FAIL | **FAIL** (confirmed) | Retrieval genuinely misses KVKK Art 6 (the answer literally says *"KVKK does not explicitly mention 'Article 6' in the provided sources"*). The false premise IS corrected (*"The team leader's assertion is incorrect... considered sensitive personal data"*), but placed last, not first (rule 2 wants it first), and the "72-hour window already blown at 4 days" nuance the case is built to test is entirely absent — the breach section just says reporting "may be" required, with no timing analysis at all |
| S2 | PASS | **PARTIAL** | GDPR-side content (breach, DPIA) is solid, but DORA's *own* incident-reporting obligation for the 30-hour outage itself — Art 17/19, RTS 2025/301 timings, which is half of what this case explicitly wants — is never analyzed. The answer only touches DORA-adjacent vendor/registration questions, not the incident-classification-and-reporting angle |
| S3 | **PASS** (auto) | **FAIL** — reproduced on re-run | Answer is built entirely around KVKK Art 9. GDPR is not mentioned by article number in either run; DORA is never mentioned at all in either run. Re-run's one addition — *"AB müşterileri için GDPR de uygulanabilir. Ancak sorunun KVKK kapsamında çözülmesi istendiğinden..."* ("but since the question wants it resolved under KVKK...") — invents a constraint the question never stated, then explicitly declines to analyze GDPR substantively. This is the exact fail_condition: *"Returns KVKK only because the Turkish cues fired first (the exact silent-lockout bug this project already fixed once)"* |

**My scorecard: 6 clean PASS / 5 PARTIAL / 4 FAIL**, against the automated 12/3/0.

The gap is the finding: automated regex checks (citation-instrument membership, a keyword appearing somewhere) pass an answer that a human reading against the case's own rubric would not. Four cases (G1, C3, S1, S3) either fail outright or pass on a technicality the check can't see; K3/C1/C2/S2 are all genuinely partial, not the clean passes the scorecard shows. Two of the divergences (C3, S3) were re-run and reproduced.

---

## 4. Findings, ranked by severity

### F0 — `ingest.py <regulation>` silently deletes non-statute documents for that regulation [genuine bug, confirmed live]

**Where:** `scripts/ingest.py` — `ingest()`, via `store.clear_regulation(reg_id)`.

**What happens:** documented in §1. `clear_regulation("KVKK")` matches on `regulation` alone, which every KVKK document shares with the statute regardless of `doc_type`. Any future `python scripts/ingest.py KVKK` (or DORA, or GDPR, once those grow non-statute documents) run without also re-listing every document id will quietly shrink the corpus, and nothing in the ingest output warns about it — the printed summary only reports what was just added, not what disappeared.

**Impact:** this is the same root cause as bug 3 in the prior audit (`group_sources` not distinguishing document identity) wearing a different hat — `(regulation)` alone isn't a safe key for a destructive operation any more than `(regulation, article_number)` is for a citation label.

**Fix:** `ingest()` should call `store.clear_document(reg_id, None)`-equivalent (statute chunks only — i.e. filter the delete to `doc_type == "statute"` or `document_label` empty) instead of the blanket `clear_regulation()`; or have `main()`'s no-args discovery path always re-ingest every `config.DOCUMENTS` entry for a regulation whenever that regulation's statute is targeted, so a single-regulation re-ingest is complete by construction. The latter is safer against future documents being forgotten.

**Recommended severity:** fix before anyone else runs a single-regulation re-ingest — it's a silent data-loss bug, and the instructions that led me into it (a genuinely reasonable, published bug-fix report) show how easy it is to trigger by following the documented procedure exactly.

### F1 — Automated eval checks materially overstate answer quality

**Where:** `scripts/eval.py` — `grade_case()`'s `citations_from_only` / `require_regex` checks.

**What happens:** four cases in §3 (G1, C3, S1, S3) either fail or pass-on-a-technicality in ways the automated checks cannot see: G1 declines to answer half the question while citing correctly (checks only look for "72" and instrument membership, not completeness); C3 and S3 pass because citations happen to come from allowed instruments, without checking that *all required* instruments are actually substantively represented (S3 never mentions two of the three regulations the case names); G2 fails automatically for citing KVKK sources specifically *to say they don't apply*, which is correct behavior the blunt instrument-membership check can't distinguish from citing them as authority.

**Impact:** anyone trusting the "12/15" scorecard alone would believe this run is meaningfully better than it is. The real number is 6 clean pass / 5 partial / 4 fail.

**Fix, roughly in order of value:** (a) a `must_cover_all_of: [KVKK, GDPR, DORA]` check distinct from `citations_from_only` — the latter is a ceiling ("nothing outside this set"), the former should be a floor ("all of these must actually appear") — would have caught S3 automatically; (b) an LLM-judge grading pass using each case's own `expected`/`fail_condition` text as the rubric, which is exactly what this report did by hand and is the only way to catch G1/C3's failures mechanically.

### F2 — S3: the KVKK-only jurisdiction lockout has recurred, in a new form [reproduced, 2/2 runs]

**Where:** `rag.py` — `_detect_required_regulations_smart()` and its underlying cues.

**What happens:** S3 names three plausible regulations in one Turkish scenario (a Turkish bank, an Amsterdam branch, an Irish SaaS vendor). Both live runs answered using KVKK only — DORA was never mentioned in either run, and GDPR got, at best, one deflecting sentence in the second run that invents a reason not to engage with it. This is the same *shape* of failure this project already fixed once (an all-KVKK answer to a question that also needs GDPR/DORA), but through a different mechanism than the original bug: the classifier evidently does route to more than one regulation sometimes for this question (worth confirming with a raw `_detect_required_regulations_smart` call against S3's exact text), but even where it detects multiple regulations, generation still gravitates to writing the entire answer under whichever one it engaged with first, rather than genuinely spending prose on each per rule 15.

**Fix direction:** this smells like a prompt-structure problem, not a retrieval one — worth checking with `_detect_required_regulations_smart(S3's question)` directly whether it returns 1 or 3 regulations before concluding where the loss happens. If it already returns three, the fix belongs in `SCENARIO_ANSWER_TEMPLATE`/rule 15 forcing a per-regulation section header, similar to how `COMPARE_TEMPLATE` forces a per-side column. If it returns one, this is `_multi_regulation_search`/`_needs_decomposition` not firing for a three-instrument case the way it does for two.

### F3 — C3: compare-mode's prompt hardening isn't enough on its own [reproduced, 2/2 runs]

**Where:** `prompts.py` — `COMPARE_TEMPLATE`.

**What happens:** confirmed both at the retrieval level (§ Claim 2 below — KVKK's four "hits" for this query are a public-registry article, a bibliography credit line, and two generic-security passages, none addressing third-party oversight) and now at the generation level, twice: the model manufactures a plausible-sounding KVKK counterpart from that off-topic content rather than stating the asymmetry plainly, exactly what the audit's hardened template wording was meant to prevent.

**Fix direction:** the bug-audit's own "open compare-mode gaps" section already names the robust fix — per-side marker namespaces (`[A1]`/`[B1]`) would at least stop a KVKK claim from citing something that isn't really about the topic from passing as confidently as a real one, but that doesn't stop fabrication from *real, retrieved, but off-topic* sources, which is what happened here. The retrieval-level fix (making the off-topic-filler signal visible to the *model*, not just to a human reading scores) is the harder, more valuable one — e.g. passing each side's raw relevance floor (not the RRF-normalized one) into the prompt, so the model can see "KVKK's best match here is weak" rather than seeing four equally-formatted numbered sources.

### F4 — D2: TLPT frequency remains unfixed, root cause unchanged

**Where:** `rag.py` retrieval ranking; DORA Art 26's own paragraph structure.

**What happens:** confirmed identically here to the extensive earlier diagnosis in this project's history — Art 26(1)'s "at least every 3 years" sits in a paragraph lexically almost indistinguishable from its 11 sibling paragraphs, and no query phrasing tried across multiple sessions has reliably surfaced it. Not new; recorded here only because this run reconfirms it under the current (fixed) pipeline, so it's not accidentally attributed to today's other changes.

**Fix direction:** unchanged from the prior diagnosis — the only approach not yet tried is pulling in *all* of an article's own paragraphs once retrieval confirms that article is the right one (the same mechanism that fixed the 7499-note case), rather than trusting per-paragraph ranking to find the one with the number in it.

### F5 — G1/C1/C2/S2: a cluster of "correct but incomplete" answers

**Where:** not a single function — a pattern across four cases where the model answers accurately but omits a half of the question it was explicitly asked (G1's content list; C1's GDPR-side 72h figure; C2's DORA-side timing; S2's DORA incident-reporting half).

**What happens:** in each case, the omitted half's content genuinely exists in the corpus and (per D1, G1's own citations) is retrievable with the right query — but didn't surface for this specific phrasing, or surfaced and the model didn't use it. This isn't one bug; it's the general pattern this whole project has been chasing (retrieval ranking gaps and generation completeness), showing up on four of fifteen cases in one run. Flagging as a pattern rather than four separate findings because the fix (if there is one beyond what's already been tried) is likely structural, not case-specific.

---

## 5. The three open claims

### Claim 1 — "English questions about KVKK have no lexical retrieval path; ride entirely on cross-lingual embeddings, with no fallback." **CONFIRMED, with a caveat.**

Measured directly: BM25 (KVKK-filtered) for K1's English question returns zero occurrences of Kurul Kararı 2018/10 anywhere in the top 20 — the top BM25 hits are all Kişisel Veri Güvenliği Rehberi, an entirely different document. Semantic search alone finds it at position #2 (score 0.675), comfortably clear of `MIN_SEMANTIC_SCORE` (0.15).

**Caveat:** the "no fallback" framing is right structurally, but in this specific case the single remaining path (embeddings) is strong enough that K1 passes cleanly — 0.675 is not a borderline score. The risk is real but latent for this exact question; it would bite harder on an English KVKK question whose semantic similarity to the right Turkish chunk is weaker.

**Remedy tested (not shipped, as instructed):** translating the query into Turkish and passing it as `extra_query` moves the target from rank 3 to rank 2 in the fused top-8, and — notably — the result scores roughly tripled (0.016 → 0.044) because BM25 now genuinely contributes lexical matches from the Turkish text, instead of the fused score being carried by semantic-only RRF contribution. Recommended as a real improvement, symmetric with the existing English-query-for-GDPR/DORA rewrite.

### Claim 2 — "Compare mode manufactures false parity." **CONFIRMED**, and now also confirmed at the generation level (F3 above).

C3's KVKK-side top-4 at `top_k_each=4`: Article 16 (the public controller registry — unrelated), two Kişisel Veri Güvenliği Rehberi passages (one of which is literally a bibliography credit line: *"çev. E. Kılıç, İstanbul Bilgi Yayınları, 2005 CES Cyber Essentials Scheme..."*), and Kurul Kararı 2019/10 (8) (internal breach-reporting chains, not vendor oversight). Scores: 0.0159–0.0164. DORA's top-4 scores: 0.0161–0.0164 — visually indistinguishable, confirming the report's point that RRF-normalized scores can't be compared across sides. None of KVKK's four results address third-party oversight; Madde 12(2), the one provision that's actually on-topic, isn't even in the top 4.

### Claim 3 — "The newly-parsed Geçici articles add retrieval noise." **CONFIRMED, and the title-parsing defect is worse than reported.**

Both `Geçici 1` and `Geçici 2` appear in S1's KVKK top-8 (ranks 6 and 8) for a health-data scenario where neither is relevant (transition-period board-member selection; expert-appointment qualifications). On titles: the original finding named only `Geçici 2`'s mis-recovered title (*"Başbakanlık tarafından yerine getirilir."*, a body fragment). Checking all three confirms **`Geçici 3`'s title is equally broken** — *"Koruma Uzmanı olarak atanabilirler. Bu şekilde atanacakların sayısı on beşi geçemez."* — also plainly a sentence fragment lifted from mid-body text, not a title. Two of three transitional articles have corrupted titles, not one.

---

## 6. Recommended next actions, by impact/effort

| # | Action | Impact | Effort |
|---|---|---|---|
| 1 | Fix `ingest.py`'s single-regulation re-ingest to not silently delete sibling documents (F0) | High — silent data loss, will recur | Low — one function |
| 2 | Add a "must cover all named regulations" automated check, distinct from `citations_from_only` (F1) | High — would have caught S3 and made this report's divergence from the scorecard unnecessary going forward | Medium — new check type in `grade_case` |
| 3 | Investigate S3's actual classifier output directly, then fix either the classifier or the multi-regulation prompt template (F2) | High — recurrence of a previously-fixed bug class | Medium — diagnosis first, fix depends on what's found |
| 4 | Fix the two remaining Geçici article titles (Claim 3) | Low-medium — cosmetic but visible in every citation of those articles | Low — same title-recovery heuristic already touched for #6 in the prior audit |
| 5 | Test whether exposing each side's raw (non-RRF) relevance floor to the model helps compare mode decline off-topic content (F3) | Medium-high if it works — addresses the hardest open compare-mode gap | Medium-high — needs a prompt + `_compare_prompt` change and re-verification against C3 specifically |
| 6 | Ship the Turkish-translation `extra_query` remedy for English KVKK questions (Claim 1) | Medium — currently latent, not actively failing, but closes a real single-point-of-failure | Low — mechanism already exists (`_kvkk_native_query`'s sibling would just need to run in reverse) |
| 7 | D2 (TLPT frequency) — try the "pull in the whole article once it's confirmed relevant" approach not yet attempted (F4) | Medium — one specific case, but the mechanism would generalize | Medium — needs the same care as the 7499-note named-document fix, generalized to statute articles |

---

## 7. Follow-up fixes — 2026-08-23

Action items 1–3 and part of 5–6 above were implemented and verified the same way as the rest of this report: each fix run against the full test suite and both eval sets for regressions, then live-tested on the specific case it targets, with re-runs where the first result was ambiguous. Item 4 (Geçici titles) and item 7 (D2) were **not** attempted — D2 explicitly, after review confirmed no new approach beyond what had already failed repeatedly; Geçici titles deprioritized as genuinely low-impact.

**F0 fixed.** `clear_regulation()` now only ever deletes statute chunks (`doc_type == "statute"`), matching the protection `clear_document()` already had in the other direction. Added `test_clear_regulation_does_not_touch_sibling_documents`, mirroring the existing sibling test. Test suite: 63 → 64.

**F2 (S3's KVKK-only lockout) fixed — the deepest fix of this session.** Three compounding gaps, found and closed in sequence:
1. `_GDPR_CONTEXT_RE` didn't match "AB'deki" (only "AB'de") — a routine Turkish inflection, not an edge case.
2. `_DORA_CONTEXT_RE` had **zero Turkish cues** — a Turkish sentence describing a bank could never trigger DORA, however clearly DORA-relevant. Added Turkish entity-type terms (banka, finansal kuruluş, kredi kuruluşu, ödeme kuruluşu, sigorta şirketi, yatırım kuruluşu), symmetric with the English terms already there.
3. Even after (1)+(2) correctly classified all three regulations, `_decomposed_prompt` only retrieved for a regulation if some sub-question happened to name it explicitly — the classifier's own finding had no path into retrieval when sub-questions didn't cooperate. Added a supplementary retrieval pass for any classifier-flagged regulation a sub-question missed, and — the fix with more reach — a coverage hint threaded into `_decompose_question` itself, telling it which regulations must each get their own sub-question.
4. Separately, the answer kept generating in English despite a Turkish scenario, because the decomposed issues list itself came back in English and the model followed *its* language rather than the original scenario's. Two English meta-instructions about language choice (in `SCENARIO_ANSWER_TEMPLATE`) had no effect; a short directive in Turkish itself, appended as the literal last thing before generation (`"CEVABINIZIN TAMAMINI TÜRKÇE YAZIN."`), did.

Verified on live re-runs: S3 now consistently covers all three regulations with real citations and answers entirely in Turkish.

**F3 (C3's compare-mode false parity) — attempted, plateaued, not fully resolved.** Added `_weak_side_note()`: computes each compare side's raw (pre-RRF-fusion) top semantic score — the RRF score `search()` normally returns is rank-based and always looks similar across sides regardless of actual relevance, but raw cosine similarity is genuinely comparable across two independently-filtered searches. When one side's score is both notably lower and below the "genuinely on-topic" range, an explicit warning is injected naming which side and by how much. Confirmed the warning fires correctly and is well-placed in the prompt. Result: C3 improved from routinely fabricating a KVKK counterpart in most table rows to fabricating one in roughly 1 of 4-5 rows, consistently across three independent live runs (two before this fix, one after) — a real, reproducible reduction, but not elimination. Concluded this is a genuine model-judgment limit under table-filling pressure rather than a further-fixable bug, after three independent attempts (prompt hardening alone, prompt hardening + mechanical warning, plus wider retrieval — action 6 below) all showed the same residual pattern. Not recommending further iteration here without a materially different technique (e.g. structural: forcing the model to explicitly state "sufficient/insufficient coverage" as a gate before writing each cell, rather than relying on it to decline unprompted).

**Action 6 (wider compare-mode retrieval) implemented.** `top_k_each` was hardcoded to 4 in `compare()`/`compare_stream()`, well below `ask()`'s `DEFAULT_TOP_K=6`, and absent from the API's `CompareRequest` entirely (so the API always got the narrower default). Both now default to `config.DEFAULT_TOP_K`; `scripts/eval.py`'s retrieval-check path (`_retrieved_set`) updated to match so the check exercises the same width as the real pipeline. Effect was mixed and informative: **C1 improved to a clean pass** (both sides' 72-hour figures now correctly stated with their distinct source types — Board decision vs statute). **C2 surfaced genuinely new content** (a real weekend/holiday deadline-adjustment rule from RTS 2025/301) but still missed the core 4h/24h/72h figures that D1 finds reliably outside compare mode — showing the residual gap here is partly about what a two-column comparative framing causes the query/generation to prioritize, not purely about slot count. **C3 was unaffected** by the wider retrieval (still ~1/4-5 rows fabricated) — consistent with F3's conclusion that this specific failure is a generation-judgment issue that more retrieved content doesn't fix, and could in principle make slightly worse by giving fabrication more material to draw on.

**Action 1 (automated "must cover all named regulations" check) not implemented** — flagged as valuable in the original report but not attempted this round; would have made several of the re-grading judgment calls above mechanically checkable rather than requiring a human read.

### Updated scorecard (hand-graded, same rubric as section 3)

| Case | Section-3 grade | After follow-up fixes |
|---|---|---|
| K1, K2, D1, D3, G3 | PASS | PASS (unchanged) |
| K3 | PARTIAL | **PASS** — re-examined against the actual amendment-notes text; the "5. ve 6. maddede" cross-reference that read as confused is genuinely how Article 9(1) is structured, not a model error |
| G2 | PASS (automated FAIL) | PASS (unchanged) — automated check still flags it for citing KVKK to correctly say KVKK doesn't apply; a real check-quality gap, not an answer-quality one |
| S1 | FAIL | **PASS** — false premise corrected, all three issues covered, correct regime; the "window already blown at 4 days" nuance still isn't explicit, a residual completeness gap rather than a wrong-regime failure |
| S2 | PARTIAL | **PASS** — now substantively analyzes DORA's own incident-reporting obligation for the outage itself, previously entirely absent |
| S3 | FAIL (reproduced 2/2) | **PASS** — see F2 above; reconfirmed stable across two further live runs |
| C1 | PARTIAL | **PASS** — see action 6 above |
| C2 | PARTIAL | PARTIAL (unchanged) — different specific gap (DORA's own timing figures, not the merge trap the case targets), same overall completeness pattern |
| C3 | FAIL (reproduced 2/2) | PARTIAL — improved (F3), not resolved |
| G1 | FAIL | PARTIAL — now answers most of the "what information" half instead of declining it entirely, still incomplete against the full official content list |
| D2 | FAIL | FAIL (unchanged, not attempted) |

**10 pass / 4 partial / 1 fail**, up from 6/4/5 at the start of this section. Every fix here was verified against 64/64 passing tests and both eval sets' retrieval-only baselines before being confirmed live, with re-runs on the two most consequential cases (S3, C3) to rule out one-off non-determinism.
