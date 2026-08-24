# v4 test run — 15 questions across KVKK, GDPR, DORA

**Date:** 2026-08-22 · **Corpus:** your real `data/raw/` (25 files, rebuilt from source) · **Cases:** `tests/eval_cases_v4.yaml`

## What I could and couldn't run — read this first

I rebuilt your index from the real corpus in this sandbox and ran all 15 questions through the actual pipeline. Two limits on what that proves:

| Layer | Ran? | Why |
|---|---|---|
| Parsing + chunking + indexing | ✅ full | rebuilt from your `data/raw/` |
| **BM25 (keyword) retrieval** | ✅ full | pure Python |
| **Semantic (embedding) retrieval** | ❌ **off** | `huggingface.co` is outside this sandbox's egress allowlist, so the multilingual embedding model can't be downloaded here |
| LLM query rewrite + decomposition | ❌ off | ran under `EchoLLM` |
| **Generated answers / reasoning** | ❌ **not run** | needs your OpenRouter key |

So this run tests **retrieval routing and the keyword half of hybrid search**, not the semantic half and not reasoning. Your machine has both (the model is cached in `~/.cache/huggingface`, and your key is in the environment) — the command to run the full thing is at the bottom, and it takes about a minute.

I've flagged every result below as **real finding** or **sandbox artifact** so you don't chase the wrong ones.

---

## The question set

Three per regulation, three comparisons, three scenarios. Design choices worth knowing:

- **Language is deliberately crossed.** KVKK is asked once in English (K1) and GDPR once in Turkish (G3), because a same-language-only set never exercises cross-lingual retrieval or the cross-lingual citation-grounding path — which is where your original "wrong sources" bug lived.
- **Every regulation gets one question whose real answer is outside its base statute** (K1→Kurul 2018/10, K2→Kurul 2019/10, D1→RTS 2025/301). A statute-only paraphrase gets these silently wrong, which is the failure your corpus expansion was meant to fix.
- **C3 (DORA vs KVKK) is deliberately lopsided.** KVKK has essentially nothing on ICT vendor risk. The correct answer says so; the tempting wrong answer manufactures a counterpart.
- **S1/S2/S3 embed false or unstated premises** (a team lead insisting diagnosis notes "aren't health data"; an outage that is simultaneously an ICT incident and a data breach; a bank whose data sits under two jurisdictions at once).

Full text of all 15, with expected answers and fail conditions, is in `tests/eval_cases_v4.yaml`.

---

## Results (retrieval only, BM25-only)

| Case | Topic | Routing detected | Retrieval | Verdict |
|---|---|---|---|---|
| K1 | KVKK special-category measures *(asked in EN)* | KVKK ✅ | ❌ missed Kurul 2018/10 | **sandbox artifact** — no lexical path exists (see below) |
| K2 | KVKK breach deadline to the Board | KVKK ✅ | ✅ found Kurul 2019/10 | pass |
| K3 | KVKK cross-border transfer | KVKK ✅ | ✅ found Madde 9 | pass |
| G1 | GDPR breach notification content + deadline | — | ✅ found Art 33 | pass |
| G2 | GDPR DPIA triggers | — | ✅ found Art 35 | pass |
| G3 | GDPR transfer safeguards *(asked in TR)* | — | ❌ retrieved **nothing** | **sandbox artifact**, but see finding 3 |
| D1 | DORA incident reporting deadlines | — | ✅ found RTS 2025/301 | pass |
| D2 | DORA TLPT frequency | DORA ✅ | ✅ found Art 26 | pass |
| D3 | DORA register of information | DORA ✅ | ✅ found Art 28 | pass |
| C1 | KVKK ↔ GDPR breach notification | — | ✅ | pass |
| C2 | DORA ↔ GDPR incident timelines | — | ❌ missed GDPR Art 33 | mixed — see finding 4 |
| C3 | DORA ↔ KVKK third-party oversight | — | (no retrieval assertion) | needs live LLM |
| S1 | Istanbul health-tech scenario | KVKK ✅ | ❌ missed Madde 6 | artifact (no decomposition under Echo) |
| S2 | EU payment institution outage | DORA ✅ *(was: none)* | ❌ missed GDPR Art 33 | mixed — see finding 2 |
| S3 | Turkish bank with EU branch | KVKK ✅ | (no retrieval assertion) | needs live LLM |

**8 pass / 5 fail / 2 need the live model**, before accounting for the missing semantic half.

---

## Real findings (fixed)

### 1. English questions about Turkish law routed to GDPR — **fixed**

`_KVKK_CONTEXT_RE` matched `türkiye` but not the English words **"Turkey"** or **"Turkish"**. An English question that literally says *"under Turkish data protection law"* hit no KVKK cue, retrieval ran unfiltered, and — measured — it returned **8 GDPR articles and zero KVKK**. The English-language, higher-volume half of the corpus wins every unfiltered ranking, which is the same F4-class failure your project already fixed once for Turkish-language questions.

This is a whole class of questions: any English speaker asking about KVKK. Cue added; K1 now routes to KVKK correctly.

### 2. DORA's own scoped entity types weren't cues — **fixed**

`REGULATION_TRIAGE_PROMPT` lists what DORA covers — *"banks, payment/e-money institutions, investment firms, insurers"* — but only `credit institution` and `eu bank` were ever encoded in `_DORA_CONTEXT_RE`. So S2's *"EU-licensed payment institution"* with an ICT vendor outage matched nothing and retrieved **zero DORA sources** on a question that is almost entirely about DORA. Added `payment institution`, `e-money institution`, `investment firm`, `insurance undertaking`, `payment service provider`. S2 now retrieves both DORA and GDPR.

Both fixes are additive-only (they can only add a regulation to the required set, never remove one), matching the pattern your codebase already established for `_GDPR_CONTEXT_RE`. Three regression tests added, including a guard that the pinned "payments startup in Istanbul" case still routes KVKK-only and doesn't start matching DORA.

---

## Real findings (reported, not changed)

### 3. English questions about KVKK have **no lexical retrieval path at all**

Measured directly: BM25 token overlap between the English K1 query and Kurul Kararı 2018/10 is **exactly zero**. That's structural, not tuning — different alphabets of vocabulary.

The mechanism built to solve this, `_kvkk_native_query()` / `search(extra_query=...)`, only fires when the *question* is in Turkish. For an English question it returns `None`. So:

- Turkish question → KVKK: BM25 works, plus native-language extra query. Two independent paths.
- **English question → KVKK: BM25 contributes nothing, no extra query. Cross-lingual embeddings are the *only* path, with no fallback** — and `search()`'s own docstring documents that cross-lingual embedding similarity is "markedly weaker than same-language similarity for the exact same content."

This is the weakest retrieval path in the system, and it's the one an English-speaking compliance reviewer would use every day. The symmetric fix is to translate the query *into Turkish* for KVKK-filtered searches (mirroring what the existing English rewrite does for the EU instruments) and pass it as `extra_query` — giving the KVKK side the same-language lexical signal it currently only gets from Turkish questions.

Related: the query rewrite's failure path is `except Exception: return question` — for a cross-lingual question that fallback returns the raw untranslated question, which yields **zero results** rather than degraded ones. A hard failure disguised as a graceful one.

### 4. Compare mode's false parity, observed live

C2 (DORA ↔ GDPR) retrieved, on the GDPR side, Articles **54, 47, 83, 28** — supervisory authority rules, binding corporate rules, fines, and processor terms — for a question about incident reporting deadlines. Article 33 didn't make the cut. Meanwhile the DORA side returned four genuinely on-topic documents.

This is exactly the structural issue from the compare-mode audit: each side's relevance floor is relative to *its own* best match, so a side with nothing on-topic still delivers a full, confident-looking quota. Part of this specific miss is the missing query rewrite, but the shape is real and the hardened `COMPARE_TEMPLATE` (telling the model that equal list length is not evidence of equal coverage) is what has to carry it.

### 5. Minor: the newly-parsed Geçici articles add scenario noise

Fix #6 from the earlier audit correctly split KVKK's transitional articles out — the parser now yields 36 articles including `Geçici 1/2/3`. Side effect: in S1, `Geçici 1` and `Geçici 2` (about transition periods and expert appointments) surfaced in the top 8 for a health-data scenario where they're irrelevant. Two of their titles are also mis-recovered by the title heuristic (`Geçici 2` is titled *"Başbakanlık tarafından yerine getirilir."*, a fragment of the preceding line). Cosmetic and low-impact, but worth a look if you re-tune the parser.

---

## Sandbox artifacts — do not chase these

- **K1, G3 retrieval misses** — pure cross-lingual cases. BM25 alone cannot cross the language boundary; these need the embedding model that couldn't be downloaded here.
- **S1, S2, C2 partial misses** — under `EchoLLM` there is no query rewrite and no scenario decomposition, so a long multi-issue scenario runs one undifferentiated retrieval pass against the raw question. Your `eval_cases.yaml` header already documents this caveat for the original set; it applies identically here.

---

## Run the full version yourself

Everything is committed. On your Mac, in the repo, with `REGINTEL_API_KEY` set:

```bash
# retrieval only, no model, no cost — but WITH semantic search
python scripts/eval.py --cases tests/eval_cases_v4.yaml --llm echo

# full run: retrieval + generated answers + citation checks
python scripts/eval.py --cases tests/eval_cases_v4.yaml
```

The full run writes a report with every answer and its cited sources to `data/eval_runs/<timestamp>.md`. Fifteen questions against qwen3-14b on OpenRouter is a few cents.

I added a `--cases` flag to `scripts/eval.py` for this, so the v4 set runs without disturbing your original 20-case set.

**Send me that report file and I'll analyze the reasoning half** — the answer-quality checks (`citations_from_only`, `forbid_regex`, `require_regex`) only grade when a real model runs, and the things that actually matter here (did it correct S1's false premise? did it keep C2's two 72-hour figures apart? did S3 stay in Turkish throughout?) need reading, not regex.

---

## Live run — 2026-08-22

The full live run (semantic search + hosted API model + hand-graded answers) is done: **`docs/eval-v4-live-run-2026-08-22.md`**.

Headline: automated checks say 12/15 pass, but reading every answer against its own `expected`/`fail_condition` puts the real number at 6 clean pass / 4 partial / 5 fail — including two reproduced failures (C3, S3) that the automated `citations_from_only` check cannot see. It also caught a live, previously-unknown bug: `python scripts/ingest.py KVKK` (the very re-ingest step this report told you to run) silently deletes every KVKK Board decision from the index, because `clear_regulation()` matches on regulation alone. Full detail, the three open claims (all confirmed), and a ranked action list are in the new report.

**Follow-up (2026-08-23):** most of that action list got done — see §7 of the live-run report. `clear_regulation()` fixed, the KVKK-only lockout in S3 fully resolved (three compounding causes, including a Turkish-language directive that succeeded where two English ones didn't), compare mode's retrieval width widened. Score moved to **10 pass / 4 partial / 1 fail**. D2 (TLPT frequency) remains the one confirmed, unfixed failure — treated as a genuine model/retrieval limit after repeated attempts, not a bug.
