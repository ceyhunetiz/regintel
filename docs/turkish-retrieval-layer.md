# Design note: a Turkish retrieval layer for the EU instruments

**Status:** proposal, not implemented. Needs a decision from whoever owns compliance risk at the bank.
**Date:** 2026-08-22
**Context:** the bank's users ask questions in Turkish. KVKK is a Turkish corpus; DORA and GDPR are English.

---

## The question this answers

Should DORA and GDPR be translated into Turkish so that Turkish questions match Turkish text directly?

**Short answer: no — not the source text.** But there is a version of the idea that is both safe and useful, and this note specifies it.

## Why not translate the corpus

**1. There is no authentic Turkish text of either instrument.** The EU publishes legislation in its 24 official languages; Turkish is not one of them. Any Turkish DORA or GDPR would be an unofficial translation with no legal standing.

**2. The project already has a rule against exactly this.** `SYSTEM_PROMPT` rule 13: a source whose label says it is not official text "is never authority for a legal obligation's operative wording." That rule exists for one small file — the `7499 Değişiklik Notu`, marked *gayriresmi*. Translating the EU corpus would create ~163 statutory articles and 15 Level-2 documents in that same category, and they would be the *only* text available for those instruments. A compliance file citing an unofficial Turkish rendering of GDPR Art 33 as operative wording is an audit finding.

**3. Translation would make the instruments look more alike than they are.** Rendering GDPR into Turkish means rendering it into *KVKK's* legal vocabulary — *veri sorumlusu*, *veri işleyen*, *açık rıza* — terms that already carry defined meanings under Law 6698. GDPR's "without undue delay" and KVKK's "en kısa sürede" are different legal standards that a translation collapses into one phrase. That works directly against rule 8 (instrument fidelity) and rule 15 (jurisdiction-conditional answers), which most of this project's engineering effort has gone into protecting.

**4. It solves a problem the bank does not have.** Turkish → KVKK is native. Turkish → DORA/GDPR already works, because the pipeline rewrites the question into English before searching. The genuinely weak path is *English → KVKK*, which these users will not take.

## What to build instead

**Translate for findability; cite the original.**

Keep the authentic English text as the only indexed, quotable, citable source. Attach a Turkish *retrieval aid* — never presented as the law's wording:

- a Turkish rendering of each article's title;
- one Turkish sentence naming the article's subject matter and its key domain terms.

### Where it goes

`chunker.py` already prepends a context line to each chunk's embedded text:

```python
text=f"{article.regulation}, {label}:\n{piece}"
```

The gloss belongs in that prefix — visible to BM25 and the embedder, structurally separate from the provision:

```python
gloss = f"[TR arama yardımı — resmi metin değildir] {article.gloss_tr}\n" if article.gloss_tr else ""
text=f"{article.regulation}, {label}:\n{gloss}{piece}"
```

### Rules it must obey

1. **Never citable as wording.** Extend rule 13 to name the gloss explicitly. The marker `[TR arama yardımı — resmi metin değildir]` must appear in the chunk so the model can see what it is.
2. **Never the answer's substance.** A gloss is a search aid; the obligation must be quoted from the English provision.
3. **Generated once, reviewed, versioned.** Not regenerated per query. Store alongside the corpus so a reviewer can diff it.
4. **Excluded from `format_sources`' quotable body** if reviewers prefer belt-and-braces — the gloss can live in a separate metadata field indexed for search but stripped before the text reaches the prompt.

Option 4 is the strictest and the one I would recommend if the compliance function is uneasy: retrieval matches on the Turkish gloss, but the model never sees it, so it cannot possibly quote it.

### Cost

163 statutory articles plus ~15 Level-2 documents. One LLM pass at generation time, reviewed once by someone who reads both languages. Regenerated only when the corpus changes.

### What it buys

Turkish questions gain a same-language lexical (BM25) path into the EU instruments, instead of depending entirely on the query-rewrite step. Today that rewrite is a single point of failure: if it fails, an unrewritten Turkish query returns **zero** BM25 chunks from GDPR. The hardening added on 2026-08-22 (retry, translation validation, and a degradation notice to the model) makes that failure loud rather than silent — but a Turkish gloss layer would make it non-fatal.

## Recommended order

1. ~~Harden the query rewrite~~ — **done 2026-08-22** (retry + validation + degradation notice; 7 regression tests).
2. Decide on the gloss layer with the compliance owner. If approved, build it with option 4 (search-only, never in the prompt).
3. Only then revisit whether anything further is needed.

## What must not happen

Indexing a machine-translated DORA or GDPR as though it were the instrument's text — no marker, no separate field, no review. That is the one version of this idea that turns a retrieval improvement into a compliance liability.
