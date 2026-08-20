"""RAG pipeline: retrieve -> assemble prompt -> generate -> return
answer with the sources that back it."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from regintel import config
from regintel.generation import prompts
from regintel.generation.llm import LLM, EchoLLM, get_llm
from regintel.retrieval.store import RegulationStore, SearchResult

_CITATION_RE = re.compile(r"\[(\d+)\]")

# Matches any configured regulation acronym as a whole word (case-insensitive).
_REG_NAME_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(r) for r in config.REGULATIONS) + r")\b",
    re.IGNORECASE)
_DANGLING_PREP_RE = re.compile(
    r"\b(under|per|pursuant to|according to)\s*[.,]?\s*$", re.IGNORECASE)
_ARTICLE_MENTION_RE = re.compile(r"\b(?:article|art\.?|madde)\s+(\d+)\b", re.IGNORECASE)
# Strips leading numbering/bullets ("1.", "- ", "* ") from a decomposed
# sub-question line — the prompt asks the LLM not to add these, but a
# local 7-8B model doesn't reliably comply.
_SUBQ_PREFIX_RE = re.compile(r"^[\s\-\*\d\.\)]+")

# Contextual cues that imply a regulation applies even when its acronym is
# never named — e.g. "payments startup in Istanbul" implies KVKK without
# saying "KVKK". See _detect_required_regulations().
_KVKK_CONTEXT_RE = re.compile(
    r"\b(istanbul|ankara|izmir|t[üu]rkiye|t[üu]rk vatanda|tckn|verb[iİ]s)\b",
    re.IGNORECASE)
_DORA_CONTEXT_RE = re.compile(
    r"\b(eu bank|european bank|financial entit(?:y|ies)|"
    r"ict third[- ]party|digital operational resilience|credit institution)\b",
    re.IGNORECASE)


def _strip_regulation_names(query: str) -> str:
    """Remove regulation acronyms (DORA, GDPR, KVKK...) from a retrieval
    query.

    The rewrite prompt already asks the model not to include them, but
    that's not guaranteed — and when it does, the acronym measurably
    drags cross-lingual semantic search toward generic front-matter
    articles ("what is this law") instead of the actual topic, since
    retrieval already filters by regulation separately. Belt-and-braces:
    strip mechanically regardless of prompt compliance.
    """
    stripped = _REG_NAME_RE.sub("", query)
    stripped = _DANGLING_PREP_RE.sub("", stripped)
    stripped = re.sub(r"\s{2,}", " ", stripped).strip(" .,")
    return stripped or query  # never return an empty query


def _detect_single_regulation(question: str) -> str | None:
    """If the question unambiguously names exactly one known regulation,
    return it; otherwise None.

    Used to auto-filter retrieval even when the caller didn't pass an
    explicit `regulation=` (e.g. UI "All" mode). Unfiltered retrieval
    systematically lets same-language instruments (GDPR/DORA, both
    English) outrank a genuinely relevant chunk from a different-
    language instrument (KVKK) for English questions — see the P0
    investigation. When the question names more than one regulation
    (e.g. a deliberate cross-instrument comparison), this returns None
    and retrieval stays unfiltered, since forcing a single instrument
    there would be wrong.
    """
    found = {m.upper() for m in _REG_NAME_RE.findall(question)}
    return found.pop() if len(found) == 1 else None


def _detect_required_regulations(question: str) -> set[str]:
    """Regulations that MUST be represented in retrieval, combining
    explicit acronym mentions with contextual jurisdiction cues.

    Scenario questions rarely name an instrument ("I'm a backend
    developer at a payments startup in Istanbul...") — a plain
    _detect_single_regulation() call finds nothing, retrieval stays
    unfiltered, and whichever regulation happens to win RRF fusion
    monopolizes the answer (the F4 finding: an Istanbul/KVKK scenario
    answered entirely from GDPR+DORA). This adds the missing signal
    without touching _detect_single_regulation, which existing callers
    rely on for its narrower "explicit mention only" contract.
    """
    required = {m.upper() for m in _REG_NAME_RE.findall(question)}
    if "KVKK" in config.REGULATIONS and _KVKK_CONTEXT_RE.search(question):
        required.add("KVKK")
    if "DORA" in config.REGULATIONS and _DORA_CONTEXT_RE.search(question):
        required.add("DORA")
    return required


# Citation-binding check (item 5): a marker citing a real, in-range
# source can still be attached to a claim that source doesn't actually
# support — the model reaching for the nearest numbered source rather
# than the right one. Checked with lexical overlap rather than a second
# LLM call: cheap, deterministic, and adds no generation latency. Short
# stopword lists (EN + TR) keep the overlap signal on content words
# rather than grammatical glue that would overlap with almost anything.
_STOPWORDS = frozenset({
    "the", "and", "for", "that", "with", "this", "from", "shall", "must",
    "have", "been", "will", "not", "are", "was", "were", "its", "their",
    "they", "which", "such", "also", "when", "where", "under", "into",
    "only", "than", "then", "these", "those", "about", "between",
    "within", "upon", "more", "some", "any", "all", "can", "may",
    "should", "would", "could", "does", "each", "other", "being",
    "veya", "gibi", "için", "olan", "olarak", "ile", "ancak", "ise",
    "kadar", "göre", "üzere", "olması", "olduğu", "olduğunu", "olup",
    "değil", "edilir", "edilmesi", "edilecek", "yapılır", "veya",
    "kişisel", "verilerin",
})
_MIN_CITATION_OVERLAP = 0.12
_WORD_RE = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]{4,}")


def _content_words(text: str) -> set[str]:
    return {w for w in (m.lower() for m in _WORD_RE.findall(text))
           if w not in _STOPWORDS}


def _marker_claim(text: str, start: int, end: int) -> str:
    """The sentence around a [n] marker's position in `text` — from the
    previous sentence boundary to the next — i.e. the claim the marker is
    actually attached to, not the whole answer."""
    left = max(text.rfind(".", 0, start), text.rfind("!", 0, start),
              text.rfind("?", 0, start), text.rfind("\n", 0, start))
    right_candidates = [p for p in (
        text.find(".", end), text.find("!", end),
        text.find("?", end), text.find("\n", end)) if p != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1:right + 1]


def _is_grounded(claim: str, source_text: str) -> bool:
    claim_words = _content_words(claim)
    if not claim_words:
        return True  # nothing substantive in the claim to check
    overlap = claim_words & _content_words(source_text)
    return len(overlap) / len(claim_words) >= _MIN_CITATION_OVERLAP


def cited_sources(answer: str, results: list[SearchResult]
                  ) -> tuple[str, list[int], list[SearchResult]]:
    """Sources the answer actually cited via [n] markers, not the raw
    retrieval set — plus the answer text with any dangling or unsupported
    markers stripped.

    Retrieval always returns top_k results whether or not they're
    relevant; without this, a refusal ("sources don't cover this") would
    still display every retrieved chunk as if it backed the answer. Only
    markers the model actually wrote, that resolve to a real source AND
    whose immediate claim is actually grounded in that source's text
    (see _is_grounded), are kept — a refusal with no [n] markers
    correctly comes back empty. A marker citing a source number that
    doesn't exist (e.g. [6] against 3 sources) or citing a real source
    that doesn't support the sentence it's attached to (the model
    reaching for the nearest numbered source instead of the right one)
    is stripped from the answer text entirely rather than left dangling:
    an unresolved or false attribution reads as per-claim evidence to a
    reader and is worse than no marker at all. Indices are the original
    1-based marker numbers (matching format_sources' numbering), not a
    fresh 1..k, so a displayed "[3]" always matches the "[3]" left in the
    answer text.
    """
    valid_max = len(results)

    def _strip_ungrounded(m: re.Match) -> str:
        n = int(m.group(1))
        if not (1 <= n <= valid_max):
            return ""
        claim = _marker_claim(m.string, m.start(), m.end())
        return m.group(0) if _is_grounded(claim, results[n - 1].text) else ""

    clean_answer = _CITATION_RE.sub(_strip_ungrounded, answer)
    # Tidy up connective words/punctuation left dangling by a removed
    # marker, e.g. "...by [3] and [6]." -> "...by [3] and ." -> "...[3]."
    clean_answer = re.sub(r"\s+and\s*([.,;:]|$)", r"\1", clean_answer)
    clean_answer = re.sub(r",\s*([.,;:])", r"\1", clean_answer)
    clean_answer = re.sub(r" {2,}", " ", clean_answer)
    clean_answer = re.sub(r" ([.,;:])", r"\1", clean_answer)

    cited = {int(m) for m in _CITATION_RE.findall(clean_answer)}
    pairs = [(i, r) for i, r in enumerate(results, 1) if i in cited]
    return clean_answer, [i for i, _ in pairs], [r for _, r in pairs]


def group_sources(indices: list[int], results: list[SearchResult]) -> list[dict]:
    """Group cited sources by article for display.

    A long article split into several chunks (e.g. 5 of the 7 parts of
    DORA Art 19 all cited) should render as one line listing every part,
    not 5 separate lines that look like near-duplicates of each other.
    """
    groups: dict[tuple, dict] = {}
    order: list[tuple] = []
    for idx, r in zip(indices, results):
        m = r.metadata
        key = (m["regulation"], m["article_number"])
        if key not in groups:
            if m.get("doc_type", "statute") != "statute" and m.get("document_label"):
                item = f" ({m['article_number']})" if m.get("article_number") else ""
                base_citation = f"{m['regulation']}, {m['document_label']}{item}"
            else:
                title = f" — {m['article_title']}" if m.get("article_title") else ""
                base_citation = f"{m['regulation']}, Article {m['article_number']}{title}"
            groups[key] = {
                "indices": [], "parts": [], "total": m.get("total_chunks", 1),
                "texts": [], "base_citation": base_citation,
            }
            order.append(key)
        g = groups[key]
        g["indices"].append(idx)
        g["texts"].append(r.text)
        if g["total"] > 1:
            g["parts"].append(m["chunk_index"] + 1)

    out = []
    for key in order:
        g = groups[key]
        citation = g["base_citation"]
        if g["parts"]:
            parts = ", ".join(str(p) for p in sorted(g["parts"]))
            citation += f" (parts {parts} of {g['total']})"
        out.append({"indices": g["indices"], "citation": citation,
                    "text": "\n\n---\n\n".join(g["texts"])})
    return out


@dataclass
class RagResponse:
    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    cited_indices: list[int] = field(default_factory=list)  # parallel to sources

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [
                {"index": idx, "citation": s.citation, "text": s.text,
                 "metadata": s.metadata, "score": s.score}
                for idx, s in zip(self.cited_indices, self.sources)
            ],
        }


class RagPipeline:
    def __init__(self, store: RegulationStore | None = None,
                 llm: LLM | None = None):
        self.store = store or RegulationStore()
        self.llm = llm or get_llm()

    def _retrieval_query(self, question: str) -> str:
        """Rewrite the question into a short English search query.

        Regulations are indexed in English; a Turkish question would miss
        BM25 entirely and weaken semantic search. Falls back to the raw
        question if rewriting fails.
        """
        if isinstance(self.llm, EchoLLM):
            return question
        try:
            q = self.llm.chat(prompts.QUERY_REWRITE_PROMPT, question).strip()
            q = q[:300] if q else question
        except Exception:
            return question
        return _strip_regulation_names(q)

    def _ask_prompt(self, question: str, regulation: str | None,
                     top_k: int) -> tuple[list[SearchResult], str]:
        if self._needs_decomposition(question):
            sub_questions = self._decompose_question(question)
            if len(sub_questions) > 1:
                return self._decomposed_prompt(question, sub_questions, regulation, top_k)

        # Auto-filter to a single named instrument even if the caller
        # left `regulation` unset (e.g. UI "All" mode) — see
        # _detect_single_regulation's docstring for why this matters.
        if regulation is None:
            regulation = _detect_single_regulation(question)
        query = self._retrieval_query(question)

        if regulation is None:
            # A scenario question can imply more than one instrument
            # without naming any of them (F4 in the eval report) — force
            # per-regulation retrieval so a required instrument can't be
            # crowded out of the RRF ranking entirely.
            required = _detect_required_regulations(question)
            if len(required) >= 2:
                results = self._multi_regulation_search(query, required, top_k)
                scope_note = "".join(self._scope_note(r) for r in sorted(required))
                prompt = prompts.ANSWER_TEMPLATE.format(
                    scope_note=scope_note,
                    sources=prompts.format_sources(results), question=question)
                return results, prompt
            if len(required) == 1:
                regulation = next(iter(required))

        results = self.store.search(query, top_k=top_k, regulation=regulation)
        prompt = prompts.ANSWER_TEMPLATE.format(
            scope_note=self._scope_note(regulation),
            sources=prompts.format_sources(results), question=question)
        return results, prompt

    def _needs_decomposition(self, question: str) -> bool:
        """Trigger for scenario decomposition: a long message, or one
        whose context cues already imply multiple regulations — both are
        signs of a multi-issue scenario a single retrieval pass would
        under-serve. Short single-issue questions (the common case) never
        cross the length threshold, so their latency is unaffected.

        The multi-regulation branch also requires
        SCENARIO_MIN_LENGTH_FOR_MULTI_REG — see that constant's comment
        in config.py for why a short comparative question naming two
        regulations shouldn't decompose on its own.
        """
        if not config.SCENARIO_DECOMPOSITION_ENABLED:
            return False
        if len(question) > config.SCENARIO_LENGTH_THRESHOLD:
            return True
        if len(question) < config.SCENARIO_MIN_LENGTH_FOR_MULTI_REG:
            return False
        return len(_detect_required_regulations(question)) >= 2

    def _decompose_question(self, question: str) -> list[str]:
        """Extract discrete legal sub-questions via one LLM call. Falls
        back to [question] (i.e. "don't decompose") for EchoLLM or if the
        call fails — same fallback pattern as _retrieval_query, and the
        caller (_ask_prompt) already treats a single-item result as "no
        decomposition happened" and continues down the normal path.
        """
        if isinstance(self.llm, EchoLLM):
            return [question]
        try:
            raw = self.llm.chat(prompts.DECOMPOSE_PROMPT, question)
        except Exception:
            return [question]
        sub_qs = [_SUBQ_PREFIX_RE.sub("", line).strip() for line in raw.splitlines()]
        sub_qs = [q for q in sub_qs if len(q) > 8]  # drop stray blank/junk lines
        return sub_qs[:5] if sub_qs else [question]

    def _decomposed_prompt(self, question: str, sub_questions: list[str],
                            regulation: str | None, top_k: int
                            ) -> tuple[list[SearchResult], str]:
        """Retrieval runs once per extracted issue and results are
        merged (deduped by chunk id). `regulation` is the caller's
        explicit filter, if any (e.g. the UI's regulation dropdown), and
        is respected for every sub-question; when unset, each
        sub-question detects its own regulation independently — a
        multi-issue scenario often has different issues governed by
        different instruments (see _decompose_question's docstring).
        """
        per_issue_k = max(top_k // len(sub_questions), 3)
        results: list[SearchResult] = []
        seen_ids: set[str] = set()
        regs_covered: set[str] = set()
        for sub_q in sub_questions:
            sub_reg = regulation or _detect_single_regulation(sub_q)
            sub_query = self._retrieval_query(sub_q)
            for r in self.store.search(sub_query, top_k=per_issue_k, regulation=sub_reg):
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    results.append(r)
            if sub_reg:
                regs_covered.add(sub_reg)

        scope_note = "".join(self._scope_note(r) for r in sorted(regs_covered))
        issues = "\n".join(f"- {q}" for q in sub_questions)
        prompt = prompts.SCENARIO_ANSWER_TEMPLATE.format(
            scope_note=scope_note, sources=prompts.format_sources(results),
            question=question, issues=issues)
        return results, prompt

    def _multi_regulation_search(self, query: str, regulations: set[str],
                                  top_k: int) -> list[SearchResult]:
        """Search each required regulation separately and merge, so every
        regulation _detect_required_regulations() flagged actually
        contributes chunks — mirrors _compare_prompt's per-regulation
        retrieval, generalized to N regulations from context cues rather
        than the two explicit reg_a/reg_b of comparison mode.
        """
        per_reg_k = max(top_k // len(regulations), 3)
        results: list[SearchResult] = []
        seen_ids: set[str] = set()
        for reg in sorted(regulations):
            for r in self.store.search(query, top_k=per_reg_k, regulation=reg):
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    results.append(r)
        return results

    def _scope_note(self, regulation: str | None) -> str:
        """Corpus scope line for a single-instrument question — lets the
        model correctly resolve "does article N exist" without having to
        guess (retrieval only ever hands it snippets, not corpus-wide
        facts), and states which document types are actually indexed so
        it can give the out-of-corpus answer (SYSTEM_PROMPT rule 12)
        instead of paraphrasing statute text for a question that's really
        about a Board decision. Empty when no single regulation is
        targeted (e.g. "All" mode, or a genuine cross-instrument question).
        """
        if not regulation:
            return ""
        r = self.store.article_range(regulation)
        types = self.store.doc_types(regulation)
        if not r and not types:
            return ""

        parts = []
        if r:
            lo, hi = r
            parts.append(f"{regulation} statute is indexed from Article "
                        f"{lo} to Article {hi}")
        non_statute = types - {"statute"}
        if non_statute:
            labels = ", ".join(sorted(t.replace("_", " ") for t in non_statute))
            parts.append(f"also indexed for {regulation}: {labels}")
        else:
            parts.append(f"no Board (Kurul) decisions or guidance documents "
                        f"are indexed for {regulation} — statute text only")
        return "Corpus scope: " + "; ".join(parts) + ".\n\n"

    def _out_of_range_article(self, question: str, regulation: str | None
                              ) -> str | None:
        """If the question names one regulation and one explicit article
        number, and that number is outside the corpus's confirmed
        indexed range, return a deterministic "does not exist" answer —
        else None.

        The prompt already tells the model this (see _scope_note) and
        instructs it to act on it (SYSTEM_PROMPT rule 11), but a local
        8B model doesn't reliably apply a numeric range-check from
        instructions alone — verified empirically (it kept saying
        "sources don't cover this" for KVKK Art 47 / DORA Art 72 even
        with the range stated right above the sources). Denying a real
        provision is exactly as damaging as inventing one, so for this
        narrow, structurally-detectable case, check deterministically
        instead of trusting the model's reasoning: it can only fire when
        the number is confirmed OUTSIDE the indexed range, so it can
        never falsely deny a real article.

        This bypasses the LLM entirely (SYSTEM_PROMPT rule 6, "answer in
        the language the question was asked in", never applies), so the
        message picks its own language from which word the question
        used to name the article — "madde" for Turkish, "article"/"art."
        for English.
        """
        reg = regulation or _detect_single_regulation(question)
        if not reg:
            return None
        m = _ARTICLE_MENTION_RE.search(question)
        if not m:
            return None
        n = int(m.group(1))
        r = self.store.article_range(reg)
        if r and not (r[0] <= n <= r[1]):
            if m.group(0).lower().startswith("madde"):
                return (f"{reg} Madde {n} mevcut değil — bu külliyatta {reg} "
                        f"Madde {r[0]} ile Madde {r[1]} arasında indekslenmiştir.")
            return (f"{reg} Article {n} does not exist — {reg} in this corpus "
                    f"is indexed from Article {r[0]} to Article {r[1]}.")
        return None

    def ask(self, question: str, regulation: str | None = None,
            top_k: int = config.DEFAULT_TOP_K) -> RagResponse:
        """Answer a question, optionally restricted to one regulation."""
        oor = self._out_of_range_article(question, regulation)
        if oor:
            return RagResponse(answer=oor)
        results, prompt = self._ask_prompt(question, regulation, top_k)
        answer = self.llm.chat(prompts.SYSTEM_PROMPT, prompt)
        answer, indices, sources = cited_sources(answer, results)
        return RagResponse(answer=answer, sources=sources, cited_indices=indices)

    def ask_stream(self, question: str, regulation: str | None = None,
                    top_k: int = config.DEFAULT_TOP_K
                    ) -> tuple[list[SearchResult], Iterator[str]]:
        """Like ask(), but returns the raw retrieved sources plus a token
        generator for the answer so the caller can render it as it's
        produced. The answer text isn't known until the stream is
        consumed, so callers must apply cited_sources(answer, results)
        themselves afterward to get the cleaned answer text and the
        citation-grounded source list — these raw `results` are the
        full retrieval set, not yet filtered."""
        oor = self._out_of_range_article(question, regulation)
        if oor:
            return [], iter([oor])
        results, prompt = self._ask_prompt(question, regulation, top_k)
        return results, self.llm.stream_chat(prompts.SYSTEM_PROMPT, prompt)

    def _compare_prompt(self, question: str, reg_a: str, reg_b: str,
                         top_k_each: int) -> tuple[list[SearchResult], str]:
        """Retrieval runs separately per regulation (metadata-filtered) so
        both sides are actually represented in the context — a single
        unfiltered search often returns chunks from only one framework.
        """
        query = self._retrieval_query(question)
        results_a = self.store.search(query, top_k=top_k_each, regulation=reg_a)
        results_b = self.store.search(query, top_k=top_k_each, regulation=reg_b)
        prompt = prompts.COMPARE_TEMPLATE.format(
            scope_note=self._scope_note(reg_a) + self._scope_note(reg_b),
            reg_a=reg_a, reg_b=reg_b,
            sources_a=prompts.format_sources(results_a),
            sources_b=prompts.format_sources(results_b, start=len(results_a) + 1),
            question=question,
        )
        return results_a + results_b, prompt

    def compare(self, question: str, reg_a: str, reg_b: str,
                top_k_each: int = 4) -> RagResponse:
        """Compare two regulations on a topic."""
        results, prompt = self._compare_prompt(question, reg_a, reg_b, top_k_each)
        answer = self.llm.chat(prompts.SYSTEM_PROMPT, prompt)
        answer, indices, sources = cited_sources(answer, results)
        return RagResponse(answer=answer, sources=sources, cited_indices=indices)

    def compare_stream(self, question: str, reg_a: str, reg_b: str,
                        top_k_each: int = 4
                        ) -> tuple[list[SearchResult], Iterator[str]]:
        """Like compare(), but returns the raw sources plus a token
        generator — see ask_stream()'s note on applying cited_sources()."""
        results, prompt = self._compare_prompt(question, reg_a, reg_b, top_k_each)
        return results, self.llm.stream_chat(prompts.SYSTEM_PROMPT, prompt)
