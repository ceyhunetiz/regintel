"""RAG pipeline: retrieve -> assemble prompt -> generate -> return
answer with the sources that back it."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from regintel import config
from regintel.generation import prompts
from regintel.generation.llm import LLM, EchoLLM, REASONING_EXHAUSTED_MESSAGE, get_llm
from regintel.retrieval.store import RegulationStore, SearchResult

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")
# A named "<Regulation> Article N" mention in plain prose reads as a
# citation to anyone reading it, but rule 0's citation-grounding check
# (cited_sources, below) only ever looks at bracketed "[n]" markers — a
# specific article number the model recalls from its own training data,
# never bracketed at all, sails through completely unchecked (observed:
# an otherwise well-cited answer stating "GDPR Article 6" as settled
# fact when Article 6 was never among that answer's actually-retrieved
# sources at all — confirmed by checking the real retrieval set). Only
# matches a plain numbered statute article ("Article 6", "Madde 12") —
# a non-statute document (a Board decision, an RTS...) is cited by name
# ("Kurul Kararı 2019/10"), not this pattern, so this never touches those.
_PROSE_ARTICLE_RE = re.compile(
    r"\b(DORA|GDPR|KVKK)\s+(?:Article|Art\.?|Madde)\s+(\d+)\b", re.IGNORECASE)

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
# saying "KVKK". See _detect_required_regulations(). "kurul"/"kurulun"...
# is the Turkish Personal Data Protection Board — a question referencing
# a "Kurul Kararı" (Board decision) by number alone, with no other KVKK
# cue in it, previously matched nothing here: retrieval ran unfiltered,
# and the query-rewrite step (translating to English for indexed-content
# search) had nothing to anchor it, observed producing a hallucinated
# English rewrite ("regulation governing financial institutions") that
# then let unrelated DORA content dominate an unfiltered top-k (v2 eval,
# case Q8: "does Kurul Kararı 2019/10 still apply?" answered "not in my
# sources" despite it being indexed).
# "turkey|turkish" alongside "türkiye": an ENGLISH question about Turkish
# law ("what does Turkish data protection law require...") contains no
# Turkish characters and never spells the country the Turkish way, so it
# used to hit no KVKK cue at all — retrieval ran unfiltered and the
# English-language, higher-volume GDPR content monopolized the result
# set. Observed on the v4 eval set, case K1: an English question
# explicitly asking about Turkish law retrieved 8 GDPR articles and zero
# KVKK.
_KVKK_CONTEXT_RE = re.compile(
    r"\b(istanbul|ankara|izmir|t[üu]rkiye|turkey|turkish|t[üu]rk vatanda|"
    r"tckn|verb[iİ]s|kurul(?:un|ca|dan|u)?)\b",
    re.IGNORECASE)
# The entity types below are the ones DORA itself is scoped to, and the
# ones REGULATION_TRIAGE_PROMPT already lists in prose ("banks,
# payment/e-money institutions, investment firms, insurers") — but only
# "credit institution" and "eu bank" were ever encoded here, so a
# question naming any of the others routed nowhere. Observed on the v4
# eval set, case S2: "EU-licensed payment institution" with an ICT vendor
# outage matched no DORA cue and retrieved zero DORA sources.
#
# Turkish terms added separately (banka/bankası, ödeme kuruluşu...): the
# English-only list above can never fire for a Turkish-language
# scenario naming a bank, no matter how clearly financial the entity is
# — observed on eval case S3 ("Türkiye'de kurulu bir bankanın..."),
# where DORA never even entered the classifier's consideration.
_DORA_CONTEXT_RE = re.compile(
    r"\b(eu bank|european bank|financial entit(?:y|ies)|"
    r"ict third[- ]party|digital operational resilience|credit institution|"
    r"payment institution|e-?money institution|electronic money institution|"
    r"investment firm|insurance undertaking|payment service provider|"
    r"banka(?:s[ıi]|n[ıi]n|da|dan|ya)?|[öo]deme kurulu[şs]u|"
    r"finansal kurulu[şs]|yat[ıi]r[ıi]m kurulu[şs]u|sigorta [şs]irket\w*)\b",
    re.IGNORECASE)
# An explicit EU/Avrupa Birliği mention is a deterministic, additive-only
# GDPR signal, kept as a regex rather than folded into the LLM triage
# classifier (REGULATION_TRIAGE_PROMPT) after a prompt-wording attempt at
# this exact gap made things strictly worse: it didn't fix the miss this
# targets and newly broke a previously-correct case (a payment-outage
# question started incorrectly flagging GDPR). A holistic classification
# prompt can't be reliably steered by prose alone; a narrow, mechanical,
# ADD-only check for this one explicit, high-confidence cue can't cause
# that kind of regression, since it only ever adds GDPR on top of
# whatever the classifier already found, never removes anything.
# Concretely observed: a Turkish-language question naming "Türkiye" (an
# established KVKK cue) alongside an explicit "AB'de" / "Avrupa
# Birliği" (EU) mention was classified KVKK-only, missing GDPR entirely
# even though the question explicitly named EU customers as in scope.
# ['’´] — Turkish suffix apostrophes come in three flavors in real
# input: straight ('), curly (’, what macOS/Word/phones type by
# default), and acute (´); matching only the straight one silently
# missed the most common typed form of "AB’de".
#
# de(?:ki(?:ler)?)? — "AB'de" (in the EU) has a sibling form "AB'deki"
# (the [X] that's in the EU, e.g. "AB'deki müşterilerin verileri" = "the
# data OF THE CUSTOMERS THAT ARE IN THE EU") that differs by exactly
# this suffix. \b after a bare "de" never matches inside "deki" — "de"
# and "ki" are one continuous word token, not two — so the un-suffixed
# pattern silently missed every "AB'deki ..." phrasing. Concretely
# observed on eval case S3: "Türkiye'deki hem AB'deki müşterilerin
# verileri" matched the KVKK cue and missed this GDPR one entirely,
# because "AB'deki" specifically (not "AB'de") is the form that names
# EU customers as in scope.
_GDPR_CONTEXT_RE = re.compile(
    r"\b(ab['’´](?:de(?:ki(?:ler)?)?|nin|ye|den|yle)|avrupa birli[gğ]i|"
    r"european union|eu)\b",
    re.IGNORECASE)

# A "home jurisdiction sends data to a foreign destination" pattern —
# e.g. "İstanbul'da bir sağlık teknolojisi girişimindeyim... Frankfurt
#'taki bir bulut sağlayıcısına gönderiyoruz" (eval case S1). Left
# alone, a scenario like this let the destination's own regime become
# the PRIMARY lens for the ENTIRE answer just because it was the last
# jurisdiction mentioned (observed: a KVKK-primary Turkish scenario
# answered "based only on GDPR" once Frankfurt appeared — GDPR's
# larger English corpus dominating both retrieval and the model's own
# framing). The fix isn't retrieval, it's telling the model explicitly
# which side is the HOME regime and which is only relevant for the
# transfer leg — see _home_destination_note.
#
# Separate from _GDPR_CONTEXT_RE on purpose: that regex is deliberately
# narrow (explicit "AB"/"EU" mentions only) and used elsewhere to
# decide which regulations retrieval must cover — broadening it to
# include specific cities risks changing that unrelated behavior. This
# is scoped only to the home/destination check below.
_EU_LOCATION_RE = re.compile(
    r"\b(ab['’´](?:de(?:ki(?:ler)?)?|nin|ye|den|yle)|avrupa birli[gğ]i|"
    r"european union|frankfurt|berlin|m[üu]nih|munich|almanya|germany|"
    r"amsterdam|hollanda|netherlands|dublin|[iİ]rlanda|ireland|"
    r"paris|fransa|france)\b",
    re.IGNORECASE)
# Turkish/English verbs for actively SENDING data somewhere — not just
# "using" or "accessing" a foreign party, which describes a genuinely
# separate relationship (eval case S3's Ireland-based SaaS provider
# and Amsterdam branch), not a transfer of the SAME data the rest of
# the scenario is about. Requiring this verb, not just a second
# jurisdiction cue anywhere in the text, is what keeps this from
# firing on S3 — it has no send/transfer verb linking its jurisdiction
# mentions at all.
_TRANSFER_VERB_RE = re.compile(
    r"\b(g[öo]nder\w*|aktar\w*|y[üu]kle\w*|ilet\w*|"
    r"send\w*|transfer\w*|upload\w*|backup\w*|forward\w*)\b",
    re.IGNORECASE)
_TRANSFER_VERB_WINDOW = 80  # characters either side of the verb


def _detect_home_destination(question: str) -> tuple[str, str] | None:
    """Detect a scenario with a single, unambiguous "home jurisdiction
    sends data to a foreign destination" shape and return (home_reg,
    destination_reg), else None.

    Requires: a KVKK cue, a GDPR/EU-location cue, and a transfer verb
    sitting near exactly ONE of the two cue types, with no OTHER,
    unrelated occurrence of that same cue type elsewhere in the text.
    That last check is what tells a real transfer (S1: one Turkish
    company, one dataset, one send-to-Frankfurt) apart from a scenario
    with genuinely separate, parallel jurisdiction-specific stories
    (S3: a Turkish bank branch AND an independent Amsterdam branch AND
    an Ireland-based vendor relationship — three different things, not
    one thing moving from A to B). Firing on the latter would silently
    reintroduce the KVKK-cue lockout bug this project already fixed
    once (a scenario answered only from whichever jurisdiction's cue
    happened to appear first).
    """
    kvkk_positions = [m.start() for m in _KVKK_CONTEXT_RE.finditer(question)]
    eu_positions = [m.start() for m in _EU_LOCATION_RE.finditer(question)]
    if not kvkk_positions or not eu_positions:
        return None

    for verb_m in _TRANSFER_VERB_RE.finditer(question):
        lo, hi = verb_m.start() - _TRANSFER_VERB_WINDOW, verb_m.end() + _TRANSFER_VERB_WINDOW
        near_eu = [p for p in eu_positions if lo <= p <= hi]
        near_kvkk = [p for p in kvkk_positions if lo <= p <= hi]
        if near_eu and not near_kvkk and len(near_eu) == len(eu_positions):
            return ("KVKK", "GDPR")
        if near_kvkk and not near_eu and len(near_kvkk) == len(kvkk_positions):
            return ("GDPR", "KVKK")
    return None


_JURISDICTION_LABEL = {"KVKK": "Turkey (governed by KVKK)", "GDPR": "the EU/EEA (governed by GDPR)"}


def _home_destination_note(question: str) -> str:
    pair = _detect_home_destination(question)
    if not pair:
        return ""
    home, dest = pair
    home_label, dest_label = _JURISDICTION_LABEL[home], _JURISDICTION_LABEL[dest]
    return (f"Jurisdiction note: this scenario describes an entity based in "
            f"{home_label} sending data to a location in {dest_label}. "
            f"Treat {home}'s regime as PRIMARY, governing the underlying "
            f"processing, security, and breach obligations for this data — "
            f"{dest}'s regime is relevant ONLY for the cross-border "
            f"transfer step itself (the legal basis/safeguard needed to "
            f"send data there), never as the primary framework for the "
            f"rest of the scenario.\n\n")


# Catches a DIRECT (non-scenario) question that bundles two distinct
# factual asks under one regulation — e.g. "how often ... and how long
# does the active phase last" — where a single fused retrieval query
# dilutes toward whichever half has more matching terms and silently
# starves the other (observed: DORA's "at least every 3 years" TLPT
# frequency answer, cleanly retrievable in isolation, never surfacing at
# all once merged with a "how long does the active phase last" clause —
# wider candidate pools didn't help, because the losing clause's best
# chunk was genuinely outranked, not merely excluded). Deliberately
# separate machinery from _needs_decomposition/_decompose_question: no
# LLM call, and no interaction with scenario decomposition's trigger,
# which is a recently-fixed and sensitive path (see the scenario-
# decomposition regression fix in the project history) — this is a
# narrow, fully mechanical regex split on a fixed, short list of
# interrogative markers, and only ever engages for an already single-
# regulation direct question. See _compound_clauses.
# Stems end in \w* so inflected Turkish forms match too — "ne kadar
# sürer" / "ne sıklıkta" previously failed the trailing \b and the
# splitter silently never fired for the most natural Turkish phrasings.
_COMPOUND_MARKERS = [
    r"how often", r"how long", r"how many", r"how much",
    r"ne s[ıi]kl[ıi]k\w*", r"ne kadar s[üu]r\w*",
    r"ka[çc] kez", r"ka[çc] g[üu]n\w*",
]
_COMPOUND_MARKER_RE = re.compile(
    r"\b(" + "|".join(_COMPOUND_MARKERS) + r")\b", re.IGNORECASE)
# Conjunction/comma boundary between two clauses of a compound question.
_CLAUSE_CONJ_RE = re.compile(r",|\b(?:ve|and)\b", re.IGNORECASE)

# A genuine RECURRENCE question — "how often", not "how long"/"how
# many"/"how much" (duration/quantity, not frequency) — biases the
# retrieval query toward two things a recurring EU-legislative
# obligation actually says (see _augment_frequency_query):
#
#  1. "at least every" — the conventional phrasing for a recurring
#     obligation ("at least every 3 years", "at least every 6
#     months"...). Tried wider candidate pools and a plain "frequency"
#     rewrite first; neither changed anything, because DORA Art 26's
#     "at least every 3 years" TLPT paragraph is genuinely outranked on
#     its own vocabulary by sibling paragraphs that repeat "financial
#     entities" more densely, not merely cut off by a narrow top-k.
#  2. The abbreviation over the spelled-out term, when one exists for
#     the topic named in the question. This one is specific rather
#     than general: DORA Art 26(1) — the paragraph that actually states
#     the frequency — itself says "TLPT", never "threat-led penetration
#     test(ing)" in full; every SIBLING paragraph spells the term out
#     (often twice), so a query using the full form matches THEM
#     better and buries the one that answers the question. Rewriting
#     the query to match the answer paragraph's own wording, not the
#     question's, is what actually differentiates it. Verified
#     directly: appending "at least every" alone still left the answer
#     unranked at position 8 of the real production rewrite (top_k=6
#     misses it); adding the TLPT substitution on top moved it to rank
#     1. A general synonym-of-every-domain-acronym table isn't
#     justified by one case — this one substitution is, since it's the
#     actual, current, and only known failure this mechanism fixes.
_FREQUENCY_MARKER_RE = re.compile(
    r"\b(how often|ne s[ıi]kl[ıi]k\w*|ka[çc] kez)\b", re.IGNORECASE)
_TLPT_SPELLED_OUT_RE = re.compile(
    r"threat[- ]led penetration test(?:ing|s)?", re.IGNORECASE)


def _augment_frequency_query(query: str, original_text: str) -> str:
    if _FREQUENCY_MARKER_RE.search(original_text):
        query = _TLPT_SPELLED_OUT_RE.sub("TLPT", query)
        return f"{query} at least every"
    return query


def _marker_stem(text: str) -> str:
    """Canonical stem for a matched marker, so two inflections of the
    same marker ("ne kadar süre" / "ne kadar sürer") count as ONE
    distinct marker — a question repeating one marker is usually one ask
    restated, not two."""
    for stem in _COMPOUND_MARKERS:
        if re.fullmatch(stem, text, re.IGNORECASE):
            return stem
    return text.lower()


def _compound_clauses(question: str) -> list[str]:
    """Split a question into one clause per distinct interrogative-marker
    match — e.g. "How often do we have to run X, and how long does Y
    last?" -> ["How often do we have to run X", "how long does Y
    last?"]. Returns [] (i.e. "not compound") unless at least 2
    DIFFERENT marker phrases are found.

    The cut between two markers falls at the LAST conjunction/comma
    before the next marker, not at the next marker itself: Turkish puts
    a clause's subject BEFORE its interrogative marker ("aktif test
    aşaması ne kadar sürer"), so cutting at marker starts (as a previous
    version did) stole each clause's subject and attached it to the
    previous clause — retrieval then ran on subject-less verb phrases.
    Text before the first marker (the shared context/subject) stays in
    the first clause for the same reason.
    """
    matches = list(_COMPOUND_MARKER_RE.finditer(question))
    distinct = {_marker_stem(m.group(0)) for m in matches}
    if len(distinct) < 2:
        return []
    cuts = [0]
    for prev, nxt in zip(matches, matches[1:]):
        seg = question[prev.end():nxt.start()]
        conj = list(_CLAUSE_CONJ_RE.finditer(seg))
        cuts.append(prev.end() + (conj[-1].start() if conj else 0))
    cuts.append(len(question))
    clauses = []
    for a, b in zip(cuts, cuts[1:]):
        clause = question[a:b].strip(" ,.")
        clause = re.sub(r"^(?:ve|and)\s+", "", clause, flags=re.IGNORECASE)
        if clause:
            clauses.append(clause)
    return clauses


# Characters that appear almost exclusively in Turkish among this corpus's
# languages (İ/ı, ğ, ş — unlike ç/ö/ü, which also occur in German/French
# loanwords). Used as a last-resort jurisdiction cue in
# _detect_required_regulations: a Turkish-language question that names no
# regulation and hits none of the KVKK/DORA context-word cues above still
# needs *some* signal, or retrieval runs unfiltered and English-language
# GDPR/DORA content — the majority of the corpus by volume — silently
# answers a KVKK-only question (observed: a Turkish question about breach-
# notification content to affected individuals was answered entirely from
# GDPR Art 34(2), with the real, indexed KVKK source — Kurul Kararı
# 2019/271 — never used). Every Turkish document in this corpus is KVKK;
# every DORA/GDPR document is English — so this is a coarse but
# well-founded default, not a guess.
_TURKISH_CHARS_RE = re.compile(r"[ığşİĞŞ]")

# Turkish users on English keyboards routinely type without diacritics
# ("veri ihlali bildirimi ne zaman yapilir") — no character in
# _TURKISH_CHARS_RE ever appears, and the KVKK last-resort default in
# _detect_required_regulations silently never fired for exactly the
# questions it exists for. These are function/domain words that are
# unambiguously Turkish even in ASCII form (none is an English word),
# so matching any of them is as strong a language signal as a special
# character.
#
# "madde\w*" (article/clause) and the mi/mı/mu/mü question-particle
# pattern were both added after a live failure: "KVKK 1. Maddeyi bana
# gosterebilirmisin" (no diacritics, and none of the words above) hit
# none of this regex's original terms, so _language_directive (which
# reuses this same detector via _looks_turkish) concluded the question
# was English and answered a live Turkish question about KVKK Article 1
# entirely in English. "madde" is arguably the single most likely word
# in this domain to appear in a diacritic-free Turkish question and
# was missing outright; the question-particle suffix ("...gösterebilir
# misin", "...var mı", "...yapabilir misiniz") is a grammatical marker
# with no English equivalent at all, so it generalizes far beyond this
# one domain-word list.
_TURKISH_ASCII_HINT_RE = re.compile(
    r"\b(nedir|nelerdir|hangi|gerekir|gerekiyor|zorunlu|yukumlu\w*|"
    r"kisisel|sorumlusu|ihlali|bildirim\w*|kanun\w*|sayili|mevzuat\w*|"
    r"madde\w*)\b|m[iıuü]\s?s[iıuü]n\w*\b",
    re.IGNORECASE)


# Corpus language per regulation, from config. Used to tell whether the
# retrieval query is in the same language as the text it will be matched
# against — see _retrieval_degraded().
_REG_LANGUAGE = {r: (meta.get("language") or "en")
                 for r, meta in config.REGULATIONS.items()}


def _looks_turkish(text: str) -> bool:
    """Is this text Turkish? Used to verify a query actually got
    translated, and to detect a degraded search.

    Deliberately NOT _TURKISH_CHARS_RE, which is narrowed to ı/ğ/ş
    because ç/ö/ü also occur in German/French loanwords — a sensible
    guard when guessing JURISDICTION from a long question, but wrong
    here: "veri ihlali bildirim süresi" contains none of ı/ğ/ş and
    sailed through as English. For a short retrieval query the broad
    diacritic set plus the ASCII word hints is the right test, since a
    correctly-translated English query should contain neither. A false
    positive (an English query naming "Zürich") costs only a retry —
    and the candidate is kept either way, see _retrieval_query.
    """
    return bool(_CLAIM_TR_RE.search(text)
                or _TURKISH_ASCII_HINT_RE.search(text))


def _language_directive(question: str) -> str:
    """An explicit, literal reminder of which language the final answer
    must be written in, meant to be appended as the very LAST thing in
    the prompt before generation.

    SYSTEM_PROMPT rule 6 already states this once ("answer in the
    language the question was asked in"), and for a Turkish scenario
    that alone proved unreliable — a Turkish-scenario decomposed answer
    kept drifting into English until a literal Turkish directive was
    added as the last text before generation (see _decomposed_prompt's
    former Turkish-only version of this same fix). The reverse failure
    is just as real: a plain English question (case G1: "What
    information must a controller include...", zero Turkish content
    anywhere in its retrieved sources) came back entirely in Turkish on
    roughly half of repeated live runs. Rule 6 stated once, mid-prompt,
    isn't enough either direction — this makes the reinforcement
    symmetric and applies it to every final-answer prompt, not just the
    decomposed-scenario path that first needed it.
    """
    if _looks_turkish(question):
        return "\n\nCEVABINIZIN TAMAMINI TÜRKÇE YAZIN."
    return "\n\nAnswer entirely in English."


def _retrieval_degraded(query: str, regulations) -> bool:
    """True when the retrieval query is still in Turkish but the corpus it
    will be searched against is English.

    This is the observable signature of a failed query rewrite. The rewrite
    (QUERY_REWRITE_PROMPT) is what lets a Turkish question reach the
    English-language instruments at all: measured against this corpus, an
    unrewritten Turkish query returns ZERO BM25 chunks from GDPR, so the
    search silently collapses. Before this check, that was indistinguishable
    from "the corpus genuinely says nothing" — and for a compliance tool,
    reporting a broken search as "this is not regulated" is the more
    dangerous of the two errors by a wide margin.

    `regulations` may be a single regulation, an iterable of them, or None
    (unfiltered, which can reach the English instruments).
    """
    if not _looks_turkish(query):
        return False          # query is (or became) English — nothing to flag
    if regulations is None:
        return True           # unfiltered search spans the English corpora
    if isinstance(regulations, str):
        regulations = [regulations]
    return any(_REG_LANGUAGE.get(r, "en") == "en" for r in regulations)


def _degraded_note(query: str, regulations) -> str:
    """Prompt preamble used when _retrieval_degraded() fires, so the model
    reports an incomplete SEARCH rather than an absent OBLIGATION."""
    if not _retrieval_degraded(query, regulations):
        return ""
    return ("Retrieval note: this question could not be translated into the "
            "language of the indexed sources, so the search below may have "
            "missed relevant provisions. If the sources do not cover the "
            "question, say that the search may have been incomplete — do "
            "NOT state that the regulation does not address it.\n\n")


def _kvkk_native_query(text: str, regulation: str | None) -> str | None:
    """The original Turkish text, to search alongside the English-
    translated retrieval query via store.search()'s extra_query — only
    when it's likely to help: the target regulation is KVKK (a Turkish-
    only corpus) and the text itself is in Turkish. See search()'s
    extra_query docstring for the concrete evidence (KVKK Article 5,
    retrievable instantly by its own wording, never surfaced by any
    English rephrasing). Returns None otherwise, so callers can pass the
    result straight through as extra_query without a separate branch.
    """
    if regulation == "KVKK" and _TURKISH_CHARS_RE.search(text):
        return text
    return None


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
    # Last resort, and deliberately conditioned on no other signal firing
    # at all: see _TURKISH_CHARS_RE's comment. The ASCII-hint fallback
    # covers Turkish typed without diacritics (see
    # _TURKISH_ASCII_HINT_RE).
    if (not required and "KVKK" in config.REGULATIONS
            and (_TURKISH_CHARS_RE.search(question)
                 or _TURKISH_ASCII_HINT_RE.search(question))):
        required.add("KVKK")
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

# Every document in this corpus, and both languages the model ever
# legitimately answers in (Turkish, English), are pure Latin script — CJK
# characters can only ever be a generation artifact, not real content
# (observed: a Turkish answer with a Chinese term — "合规审查" — spliced
# mid-sentence, immediately followed by its own English gloss, as if the
# model briefly reached for the wrong vocabulary and self-corrected
# without deleting the original). Stripped defensively rather than
# investigated further: the cause is inside the hosted model's decoding,
# not this pipeline's code, and unlike a wrong citation there's no
# grounding signal available to catch this any other way.
_UNEXPECTED_SCRIPT_RE = re.compile(
    r"[一-鿿㐀-䶿豈-﫿"  # CJK unified + ext-A + compat
    r"぀-ヿ가-힣]+")  # hiragana/katakana, hangul


def _strip_unexpected_script(text: str) -> str:
    cleaned = _UNEXPECTED_SCRIPT_RE.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", cleaned)


def _fill_blank_table_cells(text: str) -> str:
    """Put an em dash in any empty markdown-table cell.

    Stripping an ungrounded marker can empty a cell outright — a cell
    whose entire content was "[3]" becomes "|  |". COMPARE_TEMPLATE
    explicitly tells the model never to leave a column blank (a blank
    cell reads as "no requirement here", which is a substantive claim
    nobody made), so the pipeline must not manufacture one either. Also
    catches a blank the model itself left. Only touches lines that are
    already table rows, and never the "|---|---|" separator (its cells
    contain dashes, not whitespace).
    """
    if "|" not in text:
        return text
    out = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.count("|") >= 2:
            cells = stripped.split("|")
            # first/last entries are the empty strings outside the outer
            # pipes — leave them alone, they are not cells.
            for i in range(1, len(cells) - 1):
                if not cells[i].strip():
                    cells[i] = " — "
            line = "|".join(cells)
        out.append(line)
    return "\n".join(out)


def _content_words(text: str) -> set[str]:
    return {w for w in (m.lower() for m in _WORD_RE.findall(text))
           if w not in _STOPWORDS}


def _marker_claim(text: str, start: int, end: int) -> str:
    """The sentence around a [n] marker's position in `text` — from the
    previous sentence boundary to the next — i.e. the claim the marker is
    actually attached to, not the whole answer.

    "|" counts as a boundary so that a marker inside a markdown table
    cell is attributed to THAT CELL rather than the whole row. Compare
    mode answers are tables by construction (COMPARE_TEMPLATE), and a
    table row contains both regulations' cells with no sentence
    punctuation between them — so without this, a citation in the KVKK
    column was checked for grounding against a "claim" that also
    contained the entire GDPR column. Measured on a realistic row, that
    diluted a cleanly-grounded citation's overlap ratio from 0.75 to
    0.23 (floor 0.12) — margin that disappears entirely once the
    opposing cell quotes a long provision verbatim, which is exactly
    what the template asks for.
    """
    left = max(text.rfind(".", 0, start), text.rfind("!", 0, start),
              text.rfind("?", 0, start), text.rfind("\n", 0, start),
              text.rfind("|", 0, start))
    right_candidates = [p for p in (
        text.find(".", end), text.find("!", end),
        text.find("?", end), text.find("\n", end),
        text.find("|", end)) if p != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1:right + 1]


# Any Turkish-specific letter (broader than _TURKISH_CHARS_RE: ç/ö/ü
# included, since here we're classifying the ANSWER's language, not
# guessing jurisdiction) — used to decide whether a lexical-overlap
# grounding check is even meaningful for a claim/source pair.
_CLAIM_TR_RE = re.compile(r"[çğıöşüÇĞİÖŞÜ]")


def _is_grounded(claim: str, source_text: str, source_lang: str = "") -> bool:
    # Lexical overlap is meaningless across languages: an English answer
    # correctly citing a Turkish KVKK chunk (or a Turkish answer citing
    # an English GDPR/DORA chunk) shares almost no content words with it,
    # so this check used to strip perfectly valid citations wholesale for
    # every cross-language question — the sources panel then showed few
    # or no sources even though retrieval and the citations were right.
    # When the claim's language differs from the chunk's indexed
    # language, keep the citation (marker-range validation still applies)
    # rather than false-stripping it. Errs toward keeping: a diacritic-
    # less Turkish claim classifies as "en", which can only ever skip the
    # check, never wrongly strip.
    if source_lang in ("tr", "en"):
        claim_lang = "tr" if _CLAIM_TR_RE.search(claim) else "en"
        if claim_lang != source_lang:
            return True
    claim_words = _content_words(claim)
    if not claim_words:
        return True  # nothing substantive in the claim to check
    overlap = claim_words & _content_words(source_text)
    return len(overlap) / len(claim_words) >= _MIN_CITATION_OVERLAP


def _flag_unverified_prose_articles(answer: str, results: list[SearchResult]) -> str:
    """Neutralise a plain-prose "<Regulation> Article N" mention whose
    (regulation, article) pair matches none of the actually-retrieved
    results — see _PROSE_ARTICLE_RE's comment for why this class of
    claim needs its own check, separate from the [n]-marker grounding
    cited_sources already does. A real source the model just forgot to
    bracket is left untouched: this only fires when the article wasn't
    retrieved at all, never merely because it's uncited.
    """
    # Only the base statute counts. A prose "DORA Article N" / "KVKK
    # Madde N" claim refers to the base statute — an RTS/ITS/delegated
    # act is a separate legal instrument that happens to have its own
    # Article 1, 2, 3... and is cited by its own label ("Commission
    # Delegated Regulation (EU) 2025/301"), which _PROSE_ARTICLE_RE
    # deliberately never matches. Counting RTS/ITS article numbers here
    # (as a previous version did) let a hallucinated statute reference
    # pass verification whenever a sub-document's same-numbered article
    # happened to be retrieved — exactly the cross-instrument number
    # collision SYSTEM_PROMPT rule 8 warns the model about. Board
    # decisions/guidelines/notes are excluded for the same reason as
    # before: their ad-hoc numeric labels can coincide with statute
    # article numbers purely by chance.
    available = {(r.metadata["regulation"], str(r.metadata.get("article_number")))
                for r in results
                if r.metadata.get("doc_type", "statute") == "statute"}

    def _check(m: re.Match) -> str:
        reg, num = m.group(1).upper(), m.group(2)
        if (reg, num) in available:
            return m.group(0)
        # Match the language of the mention itself ("Madde" -> Turkish),
        # the same way _out_of_range_article does: this substitution
        # bypasses the LLM, so a fixed English string used to appear
        # mid-sentence inside an otherwise Turkish answer, breaking
        # SYSTEM_PROMPT rule 6 (one language, start to finish). Kept
        # short because in compare mode it lands inside a table cell,
        # where a long parenthetical wrecks the column layout.
        if "madde" in m.group(0).lower():
            return f"{reg} (madde numarası kaynaklarda doğrulanamadı)"
        return f"{reg} (article number not confirmed in the sources)"

    return _PROSE_ARTICLE_RE.sub(_check, answer)


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
    answer = _strip_unexpected_script(answer)
    answer = _flag_unverified_prose_articles(answer, results)
    valid_max = len(results)

    def _strip_ungrounded(m: re.Match) -> str:
        n = int(m.group(1))
        if not (1 <= n <= valid_max):
            return ""
        claim = _marker_claim(m.string, m.start(), m.end())
        source = results[n - 1]
        return m.group(0) if _is_grounded(
            claim, source.text, source.metadata.get("language", "")) else ""

    clean_answer = _CITATION_RE.sub(_strip_ungrounded, answer)
    # Tidy up connective words/punctuation left dangling by a removed
    # marker, e.g. "...by [3] and [6]." -> "...by [3] and ." -> "...[3]."
    clean_answer = re.sub(r"\s+and\s*([.,;:]|$)", r"\1", clean_answer)
    clean_answer = re.sub(r",\s*([.,;:])", r"\1", clean_answer)
    clean_answer = re.sub(r" {2,}", " ", clean_answer)
    clean_answer = re.sub(r" ([.,;:])", r"\1", clean_answer)
    clean_answer = _fill_blank_table_cells(clean_answer)

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
        # document_label is part of the identity: DORA's statute Article 5
        # and an RTS's own Article 5 are different instruments that merely
        # share a number — keying on (regulation, article_number) alone
        # used to merge them into one group and display one document's
        # text under the other's citation label.
        key = (m["regulation"], m.get("document_label") or "", m["article_number"])
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
        """Rewrite the question into a short English search query, then
        apply _augment_frequency_query — see that function's comment
        for why a "how often" question needs one further, targeted
        nudge beyond the LLM rewrite alone.
        """
        return _augment_frequency_query(self._rewrite_query(question), question)

    def _rewrite_query(self, question: str) -> str:
        """Rewrite the question into a short English search query.

        The EU instruments are indexed in English, so for a Turkish
        question this call is not a nicety — it is the only thing that
        makes GDPR/DORA reachable at all (an unrewritten Turkish query
        returns zero BM25 chunks from the English corpora). It is
        therefore hardened three ways:

          - Retried once. A single transient API error used to silently
            downgrade a whole answer.
          - Validated. The prompt asks for English, but compliance is not
            guaranteed (this project's own notes record smaller models
            transliterating rather than translating). A "rewrite" that is
            still Turkish for a Turkish question has not done its job, so
            it is retried rather than accepted.
          - Reported. If it still fails, the raw question is returned as
            before, but callers detect that via _retrieval_degraded() and
            tell the model the search was incomplete, so a failed
            translation can never be presented to the user as "this
            regulation does not cover it".
        """
        if isinstance(self.llm, EchoLLM):
            return question
        asked_in_turkish = _looks_turkish(question)
        candidate: str | None = None
        for attempt in range(2):
            try:
                q = self.llm.chat(prompts.QUERY_REWRITE_PROMPT, question).strip()
            except Exception:
                logger.warning("query rewrite failed (attempt %d/2)", attempt + 1)
                continue
            if not q or q == REASONING_EXHAUSTED_MESSAGE:
                # The latter happens even with reasoning=False requested
                # (see REASONING_EXHAUSTED_MESSAGE's docstring) — treated
                # as "empty" rather than a usable query, since parsing it
                # as one silently sent this placeholder sentence to
                # retrieval as if it were the actual search text.
                logger.warning("query rewrite returned empty (attempt %d/2)",
                              attempt + 1)
                continue
            q = _strip_regulation_names(q[:300])
            # A Turkish question whose "translation" is still Turkish means
            # the model transliterated or echoed instead of translating.
            if asked_in_turkish and _looks_turkish(q):
                logger.warning("query rewrite did not translate out of Turkish "
                              "(attempt %d/2): %r", attempt + 1, q)
                candidate = candidate or q   # keep it; still better than raw
                continue
            return q
        if candidate:
            # Every attempt produced something, just not convincingly
            # translated. Return the best one anyway rather than the raw
            # question: it is at least a shortened, topic-focused query,
            # and _retrieval_degraded() will flag it if it really is
            # still Turkish. Discarding it would throw away a usable
            # rewrite whenever the language check misfires (a proper noun
            # like "Zürich" in an otherwise English query).
            logger.warning("query rewrite never confirmed as translated; using "
                          "best candidate %r", candidate)
            return candidate
        logger.error("query rewrite exhausted retries; falling back to the raw "
                    "question — retrieval against English-language sources "
                    "will be degraded and callers will flag it")
        return question

    def _detect_required_regulations_smart(self, question: str) -> set[str]:
        """Like the module-level _detect_required_regulations, but falls
        back to an LLM classification pass (REGULATION_TRIAGE_PROMPT)
        when the keyword-cue regex finds nothing — the common case for a
        realistic "does what I'm doing touch any of these" question,
        which essentially never uses the regex's specific trigger phrases
        (see that prompt's docstring). The regex stays the fast, free
        first attempt; this only spends an LLM call when it comes back
        empty. Deliberately NOT used by _needs_decomposition — that
        trigger is a separate, already-tuned decision (a prior fix in
        this project's history addressed it firing wrongly on short
        comparative questions) and this method only affects which
        regulation(s) retrieval filters by, not whether to decompose.
        """
        required = _detect_required_regulations(question)
        # The LLM fallback below only fires above
        # SCENARIO_MIN_LENGTH_FOR_MULTI_REG. REGULATION_TRIAGE_PROMPT's
        # own contract is "a real-world activity, situation, or plan
        # (not a legal question...)", and a short, bare legal question
        # fed to it anyway gets triaged on subject-matter overlap
        # rather than jurisdiction, which is exactly wrong for e.g.
        # "what must a breach notification contain and by when" — GDPR
        # and KVKK both substantively cover breach notification, so the
        # triage correctly-by-its-own-rules answers "both", and the
        # model then appends an unsolicited KVKK aside to a plain GDPR
        # question (cases G1/G2: cited a KVKK Kurul Kararı on a GDPR-
        # only question, failing citations_from_only). A real scenario
        # long enough to need this classifier ("I run a payments
        # startup in Istanbul...") doesn't have that problem — its
        # jurisdiction lives in concrete detail the triage prompt is
        # actually built to read.
        if (not required and not isinstance(self.llm, EchoLLM)
                and len(question) >= config.SCENARIO_MIN_LENGTH_FOR_MULTI_REG):
            try:
                raw = self.llm.chat(prompts.REGULATION_TRIAGE_PROMPT, question).strip().upper()
                if raw != REASONING_EXHAUSTED_MESSAGE.upper():
                    required |= {r for r in config.REGULATIONS if r in raw}
            except Exception:
                pass
        # Runs unconditionally, even when `required` is already non-empty:
        # an early cue (e.g. "Türkiye") used to short-circuit this whole
        # method before the LLM path ever ran, silently locking out any
        # other regulation for the rest of the call — observed concretely:
        # a Turkish bank question naming both Türkiye AND an explicit EU
        # mention got KVKK-only, because the Turkish cue alone already
        # satisfied the old "if required: return" check before GDPR had
        # any chance to be considered. This deterministic, additive-only
        # check can't cause that kind of silent lockout — it only ever
        # adds GDPR on top of whatever was already found.
        if "GDPR" in config.REGULATIONS and _GDPR_CONTEXT_RE.search(question):
            required.add("GDPR")
        # Same reasoning, same fix, for DORA: a scenario naming a bank
        # AND an EU/Turkey jurisdiction cue used to lock onto whichever
        # jurisdiction cue fired first and never even consider DORA —
        # observed on eval case S3 (a Turkish bank's Amsterdam branch
        # and Ireland-based SaaS vendor), where DORA never entered
        # consideration despite the scenario needing it for the bank's
        # own ICT third-party arrangement.
        if "DORA" in config.REGULATIONS and _DORA_CONTEXT_RE.search(question):
            required.add("DORA")
        return required

    def _ask_prompt(self, question: str, regulation: str | None,
                     top_k: int) -> tuple[list[SearchResult], str]:
        if self._needs_decomposition(question):
            # Computed once here and threaded through to both the
            # decompose call (as a coverage hint) and _decomposed_prompt
            # (to avoid a second, redundant classifier call) — see
            # _decompose_question's docstring for why the hint matters.
            required_hint = None if regulation else self._detect_required_regulations_smart(question)
            sub_questions = self._decompose_question(question, required_hint)
            if len(sub_questions) > 1:
                return self._decomposed_prompt(question, sub_questions, regulation, top_k,
                                               required_hint)

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
            # crowded out of the RRF ranking entirely. The "smart" variant
            # adds an LLM classification fallback for realistic questions
            # that name no regulation and hit none of the keyword cues
            # either (see _detect_required_regulations_smart).
            required = self._detect_required_regulations_smart(question)
            if len(required) >= 2:
                results = self._multi_regulation_search(question, query, required, top_k)
                scope_note = "".join(self._scope_note(r) for r in sorted(required))
                prompt = prompts.ANSWER_TEMPLATE.format(
                    scope_note=_degraded_note(query, required) + scope_note,
                    sources=prompts.format_sources(results), question=question)
                return results, prompt + _language_directive(question)
            if len(required) == 1:
                regulation = next(iter(required))

        if regulation is not None:
            clauses = _compound_clauses(question)
            if len(clauses) >= 2:
                return self._compound_ask_prompt(question, clauses, regulation, top_k)

        results = self.store.search(query, top_k=top_k, regulation=regulation,
                                    extra_query=_kvkk_native_query(question, regulation))
        prompt = prompts.ANSWER_TEMPLATE.format(
            scope_note=_degraded_note(query, regulation) + self._scope_note(regulation),
            sources=prompts.format_sources(results), question=question)
        return results, prompt + _language_directive(question)

    def _compound_ask_prompt(self, question: str, clauses: list[str],
                              regulation: str, top_k: int
                              ) -> tuple[list[SearchResult], str]:
        """Retrieval runs once per clause a compound direct question was
        split into (see _compound_clauses) and results are merged, deduped
        by chunk id — same merge pattern as _decomposed_prompt, but for a
        single regulation and without an LLM decompose call. The prompt
        still presents the ORIGINAL, unsplit question — the clause split
        only shapes retrieval, not how the model is asked to answer.
        """
        per_clause_k = max(top_k // len(clauses), 4)
        results: list[SearchResult] = []
        seen_ids: set[str] = set()
        clause_queries: list[str] = []
        for clause in clauses:
            clause_query = self._retrieval_query(clause)
            clause_queries.append(clause_query)
            for r in self.store.search(clause_query, top_k=per_clause_k, regulation=regulation,
                                       extra_query=_kvkk_native_query(clause, regulation)):
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    results.append(r)
        prompt = prompts.ANSWER_TEMPLATE.format(
            scope_note=(_degraded_note(" ".join(clause_queries), regulation)
                        + self._scope_note(regulation)),
            sources=prompts.format_sources(results), question=question)
        return results, prompt + _language_directive(question)

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

    def _decompose_question(self, question: str,
                            required_hint: set[str] | None = None) -> list[str]:
        """Extract discrete legal sub-questions via one LLM call. Falls
        back to [question] (i.e. "don't decompose") for EchoLLM or if the
        call fails — same fallback pattern as _retrieval_query, and the
        caller (_ask_prompt) already treats a single-item result as "no
        decomposition happened" and continues down the normal path.

        required_hint, when it names 2+ regulations, is appended as an
        explicit coverage instruction: left to its own judgment, this
        extraction step can produce sub-questions that all gravitate to
        one instrument and never name the others at all — observed on
        eval case S3 (a Turkish bank, an Amsterdam/GDPR branch, and an
        Ireland-based vendor implicating DORA): every extracted sub-
        question came back GDPR-flavoured, and KVKK/DORA were never
        named by any of them, even though the scenario-level classifier
        correctly flagged all three as required. Because
        _decomposed_prompt's per-sub-question retrieval routes primarily
        by what each sub-question explicitly names, a regulation missing
        from every sub-question here has nowhere to be discussed later —
        _decomposed_prompt's own supplementary retrieval pass covers the
        retrieval side of this, but making the issues themselves name
        each regulation is the more direct fix, so the model's answer
        structure actually has a place to put that regulation's content.
        """
        if isinstance(self.llm, EchoLLM):
            return [question]
        prompt = prompts.DECOMPOSE_PROMPT
        if required_hint and len(required_hint) >= 2:
            prompt += (f"\n\nThis scenario has already been identified as plausibly "
                      f"touching all of: {', '.join(sorted(required_hint))}. Make sure "
                      f"your extracted sub-questions collectively include at least one "
                      f"that clearly names or unambiguously concerns each of these "
                      f"instruments — do not let the extraction gravitate to only the "
                      f"most prominent one or two.")
        try:
            raw = self.llm.chat(prompt, question)
        except Exception:
            return [question]
        if raw == REASONING_EXHAUSTED_MESSAGE:
            # Same failure mode as _retrieval_query's — reasoning=False
            # is requested but not a hard guarantee (see the constant's
            # docstring) — treated as "decomposition failed" rather
            # than letting this placeholder sentence become the sole
            # (garbled) "sub-question" the rest of the pipeline answers.
            return [question]
        sub_qs = [_SUBQ_PREFIX_RE.sub("", line).strip() for line in raw.splitlines()]
        sub_qs = [q for q in sub_qs if len(q) > 8]  # drop stray blank/junk lines
        return sub_qs[:5] if sub_qs else [question]

    def _decomposed_prompt(self, question: str, sub_questions: list[str],
                            regulation: str | None, top_k: int,
                            required_hint: set[str] | None = None
                            ) -> tuple[list[SearchResult], str]:
        """Retrieval runs once per extracted issue and results are
        merged (deduped by chunk id). `regulation` is the caller's
        explicit filter, if any (e.g. the UI's regulation dropdown), and
        is respected for every sub-question; when unset, each
        sub-question detects its own regulation independently — a
        multi-issue scenario often has different issues governed by
        different instruments (see _decompose_question's docstring).
        """
        # Floor of 4, not 3: a per-issue retrieval this narrow can rank the
        # actually-correct chunk just past the cutoff even when it's a
        # clean, unambiguous match (observed: a processor's exact
        # notification duty ranked #4 for its sub-question, invisible at
        # k=3 but included at k=4).
        per_issue_k = max(top_k // len(sub_questions), 4)
        # A scenario's jurisdiction cues (a Turkish city, "Kurul", Turkish
        # characters) live in the original question, but DECOMPOSE_PROMPT's
        # sub-questions often come back paraphrased or translated to
        # English, losing that signal — a sub-question then detects no
        # regulation of its own and retrieval runs unfiltered, letting an
        # unrelated instrument's same-language content leak in (observed:
        # a Turkish-only KVKK scenario pulling in GDPR articles for the
        # sub-questions that lost their Turkish cues in translation). Only
        # used when the scenario as a whole implies exactly one
        # regulation — a genuinely multi-instrument scenario still lets
        # each sub-question detect its own, per this method's docstring.
        # required_hint reuses the caller's already-computed classifier
        # result (see _ask_prompt) rather than calling
        # _detect_required_regulations_smart a second time.
        scenario_required = None if regulation else (
            required_hint if required_hint is not None
            else self._detect_required_regulations_smart(question))
        scenario_reg = (next(iter(scenario_required))
                        if scenario_required and len(scenario_required) == 1 else None)
        results: list[SearchResult] = []
        seen_ids: set[str] = set()
        regs_covered: set[str] = set()
        sub_queries: list[str] = []
        for sub_q in sub_questions:
            sub_reg = regulation or _detect_single_regulation(sub_q) or scenario_reg
            sub_query = self._retrieval_query(sub_q)
            sub_queries.append(sub_query)
            # sub_q itself is often already translated away from Turkish
            # (see this method's docstring on lost jurisdiction cues), so
            # the native-language fallback checks the ORIGINAL scenario
            # text, not sub_q — that's where the Turkish signal survives.
            extra = _kvkk_native_query(sub_q, sub_reg) or _kvkk_native_query(question, sub_reg)
            for r in self.store.search(sub_query, top_k=per_issue_k, regulation=sub_reg,
                                       extra_query=extra):
                if r.id not in seen_ids:
                    seen_ids.add(r.id)
                    results.append(r)
            if sub_reg:
                regs_covered.add(sub_reg)

        # A regulation the scenario-level classifier confidently flagged
        # can still end up with zero representation here: each
        # sub-question only routes to a regulation it explicitly names
        # (_detect_single_regulation), and DECOMPOSE_PROMPT's own
        # extraction can simply never produce a sub-question that names
        # one of the genuinely-relevant instruments (case S3: every
        # extracted sub-question came back GDPR-only, KVKK and DORA
        # named by none of them, despite the classifier correctly
        # flagging all three) — the required_hint coverage instruction
        # in _decompose_question reduces how often this happens, but
        # doesn't guarantee it, so this is the retrieval-side backstop:
        # a regulation the classifier flagged always gets at least one
        # retrieval pass, using the original scenario text rather than
        # trusting sub-question phrasing alone. Mirrors
        # _multi_regulation_search's guarantee for the non-decomposed
        # path.
        if scenario_required:
            missing = scenario_required - regs_covered
            if missing:
                scenario_query = self._retrieval_query(question)
                for reg in sorted(missing):
                    for r in self.store.search(scenario_query, top_k=per_issue_k, regulation=reg,
                                               extra_query=_kvkk_native_query(question, reg)):
                        if r.id not in seen_ids:
                            seen_ids.add(r.id)
                            results.append(r)
                    regs_covered.add(reg)

        scope_note = (_degraded_note(" ".join(sub_queries), regs_covered or None)
                      + _home_destination_note(question)
                      + "".join(self._scope_note(r) for r in sorted(regs_covered)))
        issues = "\n".join(f"- {q}" for q in sub_questions)
        prompt = prompts.SCENARIO_ANSWER_TEMPLATE.format(
            scope_note=scope_note, sources=prompts.format_sources(results),
            question=question, issues=issues)
        # Sub-questions are frequently extracted in a DIFFERENT language
        # than the scenario itself (DECOMPOSE_PROMPT's own instruction
        # notwithstanding) — SCENARIO_ANSWER_TEMPLATE already warns about
        # this, but a literal directive keyed off the ORIGINAL question,
        # as the very last text before generation, is what actually
        # held reliably here (see _language_directive's docstring).
        return results, prompt + _language_directive(question)

    def _multi_regulation_search(self, question: str, query: str, regulations: set[str],
                                  top_k: int) -> list[SearchResult]:
        """Search each required regulation separately and merge, so every
        regulation _detect_required_regulations() flagged actually
        contributes chunks — mirrors _compare_prompt's per-regulation
        retrieval, generalized to N regulations from context cues rather
        than the two explicit reg_a/reg_b of comparison mode.
        """
        per_reg_k = max(top_k // len(regulations), 4)
        results: list[SearchResult] = []
        seen_ids: set[str] = set()
        for reg in sorted(regulations):
            for r in self.store.search(query, top_k=per_reg_k, regulation=reg,
                                       extra_query=_kvkk_native_query(question, reg)):
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
        named = _detect_single_regulation(question)
        # The question names a different instrument than the caller's
        # filter (e.g. UI filter set to KVKK, question about "GDPR
        # Article 45") — a deterministic denial would be checked against
        # the wrong instrument's range, so let retrieval handle it.
        if regulation and named and named != regulation:
            return None
        # An "Article N" mention alongside a document number ("Article 5
        # of 2025/301...") refers to that sub-document's own numbering,
        # not the base statute's — never range-check it against the
        # statute.
        if re.search(r"\b\d{4}/\d{1,5}\b", question):
            return None
        reg = regulation or named
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
        answer = self.llm.chat(prompts.SYSTEM_PROMPT, prompt, reasoning=True)
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
        return results, self.llm.stream_chat(prompts.SYSTEM_PROMPT, prompt, reasoning=True)

    def _same_regulation_notice(self, question: str, reg: str) -> str:
        """Message for a compare call whose two sides are the same
        instrument. Retrieval would run the identical filtered search
        twice and hand the model every chunk TWICE under two different
        source numbers — inviting it to "compare" an instrument with
        itself and present duplicated text as two differing positions.
        Bypasses the LLM, so it picks its own language (cf.
        _out_of_range_article).
        """
        if _TURKISH_CHARS_RE.search(question) or _TURKISH_ASCII_HINT_RE.search(question):
            return (f"Karşılaştırma için iki farklı mevzuat seçin — her iki "
                    f"tarafta da {reg} seçili.")
        return (f"Select two different regulations to compare — {reg} is "
                f"selected on both sides.")

    def _compare_prompt(self, question: str, reg_a: str, reg_b: str,
                         top_k_each: int) -> tuple[list[SearchResult], str]:
        """Retrieval runs separately per regulation (metadata-filtered) so
        both sides are actually represented in the context — a single
        unfiltered search often returns chunks from only one framework.
        """
        query = self._retrieval_query(question)
        results_a = self.store.search(query, top_k=top_k_each, regulation=reg_a,
                                      extra_query=_kvkk_native_query(question, reg_a))
        results_b = self.store.search(query, top_k=top_k_each, regulation=reg_b,
                                      extra_query=_kvkk_native_query(question, reg_b))
        prompt = prompts.COMPARE_TEMPLATE.format(
            scope_note=(_degraded_note(query, [reg_a, reg_b])
                        + self._scope_note(reg_a) + self._scope_note(reg_b)),
            reg_a=reg_a, reg_b=reg_b,
            sources_a=prompts.format_sources(results_a),
            sources_b=prompts.format_sources(results_b, start=len(results_a) + 1),
            question=question,
        )
        return results_a + results_b, prompt + _language_directive(question)

    def compare(self, question: str, reg_a: str, reg_b: str,
                top_k_each: int = config.DEFAULT_TOP_K) -> RagResponse:
        """Compare two regulations on a topic."""
        if reg_a == reg_b:
            return RagResponse(answer=self._same_regulation_notice(question, reg_a))
        results, prompt = self._compare_prompt(question, reg_a, reg_b, top_k_each)
        answer = self.llm.chat(prompts.SYSTEM_PROMPT, prompt, reasoning=True)
        answer, indices, sources = cited_sources(answer, results)
        return RagResponse(answer=answer, sources=sources, cited_indices=indices)

    def compare_stream(self, question: str, reg_a: str, reg_b: str,
                        top_k_each: int = config.DEFAULT_TOP_K
                        ) -> tuple[list[SearchResult], Iterator[str]]:
        """Like compare(), but returns the raw sources plus a token
        generator — see ask_stream()'s note on applying cited_sources()."""
        if reg_a == reg_b:
            return [], iter([self._same_regulation_notice(question, reg_a)])
        results, prompt = self._compare_prompt(question, reg_a, reg_b, top_k_each)
        return results, self.llm.stream_chat(prompts.SYSTEM_PROMPT, prompt, reasoning=True)
