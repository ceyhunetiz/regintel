"""Regression tests for the 2026-08-22 bug-audit fixes.

Each test pins one fixed behavior from docs/bug-audit-2026-08-22.md, so
none of these bugs can silently return. Runs without embeddings,
network, or an LLM — same constraints as test_pipeline.py.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from regintel.generation.llm import _stream_strip_thinking
from regintel.generation.rag import (_compound_clauses, _is_grounded,
                                     _flag_unverified_prose_articles,
                                     _detect_required_regulations,
                                     _GDPR_CONTEXT_RE,
                                     cited_sources, group_sources)
from regintel.ingestion.chunker import chunk_articles
from regintel.ingestion.parser import parse_plain_text
from regintel.retrieval.store import RegulationStore, SearchResult, _tokenize


def _sr(text, reg="MOCK", art="1", doc_type="statute", document_label="",
        language="en", title=""):
    return SearchResult(id=f"{reg}-{art}-{document_label}", text=text, score=1.0,
                        metadata={
        "regulation": reg, "article_number": art, "article_title": title,
        "chapter": "", "chunk_index": 0, "total_chunks": 1, "source_url": "",
        "language": language, "doc_type": doc_type, "doc_date": "",
        "in_force": True, "document_label": document_label})


# --- Bug 1: cross-lingual citation grounding ---------------------------------

def test_grounding_keeps_cross_lingual_citations():
    """Lexical overlap is meaningless across languages: an English answer
    correctly citing a Turkish KVKK chunk must keep its citation."""
    claim_en = ("The controller must notify the Board within 72 hours at "
                "the latest after learning of the breach [1].")
    source_tr = ("Veri ihlalinin öğrenilmesinden itibaren gecikmeksizin ve "
                 "en geç 72 saat içinde Kurula bildirimde bulunulması gerekir.")
    assert _is_grounded(claim_en, source_tr, "tr")

    claim_tr = ("Veri sorumlusu ihlali gecikmeksizin ve en geç 72 saat "
                "içinde denetim makamına bildirmelidir [2].")
    source_en = ("the controller shall without undue delay and, where "
                 "feasible, not later than 72 hours after having become "
                 "aware of it, notify the personal data breach to the "
                 "supervisory authority")
    assert _is_grounded(claim_tr, source_en, "en")


def test_grounding_still_strips_same_language_mismatch():
    claim = ("The regulator requires quarterly penetration reports "
             "covering mainframe systems [1].")
    source = ("the controller shall notify the supervisory authority of a "
              "personal data breach without undue delay")
    assert not _is_grounded(claim, source, "en")


def test_cited_sources_keeps_english_answer_over_turkish_source():
    results = [_sr("Veri ihlalinin öğrenilmesinden itibaren en geç 72 saat "
                   "içinde Kurula bildirimde bulunulur.", reg="KVKK",
                   language="tr")]
    answer = ("The breach must be notified to the Board within 72 hours "
              "of becoming aware of it [1].")
    clean, indices, cited = cited_sources(answer, results)
    assert indices == [1]
    assert "[1]" in clean


# --- Bug 2: Turkish İ/I in the BM25 tokenizer --------------------------------

def test_tokenizer_folds_turkish_capital_i():
    indexed = _tokenize("İlgili kişinin BİLDİRİM hakkı")
    query = _tokenize("ilgili kişinin bildirim hakkı")
    assert set(query) <= set(indexed)


def test_bm25_matches_turkish_capitalized_words():
    tmp = tempfile.mkdtemp()
    s = RegulationStore(persist_dir=tmp, use_embeddings=False)
    # Three articles: with fewer, BM25Okapi's IDF is <= 0 for a term in
    # one-of-N docs (ln(1) at N=2) and nothing scores > 0 — the same
    # small-corpus artifact the distractor chunks in test_pipeline.py
    # work around. The extra articles exist only to give IDF room.
    text = ("İlgili kişinin hakları\nMadde 11\nİlgili kişi, veri "
            "sorumlusuna başvurarak BİLDİRİM isteme hakkına sahiptir.\n\n"
            "Madde 28\nÜçüncü taraf sözleşmeleri her yıl gözden "
            "geçirilir ve kayıt altında tutulur.\n\n"
            "Madde 29\nDenetim raporları beş yıl boyunca saklanır ve "
            "talep üzerine kuruma sunulur.\n")
    s.add_chunks(chunk_articles(parse_plain_text(text, "MOCK-TR", language="tr")))
    results = s.search("ilgili kişi bildirim hakkı", top_k=3,
                       regulation="MOCK-TR")
    assert results, "İ-containing Turkish words failed to match a lowercase query"


# --- Bug 3: source grouping collides different instruments -------------------

def test_group_sources_separates_statute_from_rts_same_number():
    results = [
        _sr("statute text", reg="DORA", art="5",
            title="ICT risk management framework"),
        _sr("rts text", reg="DORA", art="5", doc_type="rts",
            document_label="Commission Delegated Regulation (EU) 2025/301"),
    ]
    groups = group_sources([1, 2], results)
    assert len(groups) == 2, "different instruments merged under one citation"
    citations = {g["citation"] for g in groups}
    assert any("2025/301" in c for c in citations)


def test_group_sources_still_merges_same_article_chunks():
    meta_common = dict(reg="DORA", art="19", title="Reporting")
    a = _sr("part 1", **meta_common)
    b = _sr("part 4", **meta_common)
    a.metadata.update(chunk_index=0, total_chunks=7)
    b.metadata.update(chunk_index=3, total_chunks=7)
    groups = group_sources([1, 2], [a, b])
    assert len(groups) == 1
    assert "parts 1, 4 of 7" in groups[0]["citation"]


# --- Bug 4/8: streaming think-leak -------------------------------------------

def test_stream_strip_thinking_removes_reasoning():
    def run(chunks):
        return "".join(_stream_strip_thinking(iter(chunks)))
    assert run(["<think>reasoning</think>Answer [1]."]) == "Answer [1]."
    # tags split across chunk boundaries
    assert run(["<th", "ink>rea", "soning</thi", "nk>Ans", "wer."]) == "Answer."
    assert run(["plain answer, no tags"]) == "plain answer, no tags"
    # unterminated reasoning must never leak
    assert run(["<think>never closed..."]) == ""
    assert run(["before <think>mid</think> after"]) == "before  after"


# --- Bug 5: prose-article verifier must be statute-only ----------------------

def test_prose_article_not_validated_by_rts_number():
    """A hallucinated 'DORA Article 5' (statute) claim must not pass just
    because an RTS's own Article 5 was retrieved."""
    retrieved = [_sr("rts text", reg="DORA", art="5", doc_type="rts",
                     document_label="Commission Delegated Regulation (EU) 2025/301")]
    answer = "Under DORA Article 5, entities must maintain a framework."
    flagged = _flag_unverified_prose_articles(answer, retrieved)
    assert "Article 5" not in flagged
    assert "not confirmed" in flagged


def test_prose_article_validated_by_statute_retrieval():
    retrieved = [_sr("statute text", reg="DORA", art="5")]
    answer = "Under DORA Article 5, entities must maintain a framework."
    assert _flag_unverified_prose_articles(answer, retrieved) == answer


# --- Bug 6: GEÇİCİ MADDE parsing --------------------------------------------

def test_gecici_madde_parsed_as_own_article():
    text = ("Yürürlük\nMADDE 32- (1) Bu Kanun yayımı tarihinde yürürlüğe "
            "girer.\n\nGEÇİCİ MADDE 3- (1) Bu maddeyi ihdas eden Kanunla "
            "yapılan değişiklikler bir yıl içinde uygulanır.\n")
    articles = parse_plain_text(text, "MOCK-TR", language="tr")
    numbers = [a.article_number for a in articles]
    assert "Geçici 3" in numbers, numbers
    gecici = next(a for a in articles if a.article_number == "Geçici 3")
    assert "ihdas eden" in gecici.text
    madde32 = next(a for a in articles if a.article_number == "32")
    assert "ihdas eden" not in madde32.text, \
        "transitional article text leaked into the preceding article"
    # ids must be ascii-safe and collision-free
    chunks = chunk_articles(articles)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(id_.isascii() for id_ in ids)


def test_gecici_madde_excluded_from_article_range():
    tmp = tempfile.mkdtemp()
    s = RegulationStore(persist_dir=tmp, use_embeddings=False)
    text = ("MADDE 1- (1) Amaç bu Kanunun uygulanmasını düzenlemektir.\n\n"
            "MADDE 2- (1) Kapsam tüm veri sorumlularını içerir.\n\n"
            "GEÇİCİ MADDE 9- (1) Geçiş hükümleri iki yıl geçerlidir.\n")
    s.add_chunks(chunk_articles(parse_plain_text(text, "MOCK-TR", language="tr")))
    assert s.article_range("MOCK-TR") == (1, 2)


# --- Bug 7: compound-clause splitting ----------------------------------------

def test_compound_clauses_match_inflected_turkish_markers():
    q = ("TLPT testleri ne sıklıkla yapılmalı ve aktif test aşaması "
         "ne kadar sürer?")
    clauses = _compound_clauses(q)
    assert len(clauses) == 2, clauses
    # each clause keeps its own subject
    assert "TLPT" in clauses[0]
    assert "aktif test aşaması" in clauses[1]


def test_compound_clauses_english_keep_subjects():
    q = ("Under DORA, how often must TLPT be carried out and how long "
         "does the active phase last?")
    clauses = _compound_clauses(q)
    assert len(clauses) == 2
    assert "TLPT" in clauses[0]
    assert clauses[1].startswith("how long")


def test_repeated_marker_alone_does_not_split():
    q = "How often must we test? I mean, how often is testing required?"
    assert _compound_clauses(q) == []


# --- Bug 10/14: routing cues -------------------------------------------------

def test_gdpr_cue_matches_curly_apostrophe():
    assert _GDPR_CONTEXT_RE.search("AB’de müşterilerimiz var")
    assert _GDPR_CONTEXT_RE.search("AB'de müşterilerimiz var")


def test_diacriticless_turkish_still_routes_to_kvkk():
    q = "veri ihlali bildirimi ne zaman yapilmasi gerekir"
    assert _detect_required_regulations(q) == {"KVKK"}


# --- Bug 11: article_range must exclude sub-document numbering ---------------

def test_article_range_ignores_decision_item_numbers():
    from regintel.ingestion.parser import parse_decision_text
    tmp = tempfile.mkdtemp()
    s = RegulationStore(persist_dir=tmp, use_embeddings=False)
    statute = "Madde 1\nAmaç\nBu Kanunun amacı verileri korumaktır.\n"
    s.add_chunks(chunk_articles(parse_plain_text(statute, "MOCK-TR", language="tr")))
    decision = ("1- Birinci tedbir alınmalıdır.\n2- İkinci tedbir "
                "alınmalıdır.\n99- Doksan dokuzuncu madde işareti.\n")
    s.add_chunks(chunk_articles(parse_decision_text(
        decision, "MOCK-TR", document_label="Kurul Kararı 2099/5")))
    assert s.article_range("MOCK-TR") == (1, 1), \
        "decision item numbers polluted the statute article range"


# --- Compare-mode audit (2026-08-22, second pass) ----------------------------

def _compare_store():
    """Two mock instruments with deliberately overlapping topics."""
    from regintel.ingestion.parser import parse_plain_text
    A = ("Article 12\nData security\nThe controller shall take technical and "
         "administrative measures.\n\nArticle 15\nBreach notification\nThe "
         "controller shall notify the Board without delay upon becoming aware "
         "of a personal data breach.\n\nArticle 18\nFines\nPenalties are "
         "imposed by the Board for violations.\n")
    B = ("Article 32\nSecurity of processing\nThe controller shall implement "
         "appropriate technical and organisational measures.\n\nArticle 33\n"
         "Breach notification\nThe controller shall notify the supervisory "
         "authority not later than 72 hours after becoming aware.\n\n"
         "Article 83\nFines\nFines shall be effective and dissuasive.\n")
    s = RegulationStore(persist_dir=tempfile.mkdtemp(), use_embeddings=False)
    s.add_chunks(chunk_articles(parse_plain_text(A, "KVKK")))
    s.add_chunks(chunk_articles(parse_plain_text(B, "GDPR")))
    return s


def test_compare_source_numbering_matches_combined_results():
    """The [n] numbering printed in the prompt must line up with the
    1-based index into results_a + results_b, or every citation in the
    second regulation's column resolves to the wrong source."""
    import re
    from regintel.generation.llm import EchoLLM
    from regintel.generation.rag import RagPipeline
    pipe = RagPipeline(store=_compare_store(), llm=EchoLLM())
    results, prompt = pipe._compare_prompt("breach notification", "KVKK", "GDPR", 4)
    numbered = re.findall(r"^\[(\d+)\] (\w+), Article (\S+)", prompt, re.MULTILINE)
    assert numbered
    for n, reg, art in numbered:
        r = results[int(n) - 1]
        assert r.metadata["regulation"] == reg, (n, reg, r.metadata)
        assert str(r.metadata["article_number"]) == art


def test_marker_claim_is_scoped_to_table_cell():
    """A citation in one column must not be graded against the other
    column's text (compare answers are tables, and a table row has no
    sentence punctuation between the two regulations' cells)."""
    from regintel.generation.rag import _marker_claim
    row = ("| Deadline | Board notified without delay [1] | Supervisory "
           "authority notified not later than 72 hours [2] |\n")
    m = row.index("[1]")
    claim = _marker_claim(row, m, m + 3)
    assert "Board notified" in claim
    assert "supervisory" not in claim.lower(), \
        "claim leaked into the other regulation's column"


def test_compare_citation_survives_long_opposing_cell():
    kvkk = _sr("KVKK, Article 15: The controller shall notify the Board "
               "without delay upon becoming aware of a personal data breach.",
               reg="KVKK", art="15")
    gdpr = _sr("GDPR, Article 33: the controller shall notify the supervisory "
               "authority not later than 72 hours after having become aware of "
               "it, unless the breach is unlikely to result in a risk to the "
               "rights and freedoms of natural persons.", reg="GDPR", art="33")
    answer = ("| Aspect | KVKK | GDPR |\n|---|---|---|\n"
              "| Deadline | Board notified without delay [1] | The controller "
              "shall notify the supervisory authority not later than 72 hours "
              "after having become aware of it, unless the breach is unlikely "
              "to result in a risk to the rights and freedoms of natural "
              "persons [2] |\n")
    _, indices, _ = cited_sources(answer, [kvkk, gdpr])
    assert indices == [1, 2], "a well-grounded column lost its citation"


def test_stripped_marker_never_leaves_a_blank_table_cell():
    """COMPARE_TEMPLATE forbids blank columns (a blank cell reads as 'no
    requirement'), so marker stripping must not manufacture one."""
    kvkk = _sr("KVKK, Article 15: notification to the Board without delay.",
               reg="KVKK", art="15")
    gdpr = _sr("GDPR, Article 33: notify the supervisory authority not later "
               "than 72 hours after becoming aware.", reg="GDPR", art="33")
    # the KVKK cell contains nothing but an ungrounded marker
    answer = ("| Aspect | KVKK | GDPR |\n|---|---|---|\n"
              "| Encryption standards for backup tapes | [1] | Not addressed |\n")
    clean, _, _ = cited_sources(answer, [kvkk, gdpr])
    row = [l for l in clean.splitlines() if "Encryption" in l][0]
    cells = [c.strip() for c in row.split("|")[1:-1]]
    assert all(cells), f"blank cell produced: {cells}"


def test_prose_article_flag_matches_question_language():
    """The flag substitution bypasses the LLM, so it must not inject an
    English sentence into a Turkish answer (SYSTEM_PROMPT rule 6)."""
    retrieved = [_sr("KVKK, Article 15: ...", reg="KVKK", art="15")]
    tr = _flag_unverified_prose_articles(
        "KVKK Madde 12 uyarınca tedbir alınır.", retrieved)
    assert "doğrulanamadı" in tr
    assert "not confirmed" not in tr
    en = _flag_unverified_prose_articles(
        "Under KVKK Article 12 measures are required.", retrieved)
    assert "not confirmed" in en


def test_compare_with_same_regulation_on_both_sides_is_refused():
    """Both sides identical means every chunk is handed to the model
    twice under two different source numbers, and the model is asked to
    contrast an instrument with itself."""
    from regintel.generation.llm import EchoLLM
    from regintel.generation.rag import RagPipeline
    pipe = RagPipeline(store=_compare_store(), llm=EchoLLM())
    resp = pipe.compare("breach notification deadline", "KVKK", "KVKK")
    assert resp.sources == []
    assert "two different regulations" in resp.answer
    tr = pipe.compare("veri ihlali bildirimi süresi nedir", "KVKK", "KVKK")
    assert "iki farklı mevzuat" in tr.answer
    # streaming path guards identically
    results, stream = pipe.compare_stream("breach notification", "KVKK", "KVKK")
    assert results == []
    assert "two different regulations" in "".join(stream)


def test_compare_with_different_regulations_still_works():
    from regintel.generation.llm import EchoLLM
    from regintel.generation.rag import RagPipeline
    pipe = RagPipeline(store=_compare_store(), llm=EchoLLM())
    results, prompt = pipe._compare_prompt("breach notification", "KVKK", "GDPR", 4)
    assert {r.metadata["regulation"] for r in results} == {"KVKK", "GDPR"}
    assert "Sources from KVKK" in prompt and "Sources from GDPR" in prompt


# --- v4 eval-set findings: jurisdiction cue coverage --------------------------

def test_english_question_about_turkish_law_routes_to_kvkk():
    """An English question about Turkish law contains no Turkish
    characters and never spells the country 'Türkiye', so it used to hit
    no KVKK cue at all — retrieval ran unfiltered and the higher-volume
    English GDPR corpus monopolized the results (v4 eval, case K1:
    8 GDPR articles, 0 KVKK)."""
    q = ("What adequate measures must a data controller take when processing "
         "special categories of personal data under Turkish data protection law?")
    assert _detect_required_regulations(q) == {"KVKK"}
    assert "KVKK" in _detect_required_regulations("Does Turkey require a DPO?")


def test_dora_scoped_entity_types_are_recognised():
    """REGULATION_TRIAGE_PROMPT lists the entity types DORA covers
    (payment/e-money institutions, investment firms, insurers), but only
    'credit institution' and 'eu bank' were encoded as cues — so a
    question naming any of the others routed nowhere (v4 eval, case S2:
    an EU-licensed payment institution's ICT vendor outage retrieved
    zero DORA sources)."""
    for phrase in ("EU-licensed payment institution",
                   "e-money institution", "investment firm",
                   "insurance undertaking"):
        assert "DORA" in _detect_required_regulations(
            f"We are an {phrase} and our ICT vendor had an outage."), phrase


def test_payments_startup_in_istanbul_still_routes_kvkk_only():
    """Guard for the cue additions above: the pinned Istanbul scenario
    must not start matching DORA just because 'payment' now appears in
    its cue list ('payments startup' is not 'payment institution')."""
    assert _detect_required_regulations(
        "I'm a backend developer at a payments startup in Istanbul, "
        "what do I need to worry about?") == {"KVKK"}


# --- Query-rewrite hardening (2026-08-22, third pass) ------------------------
#
# The rewrite is what makes the EU instruments reachable from a Turkish
# question at all: an unrewritten Turkish query returns ZERO BM25 chunks
# from GDPR. Since the bank's users ask in Turkish, this call is on the
# critical path for every DORA/GDPR answer they will ever get.

from regintel.generation.llm import LLM
from regintel.generation.rag import (RagPipeline, _looks_turkish,
                                     _retrieval_degraded, _degraded_note)
from regintel.generation import prompts


class _ScriptedRewriteLLM(LLM):
    """Returns a scripted sequence for QUERY_REWRITE_PROMPT calls; an entry
    that is an Exception instance is raised instead of returned."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), 0

    def chat(self, system: str, user: str) -> str:
        if system != prompts.QUERY_REWRITE_PROMPT:
            return user
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(r, Exception):
            raise r
        return r


TR_Q = "Veri ihlali bildirimi ne kadar sürede yapılmalıdır?"


def _pipe(llm):
    return RagPipeline(store=RegulationStore(persist_dir=tempfile.mkdtemp(),
                                             use_embeddings=False), llm=llm)


def test_looks_turkish_catches_what_the_narrow_regex_misses():
    """_TURKISH_CHARS_RE excludes ç/ö/ü on purpose (loanwords), which let
    'veri ihlali bildirim süresi' pass as English. Query-language checks
    need the broad set plus the ASCII hints."""
    assert _looks_turkish("veri ihlali bildirim süresi")
    assert _looks_turkish("veri ihlali bildirimi suresi")   # no diacritics at all
    assert not _looks_turkish("personal data breach notification deadline")


def test_rewrite_retries_a_transient_failure():
    llm = _ScriptedRewriteLLM(RuntimeError("502"), "breach notification deadline")
    q = _pipe(llm)._retrieval_query(TR_Q)
    assert llm.calls == 2
    assert q == "breach notification deadline"
    assert not _retrieval_degraded(q, "GDPR")


def test_rewrite_rejects_an_untranslated_result_and_retries():
    llm = _ScriptedRewriteLLM("veri ihlali bildirim süresi",
                              "data breach notification deadline")
    q = _pipe(llm)._retrieval_query(TR_Q)
    assert llm.calls == 2
    assert q == "data breach notification deadline"


def test_rewrite_keeps_best_candidate_rather_than_raw_question():
    """If no attempt is convincingly translated, the shortened candidate is
    still a better query than the whole raw question."""
    llm = _ScriptedRewriteLLM("veri ihlali bildirim süresi")
    q = _pipe(llm)._retrieval_query(TR_Q)
    assert q == "veri ihlali bildirim süresi"
    assert q != TR_Q


def test_degradation_is_flagged_only_for_the_english_corpora():
    dead = _ScriptedRewriteLLM(RuntimeError("down"))
    pipe = _pipe(dead)
    q = pipe._retrieval_query(TR_Q)
    assert q == TR_Q                       # total failure -> raw question
    assert _retrieval_degraded(q, "GDPR")  # English corpus: unreachable
    assert _retrieval_degraded(q, "DORA")
    assert not _retrieval_degraded(q, "KVKK")   # Turkish corpus: fine
    assert _retrieval_degraded(q, None)         # unfiltered spans English


def test_degraded_search_is_reported_to_the_model():
    """A failed translation must never reach the user as 'this regulation
    does not cover it' — the prompt has to say the SEARCH was incomplete."""
    pipe = _pipe(_ScriptedRewriteLLM(RuntimeError("down")))
    _, prompt = pipe._ask_prompt(TR_Q, "GDPR", 4)
    assert "Retrieval note" in prompt
    assert "do NOT state that the regulation does not address it" in prompt
    _, tr_prompt = pipe._ask_prompt(TR_Q, "KVKK", 4)
    assert "Retrieval note" not in tr_prompt


def test_no_degraded_note_when_rewrite_works():
    pipe = _pipe(_ScriptedRewriteLLM("data breach notification deadline"))
    _, prompt = pipe._ask_prompt(TR_Q, "GDPR", 4)
    assert "Retrieval note" not in prompt
