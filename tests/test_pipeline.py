"""End-to-end pipeline tests using a synthetic mini-regulation.

Runs without embeddings, network, or an LLM (BM25-only store + EchoLLM),
so it works in CI and on any machine.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from regintel import config
from regintel.ingestion.parser import parse_plain_text, parse_eurlex_html, parse_decision_text
from regintel.ingestion.chunker import Chunk, chunk_articles
from regintel.retrieval.store import RegulationStore, SearchResult
from regintel.generation import prompts
from regintel.generation.rag import RagPipeline, cited_sources
from regintel.generation.llm import LLM, EchoLLM


class _FakeLLM(LLM):
    """Test double: returns a scripted response for the DECOMPOSE_PROMPT
    call and just echoes its input back for anything else (a no-op
    "rewrite") — lets tests exercise the real decomposition parsing/
    merge logic without needing Ollama."""

    def __init__(self, decomposition: str):
        self.decomposition = decomposition

    def chat(self, system: str, user: str) -> str:
        if system == prompts.DECOMPOSE_PROMPT:
            return self.decomposition
        return user

# Synthetic fixture — NOT real regulation text.
MOCK_REG_A = """
Article 1
Scope
This mock regulation applies to financial entities and covers digital
operational resilience testing obligations.

Article 2
Incident reporting
Financial entities shall report major ICT-related incidents to the
competent authority within 24 hours of classification. An intermediate
report shall follow within 72 hours.

Article 3
Third-party risk
Entities shall maintain a register of all contractual arrangements with
ICT third-party service providers and review it annually.
"""

MOCK_REG_B = """
Article 1
Scope
This second mock directive applies to essential and important entities
operating critical infrastructure.

Article 2
Notification obligations
Essential entities shall submit an early warning of significant
incidents within 24 hours and an incident notification within 72 hours.
"""


@pytest.fixture(scope="module")
def store():
    tmp = tempfile.mkdtemp()
    s = RegulationStore(persist_dir=tmp, use_embeddings=False)
    for text, reg in [(MOCK_REG_A, "MOCK-A"), (MOCK_REG_B, "MOCK-B")]:
        articles = parse_plain_text(text, reg)
        s.add_chunks(chunk_articles(articles))
    return s


def test_parser_extracts_articles():
    articles = parse_plain_text(MOCK_REG_A, "MOCK-A")
    assert len(articles) == 3
    assert articles[1].article_number == "2"
    assert articles[1].article_title == "Incident reporting"
    assert "24 hours" in articles[1].text


def test_turkish_madde_headings():
    text = "Madde 5\nVeri güvenliği\nVeri sorumlusu gerekli tedbirleri alır.\n"
    articles = parse_plain_text(text, "MOCK-TR", language="tr")
    assert len(articles) == 1
    assert articles[0].article_number == "5"


def test_eurlex_html_parser():
    html = """
    <html><body>
    <p class="oj-ti-section-1">CHAPTER I</p>
    <p class="oj-ti-art">Article 1</p>
    <p>Subject matter</p>
    <p>This Regulation lays down uniform requirements.</p>
    <p class="oj-ti-art">Article 2</p>
    <p>Scope</p>
    <p>This Regulation applies to financial entities.</p>
    </body></html>
    """
    articles = parse_eurlex_html(html, "MOCK-EU")
    assert len(articles) == 2
    assert articles[0].article_title == "Subject matter"
    assert articles[0].chapter == "CHAPTER I"


def test_chunks_carry_citation_metadata(store):
    results = store.search("incident reporting deadline", top_k=3)
    assert results
    top = results[0]
    assert top.metadata["regulation"] in ("MOCK-A", "MOCK-B")
    assert top.metadata["article_number"]
    assert "Article" in top.citation


def test_regulation_filter(store):
    results = store.search("incident reporting 24 hours",
                           top_k=5, regulation="MOCK-B")
    assert results
    assert all(r.metadata["regulation"] == "MOCK-B" for r in results)


def test_comparison_retrieves_both_sides(store):
    pipe = RagPipeline(store=store, llm=EchoLLM())
    resp = pipe.compare("incident reporting deadlines", "MOCK-A", "MOCK-B")
    regs = {s.metadata["regulation"] for s in resp.sources}
    assert regs == {"MOCK-A", "MOCK-B"}


def test_ask_returns_sources(store):
    pipe = RagPipeline(store=store, llm=EchoLLM())
    resp = pipe.ask("What must be in the third-party register?")
    assert resp.sources
    assert resp.answer  # EchoLLM returns context, never empty


# --- Board decision / guideline documents (not "Article N") ----------------
# NOT real regulation text — a synthetic stand-in for e.g. KVKK Kurul
# Kararı 2018/10, whose operative content is a lettered list rather than
# statute articles.
MOCK_DECISION = """
a) Özel nitelikli kişisel veriler mümkün olduğunca azaltılmalı ve
şifrelenmelidir.
b) Kriptografik anahtarlar, verinin bulunduğu ortamdan ayrı bir güvenli
ortamda saklanmalıdır.
c) Erişim yetkileri düzenli aralıklarla gözden geçirilmelidir.
"""


def test_parse_decision_text_extracts_items():
    articles = parse_decision_text(
        MOCK_DECISION, "MOCK-DEC", document_label="Kurul Kararı 2099/1",
        doc_date="2099-01-01")
    assert len(articles) == 3
    assert [a.article_number for a in articles] == ["a", "b", "c"]
    assert all(a.doc_type == "board_decision" for a in articles)
    assert all(a.document_label == "Kurul Kararı 2099/1" for a in articles)
    assert "ayrı bir güvenli" in articles[1].text


def test_parse_decision_text_falls_back_without_list_markers():
    articles = parse_decision_text(
        "Bu değişiklik 6. maddeyi etkilemiştir, ek koşul getirilmemiştir.",
        "MOCK-DEC", document_label="Değişiklik Notu")
    assert len(articles) == 1
    assert articles[0].article_number == ""


# Mirrors the real structure of Board Decision 2018/10: numbered top-level
# items, some with their own lettered sub-items that RESTART at "a" under
# every number — a bare letter alone is not a unique label in a document
# like this.
MOCK_NESTED_DECISION = """
1- Ayrı bir politika belirlenmelidir.

2- Çalışanlara yönelik:
a) Düzenli eğitim verilmelidir.
b) Gizlilik sözleşmesi yapılmalıdır.

3- Elektronik ortamda:
a) Kriptografik yöntemler kullanılmalıdır.
b) Anahtarlar ayrı ortamda tutulmalıdır.
c) İşlem kayıtları loglanmalıdır.
"""


def test_parse_decision_text_disambiguates_repeated_letters_under_numbered_groups():
    articles = parse_decision_text(
        MOCK_NESTED_DECISION, "MOCK-DEC", document_label="Kurul Kararı 2099/2")
    labels = [a.article_number for a in articles]
    assert labels == ["1", "2", "2a", "2b", "3", "3a", "3b", "3c"]
    assert len(labels) == len(set(labels)), "repeated letters collided across numbered groups"

    # The letterless top-level item's own text must survive, not be
    # dropped as "content before the first match".
    item1 = next(a for a in articles if a.article_number == "1")
    assert "Ayrı bir politika" in item1.text

    # Chunk ids must be unique too — a collision here means add_chunks()
    # would silently overwrite one section's chunks with another's.
    chunks = chunk_articles(articles)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_decision_chunk_citation_uses_document_label(store):
    articles = parse_decision_text(
        MOCK_DECISION, "MOCK-DEC", document_label="Kurul Kararı 2099/1")
    store.add_chunks(chunk_articles(articles))
    results = store.search("kriptografik anahtar güvenli ortam",
                           top_k=3, regulation="MOCK-DEC")
    assert results
    assert "Kurul Kararı 2099/1" in results[0].citation
    assert "Article" not in results[0].citation
    assert results[0].metadata["doc_type"] == "board_decision"


def test_clear_document_does_not_touch_sibling_statute():
    """clear_document() must only remove the named document's chunks —
    not the whole regulation, which would silently wipe out statute
    articles sharing the same regulation id (e.g. KVKK statute + KVKK
    Kurul Kararı 2018/10)."""
    tmp = tempfile.mkdtemp()
    s = RegulationStore(persist_dir=tmp, use_embeddings=False)

    statute_articles = parse_plain_text(MOCK_REG_A, "MOCK-A")
    s.add_chunks(chunk_articles(statute_articles))

    decision_articles = parse_decision_text(
        MOCK_DECISION, "MOCK-A", document_label="Kurul Kararı 2099/1")
    s.add_chunks(chunk_articles(decision_articles))

    s.clear_document("MOCK-A", "Kurul Kararı 2099/1")

    remaining = s._all_docs()
    labels = {m.get("document_label") for _, m in remaining.values()}
    assert "Kurul Kararı 2099/1" not in labels
    assert any(m["regulation"] == "MOCK-A" and m.get("article_number") == "1"
              for _, m in remaining.values()), "statute chunks were wiped too"


def test_scope_note_flags_missing_document_types():
    tmp = tempfile.mkdtemp()
    s = RegulationStore(persist_dir=tmp, use_embeddings=False)
    s.add_chunks(chunk_articles(parse_plain_text(MOCK_REG_A, "MOCK-A")))
    pipe = RagPipeline(store=s, llm=EchoLLM())

    statute_only_note = pipe._scope_note("MOCK-A")
    assert "no Board" in statute_only_note

    s.add_chunks(chunk_articles(parse_decision_text(
        MOCK_DECISION, "MOCK-A", document_label="Kurul Kararı 2099/1")))
    combined_note = pipe._scope_note("MOCK-A")
    assert "no Board" not in combined_note
    assert "board decision" in combined_note


def test_doc_types_reports_indexed_types():
    tmp = tempfile.mkdtemp()
    s = RegulationStore(persist_dir=tmp, use_embeddings=False)
    s.add_chunks(chunk_articles(parse_plain_text(MOCK_REG_A, "MOCK-A")))
    assert s.doc_types("MOCK-A") == {"statute"}

    s.add_chunks(chunk_articles(parse_decision_text(
        MOCK_DECISION, "MOCK-A", document_label="Kurul Kararı 2099/1")))
    assert s.doc_types("MOCK-A") == {"statute", "board_decision"}


# --- Retrieval hygiene: relevance floor, diversity cap, jurisdiction routing


def _mock_chunk(id_, text, reg, art, title=""):
    return Chunk(id=id_, text=text, metadata={
        "regulation": reg, "article_number": art, "article_title": title,
        "chapter": "", "chunk_index": 0, "total_chunks": 1, "source_url": "",
        "language": "en", "doc_type": "statute", "doc_date": "",
        "in_force": True, "document_label": ""})


def test_diversity_cap_limits_chunks_per_article():
    """One article dominating every scored chunk must not crowd every
    other relevant article out of the result list (the S3 failure: a
    single Art 9 chunk retrieved and recited five times)."""
    tmp = tempfile.mkdtemp()
    s = RegulationStore(persist_dir=tmp, use_embeddings=False)
    chunks = [_mock_chunk(f"MOCK-DIV-art9-{i}",
                          f"encryption keys management part {i} for special "
                          f"category data encryption keys transfer",
                          "MOCK-DIV", "9") for i in range(5)]
    chunks.append(_mock_chunk(
        "MOCK-DIV-art6-0",
        "special category data encryption keys must be held separately",
        "MOCK-DIV", "6"))
    # Unrelated distractor: with too few, too-similar documents, BM25's
    # IDF term can go negative for common words and short-circuit the
    # score<=0 cutoff before diversity capping even runs — a small-corpus
    # artifact, not something this test is about.
    chunks.append(_mock_chunk(
        "MOCK-DIV-artX-0",
        "third party ICT register contractual arrangements review annually",
        "MOCK-DIV", "28"))
    s.add_chunks(chunks)

    # top_k=3 matches exactly what the two relevant articles can diversely
    # supply (2 capped from article 9 + 1 from article 6) — a larger top_k
    # would correctly backfill extra article-9 chunks rather than
    # under-fill, which is intended (diversity is a preference, not a
    # reason to return fewer results than requested), so isn't what this
    # assertion is checking.
    results = s.search("encryption keys special category data",
                       top_k=3, regulation="MOCK-DIV")
    by_article: dict[str, int] = {}
    for r in results:
        by_article[r.metadata["article_number"]] = \
            by_article.get(r.metadata["article_number"], 0) + 1

    assert by_article.get("9", 0) <= config.MAX_CHUNKS_PER_ARTICLE
    assert "6" in by_article, "article 6 was crowded out entirely"


def test_bm25_relative_floor_drops_weak_matches():
    """A chunk that only incidentally shares one word with the query
    (score far below the query's own top match) should be dropped before
    it can be handed to the LLM as if it were relevant context."""
    tmp = tempfile.mkdtemp()
    s = RegulationStore(persist_dir=tmp, use_embeddings=False)
    s.add_chunks([
        _mock_chunk("A-1", "penetration testing frequency threat led "
                    "penetration testing frequency requirements",
                    "MOCK-DIV", "26"),
        _mock_chunk("A-2", "administrative penalties publication annual "
                    "report testing schedule unrelated matter",
                    "MOCK-DIV", "54"),
        # Distractor doc, same reason as test_diversity_cap_limits_chunks_per_article.
        _mock_chunk("A-3", "third party ICT register contractual "
                    "arrangements review annually",
                    "MOCK-DIV", "28"),
    ])
    results = s.search("penetration testing frequency", top_k=5, regulation="MOCK-DIV")
    ids = {r.id for r in results}
    assert "A-1" in ids
    assert "A-2" not in ids, "weak keyword-overlap chunk was not filtered out"


def test_detect_required_regulations_from_context_cues():
    from regintel.generation.rag import _detect_required_regulations

    kvkk_only = _detect_required_regulations(
        "I'm a backend developer at a payments startup in Istanbul, what do I need to worry about?")
    assert kvkk_only == {"KVKK"}

    both = _detect_required_regulations(
        "DevOps engineer at a mid-size EU bank here, our core banking API went down.")
    assert "DORA" in both

    explicit = _detect_required_regulations("Compare GDPR and KVKK encryption rules")
    assert explicit == {"GDPR", "KVKK"}


def test_multi_regulation_question_retrieves_all_named_regulations():
    """When a question names more than one regulation, both must be
    represented in the context handed to the LLM — a single unfiltered
    search can let one instrument's chunks monopolize the ranking and
    leave the other with zero representation (F4 in the eval report)."""
    tmp = tempfile.mkdtemp()
    s = RegulationStore(persist_dir=tmp, use_embeddings=False)
    # Real acronyms as the mock regulation ids so _REG_NAME_RE (built from
    # config.REGULATIONS) recognizes them without needing to monkeypatch —
    # this is an isolated temp store, not the real corpus.
    s.add_chunks(chunk_articles(parse_plain_text(MOCK_REG_A, "KVKK")))
    s.add_chunks(chunk_articles(parse_plain_text(MOCK_REG_B, "DORA")))

    pipe = RagPipeline(store=s, llm=EchoLLM())
    results, _ = pipe._ask_prompt(
        "Compare incident reporting under KVKK and DORA",
        regulation=None, top_k=6)
    regs = {r.metadata["regulation"] for r in results}
    assert regs == {"KVKK", "DORA"}


# --- Scenario decomposition -------------------------------------------------


def test_needs_decomposition_triggers_on_length_and_multi_regulation():
    pipe = RagPipeline(store=RegulationStore(
        persist_dir=tempfile.mkdtemp(), use_embeddings=False), llm=EchoLLM())

    assert not pipe._needs_decomposition("KVKK'ya göre madde 6 nedir?")
    assert pipe._needs_decomposition("x " * 300)  # well past the length threshold
    assert pipe._needs_decomposition(
        "Our Istanbul office and our EU bank subsidiary both process this data.")


def test_decompose_question_returns_single_item_for_echollm():
    pipe = RagPipeline(store=RegulationStore(
        persist_dir=tempfile.mkdtemp(), use_embeddings=False), llm=EchoLLM())
    assert pipe._decompose_question("anything") == ["anything"]


def test_decompose_question_strips_numbering_and_junk_lines():
    decomposition = "\n".join([
        "1) Does the notes field create a special-category data issue?",
        "",
        "2. Is the nightly mirror to Germany a cross-border transfer?",
        "ok",  # too short, should be dropped
        "- Is disk-at-rest encryption required here?",
    ])
    pipe = RagPipeline(store=RegulationStore(
        persist_dir=tempfile.mkdtemp(), use_embeddings=False),
        llm=_FakeLLM(decomposition))
    sub_qs = pipe._decompose_question("irrelevant, scripted response used")
    assert len(sub_qs) == 3
    assert all(not q[0].isdigit() and not q.startswith(("-", "."))
              for q in sub_qs)
    assert "special-category" in sub_qs[0]


def test_scenario_decomposition_merges_per_issue_retrieval():
    """S1/S3-style long scenario: retrieval must cover every issue the
    decomposition step extracts, not just whichever one an unfiltered
    single pass happens to favor."""
    tmp = tempfile.mkdtemp()
    s = RegulationStore(persist_dir=tmp, use_embeddings=False)
    s.add_chunks(chunk_articles(parse_plain_text(MOCK_REG_A, "KVKK")))
    s.add_chunks(chunk_articles(parse_plain_text(MOCK_REG_B, "DORA")))

    scenario = "I run a small startup with customers in Turkey. " * 15
    decomposition = "\n".join([
        "What is the incident reporting deadline under KVKK?",
        "What are the notification obligations under DORA?",
    ])
    pipe = RagPipeline(store=s, llm=_FakeLLM(decomposition))
    results, prompt = pipe._ask_prompt(scenario, regulation=None, top_k=6)

    assert "Issues identified" in prompt
    regs = {r.metadata["regulation"] for r in results}
    assert regs == {"KVKK", "DORA"}


def test_short_question_never_triggers_decomposition_path():
    """Latency guard: a normal short single-issue question must take the
    plain ANSWER_TEMPLATE path, never the scenario path, regardless of
    the LLM used."""
    tmp = tempfile.mkdtemp()
    s = RegulationStore(persist_dir=tmp, use_embeddings=False)
    s.add_chunks(chunk_articles(parse_plain_text(MOCK_REG_A, "MOCK-A")))
    pipe = RagPipeline(store=s, llm=EchoLLM())
    _, prompt = pipe._ask_prompt(
        "What must be in the third-party register?", regulation=None, top_k=6)
    assert "Issues identified" not in prompt


# --- Citation binding check --------------------------------------------------


def _sr(text: str, reg: str = "MOCK", art: str = "1") -> SearchResult:
    return SearchResult(id=f"{reg}-{art}", text=text, score=1.0, metadata={
        "regulation": reg, "article_number": art, "article_title": "",
        "chapter": "", "chunk_index": 0, "total_chunks": 1, "source_url": "",
        "language": "en", "doc_type": "statute", "doc_date": "",
        "in_force": True, "document_label": ""})


def test_cited_sources_strips_citation_unsupported_by_its_source():
    """A marker citing a real, in-range source whose claim that source
    doesn't actually support (the model reaching for the nearest numbered
    source rather than the right one) must be stripped, not just a
    marker citing a nonexistent source number."""
    results = [
        _sr("Data controllers must encrypt personal data using "
            "industry-standard cryptographic algorithms and manage "
            "encryption keys in a separate secure environment."),
        _sr("The supervisory authority publishes an annual transparency "
            "report listing administrative penalties imposed that year."),
    ]
    answer = ("Encryption keys must be stored separately from the "
              "encrypted data [1]. You are also required to water the "
              "office plants every Tuesday morning [2].")
    clean, indices, cited = cited_sources(answer, results)
    assert indices == [1]
    assert cited == [results[0]]
    assert "[2]" not in clean
    assert "[1]" in clean


def test_cited_sources_keeps_grounded_citation():
    results = [_sr("Financial entities shall report major ICT-related "
                   "incidents to the competent authority within 24 hours "
                   "of classification.")]
    answer = "You must report major ICT-related incidents within 24 hours [1]."
    clean, indices, cited = cited_sources(answer, results)
    assert indices == [1]
    assert cited == [results[0]]
    assert "[1]" in clean


def test_cited_sources_refusal_carries_no_footer():
    results = [_sr("Some unrelated source text about a completely "
                   "different regulatory topic entirely.")]
    answer = "The sources do not address this question."
    clean, indices, cited = cited_sources(answer, results)
    assert indices == []
    assert cited == []
