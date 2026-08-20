"""End-to-end pipeline tests using a synthetic mini-regulation.

Runs without embeddings, network, or an LLM (BM25-only store + EchoLLM),
so it works in CI and on any machine.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from regintel.ingestion.parser import parse_plain_text, parse_eurlex_html, parse_decision_text
from regintel.ingestion.chunker import chunk_articles
from regintel.retrieval.store import RegulationStore
from regintel.generation.rag import RagPipeline
from regintel.generation.llm import EchoLLM

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
