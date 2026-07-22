"""End-to-end pipeline tests using a synthetic mini-regulation.

Runs without embeddings, network, or an LLM (BM25-only store + EchoLLM),
so it works in CI and on any machine.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from regintel.ingestion.parser import parse_plain_text, parse_eurlex_html
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
