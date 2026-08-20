"""Central configuration for the Regulatory Intelligence Assistant."""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"

# --- Embeddings ------------------------------------------------------------
# Multilingual model so Turkish regulations (KVKK, BDDK) work later
# without re-architecting. Small enough to run on CPU.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Chunking --------------------------------------------------------------
MAX_CHUNK_CHARS = 1500   # ~350-400 tokens
CHUNK_OVERLAP_CHARS = 150

# --- Retrieval -------------------------------------------------------------
COLLECTION_NAME = "regulations"
DEFAULT_TOP_K = 6
RRF_K = 60  # reciprocal rank fusion constant
# Cosine-similarity floor for semantic search (score = 1 - distance).
# Calibrated against the multilingual MiniLM embedding model: genuinely
# on-topic in-corpus queries score ~0.6-0.8; totally unrelated queries
# score ~0.0-0.1. This only screens out that clearly-irrelevant tail —
# same-domain-but-wrong-regulation confusions can still score 0.4-0.6,
# so this is a coarse floor, not the primary relevance mechanism (see
# citation grounding in rag.py, which checks what the answer actually cites).
MIN_SEMANTIC_SCORE = 0.15

# --- LLM -------------------------------------------------------------------
# "ollama" = local model (default, fully offline)
# "api"    = OpenAI-compatible endpoint (set API_BASE_URL/API_MODEL below
#            and export REGINTEL_API_KEY). Only use enterprise endpoints
#            for real regulatory/organizational data.
LLM_PROVIDER = "ollama"
API_BASE_URL = "https://api.openai.com/v1"
API_MODEL = "gpt-4o-mini"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:8b"  # good multilingual (EN + TR) model
# Greedy decoding + a fixed seed: an audit-facing tool needs a reviewer
# re-running a query to get the answer that's in the working paper, not
# a materially different one. This applies to both the query-rewrite
# call and the answer-generation call (both go through the same
# LLM.chat/stream_chat), since rewrite non-determinism also changes
# which chunks get retrieved. Doesn't guarantee bit-for-bit reproduction
# on every backend (floating-point non-associativity in some GPU/batched
# runtimes can still cause rare drift), but removes sampling as a source
# of variance, which was the dominant one observed.
LLM_TEMPERATURE = 0.0
LLM_SEED = 42
LLM_MAX_TOKENS = 2500

# --- Known regulations -----------------------------------------------------
REGULATIONS = {
    "DORA": {
        "full_name": "Regulation (EU) 2022/2554 (Digital Operational Resilience Act)",
        "celex": "32022R2554",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2554",
        "language": "en",
    },
    "GDPR": {
        "full_name": "Regulation (EU) 2016/679 (General Data Protection Regulation)",
        "celex": "32016R0679",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679",
        "language": "en",
    },
    "KVKK": {
        "full_name": "6698 sayılı Kişisel Verilerin Korunması Kanunu",
        "url": "https://www.mevzuat.gov.tr/mevzuatmetin/1.5.6698.pdf",
        "language": "tr",
    },
    "BDDK": {
        # Which BDDK regulation(s) to index is team-specific. The most
        # relevant for cybersecurity governance is usually:
        # "Bankaların Bilgi Sistemleri ve Elektronik Bankacılık
        # Hizmetleri Hakkında Yönetmelik" (RG 15.03.2020/31069).
        # Download the PDF from mevzuat.gov.tr and save it as
        # data/raw/BDDK.pdf, then run: python scripts/ingest.py BDDK
        "full_name": "BDDK Bilgi Sistemleri Yönetmeliği",
        "url": "",  # manual download — see note above
        "language": "tr",
    },
    # Add more the same way, e.g. NIS2: {"celex": "32022L2555", ...}
}

# --- Board decisions & guidance (non-statute documents) ---------------------
# The KVKK statute text alone is not the full picture: the *operative*
# measures for several obligations (e.g. what "adequate security measures"
# means for special-category data) live in Kurul (Board) decisions and
# official guidance, not the law itself. These are NOT ingested by default
# — see data/raw/README.md for exactly which documents to obtain and from
# where (kvkk.gov.tr). Each entry here is a placeholder: ingestion looks
# for data/raw/<id>.txt and, once you've dropped the real text in, parses
# it with parse_decision_text() instead of the statute path.
DOCUMENTS = {
    "KVKK-KK-2018-10": {
        "regulation": "KVKK",
        "doc_type": "board_decision",
        "doc_date": "2018-01-31",
        "in_force": True,
        "document_label": "Kurul Kararı 2018/10",
        "full_name": ("Özel nitelikli kişisel veriler için alınması gereken "
                      "yeterli önlemler (31.01.2018, RG 07.03.2018 No. 30353)"),
        "url": "https://www.kvkk.gov.tr/",
        "language": "tr",
    },
    "KVKK-REHBER-VERI-GUVENLIGI": {
        "regulation": "KVKK",
        "doc_type": "guideline",
        "doc_date": "",
        "in_force": True,
        "document_label": "Kişisel Veri Güvenliği Rehberi",
        "full_name": "Kişisel Veri Güvenliği Rehberi (teknik ve idari tedbirler)",
        "url": "https://www.kvkk.gov.tr/",
        "language": "tr",
    },
    "KVKK-KK-2025-1572": {
        "regulation": "KVKK",
        "doc_type": "board_decision",
        "doc_date": "2025-09-04",
        "in_force": True,
        "document_label": "Kurul Kararı 2025/1572",
        "full_name": "VERBİS kayıt istisna kriterlerinin güncellenmesi (04.09.2025)",
        "url": "https://www.kvkk.gov.tr/",
        "language": "tr",
    },
    "KVKK-7499-AMENDMENT-NOTES": {
        "regulation": "KVKK",
        "doc_type": "guideline",
        "doc_date": "2024-03-02",
        "in_force": True,
        "document_label": "7499 Sayılı Kanun Değişiklik Notları",
        "full_name": ("7499 sayılı Kanun ile KVKK Madde 6 ve Madde 9 üzerinde "
                      "yapılan değişikliklerin özeti"),
        "url": "https://www.mevzuat.gov.tr/",
        "language": "tr",
    },
}
