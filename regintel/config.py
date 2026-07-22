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
LLM_TEMPERATURE = 0.1  # low temperature: we want grounded, not creative
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
