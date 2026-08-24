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
# BM25 raw scores aren't comparable across queries (no fixed ceiling like
# cosine similarity), so the floor is relative to the query's own top
# score rather than an absolute number: a chunk scoring below this
# fraction of the best keyword match in the same result set is weak
# enough to be noise, even though score > 0. Calibrated conservatively —
# this only trims the long tail, not borderline-relevant matches.
MIN_BM25_RELATIVE_SCORE = 0.15
# Cap on how many chunks a single article/section can contribute to one
# fused result list. Without this, a long article that happens to score
# well on both semantic and keyword search can occupy every context slot
# (observed: one KVKK Art 9 chunk retrieved and recited five times),
# crowding out other relevant articles entirely. Raised from 2 to 3
# (2026-08-23): GDPR Article 33 is split into 5 parts, and a genuinely
# single-topic question (case G1) needs both its trigger/deadline
# paragraph AND its separate required-content paragraph — at a cap of 2,
# whichever pair of the 5 ranked highest for a given query phrasing
# sometimes excluded the content paragraph entirely, and the answer
# incorrectly reported "sources don't specify the content" when they
# did, just not within the 2 slots this cap allowed through. 3 of 5
# parts is still a real cap, not "let one article dominate" — the
# crowding-out problem above was about an article filling most/all of
# an 8-slot result set, which this remains far short of.
MAX_CHUNKS_PER_ARTICLE = 3
# Cap on how many chunks a single non-statute DOCUMENT (a Board decision,
# RTS/ITS, guideline...) can contribute, independent of the per-article
# cap above. A Board decision or RTS is internally split into many small
# sections, each with its own distinct article_number label ("3", "3a",
# "g1b"...) — so MAX_CHUNKS_PER_ARTICLE alone never limits it, since no
# two of its chunks ever share a label. Once the corpus grew to include
# multi-section RTS documents (40+ sections each) alongside the base
# regulation, this let a single document fill most or all of a top_k
# result list and crowd out the actual statute article a query needed
# (observed post-corpus-expansion: KVKK Kurul Kararı 2018/10 alone
# filling 3 of 8 slots and the 7499 amendment notes 3 more, pushing KVKK
# Art 6 itself out of an 8-result set entirely). Statute chunks are
# unaffected (document_label is empty for them, so this cap never
# applies) — only this dimension is new; MAX_CHUNKS_PER_ARTICLE keeps
# doing its own job within a single document too.
MAX_CHUNKS_PER_DOCUMENT = 2

# A first-person scenario question ("I'm a backend developer at a payments
# startup...") often buries 3-4 distinct legal issues; a single retrieval
# pass over the whole message catches at most one. Above this length (or
# when jurisdiction routing detects 2+ regulations — see
# _detect_required_regulations in rag.py — AND the question is at least
# SCENARIO_MIN_LENGTH_FOR_MULTI_REG long) the pipeline first asks the LLM
# to extract discrete sub-questions and retrieves per-issue instead.
# Short, single-issue questions (the common case) never cross either
# threshold, so their latency is unaffected.
#
# The multi-regulation branch needs its own floor: a short one-sentence
# question that merely *names* two regulations ("Both GDPR and DORA give
# you a 72-hour deadline, right?") isn't a multi-issue scenario — it's a
# single comparative question, and _detect_required_regulations alone
# used to be enough to trigger decomposition on it (v2 eval, case Q11:
# decomposition invented unrelated sub-questions and the answer went
# off-topic). A question that short and single-issue is already handled
# correctly by the non-decomposed multi-regulation search path in
# _ask_prompt, so it doesn't need decomposition at all.
SCENARIO_DECOMPOSITION_ENABLED = True
SCENARIO_LENGTH_THRESHOLD = 400  # characters
SCENARIO_MIN_LENGTH_FOR_MULTI_REG = 180  # characters

# --- LLM -------------------------------------------------------------------
# "ollama" = local model (default, fully offline)
# "api"    = OpenAI-compatible endpoint (set API_BASE_URL/API_MODEL below
#            and export REGINTEL_API_KEY). Only use enterprise endpoints
#            for real regulatory/organizational data.
LLM_PROVIDER = "api"
# OpenRouter. History, same day (2026-08-23):
#   1. qwen3-14b -> qwen3-235b-a22b-thinking-2507, after that thinking
#      model resolved this project's hardest failure (C3: qwen3-14b
#      fabricated a KVKK counterpart to DORA's vendor-oversight regime
#      in 3/3 attempts — prompt hardening, a mechanical relevance
#      warning, and wider retrieval all failed to stop it; the thinking
#      model explicitly said "no equivalent" per aspect instead).
#   2. -> qwen3-32b, reverted off the 235B model for two reasons: (a)
#      cost — ~10x qwen3-14b's output price, real money at volume; (b) a
#      genuine bug discovered from switching, not just cost: "-thinking"
#      -suffixed OpenRouter endpoints (this one and qwen3-30b-a3b-
#      thinking-2507 alike) reject `reasoning: {enabled: false}` outright
#      (400 "Reasoning is mandatory for this endpoint") — every auxiliary
#      call this codebase makes with reasoning deliberately OFF (query
#      rewrite, scenario decomposition, regulation classification — see
#      LLM.chat's reasoning docstring) was silently failing and falling
#      back to raw, untranslated text the whole time that model was
#      active, which is what actually broke case G3 (Turkish query never
#      translated, searched cross-lingual against English-only GDPR with
#      no fallback). qwen3-32b is a genuine hybrid model — reasoning is
#      optional, confirmed directly by calling it both ways — so the
#      auxiliary pipeline works as designed again. Verified qwen3-32b's
#      reasoning is real, not a no-op: given a premise-verification probe
#      that qwen3-14b failed on this project's own S5 case ("we report
#      annually" vs. a source that actually says "upon request"),
#      qwen3-32b's reasoning-on response correctly caught the
#      contradiction, using ~355 reasoning tokens — two orders of
#      magnitude less than the 235B model's 11,584 on a comparable
#      prompt. Pricing as of 2026-08-23: qwen3-32b $0.08/M in, $0.28/M
#      out — barely above qwen3-14b's $0.12/$0.24, far below the 235B
#      model's $0.23/$2.30. Get a key at openrouter.ai, then:
# export REGINTEL_API_KEY=sk-or-...
API_BASE_URL = "https://openrouter.ai/api/v1"
API_MODEL = "qwen/qwen3-32b"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:14b"  # good multilingual (EN + TR) model; upgraded
# from qwen3:8b after v3 eval (2026-08-21) showed the 8B model repeatedly
# and reproducibly inverting the plain meaning of unambiguous, correctly-
# cited source text, and confidently fabricating figures under pressure —
# both are exactly the failure classes that scale most reliably with
# model size, unlike this project's retrieval-ranking gaps, which a
# bigger LLM alone won't fix. See data/eval_runs/20260821-v3-final.md.
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
# A thinking model's own reasoning can run to several thousand tokens
# before its answer even starts — under LLM_MAX_TOKENS the answer would
# be truncated to nothing. Only applied to calls made with reasoning=True
# (the final answer generation — see LLM.chat's reasoning docstring);
# every other call (query rewrite, decomposition, classification) keeps
# the smaller budget above, since those tasks don't reason at all.
# Measured directly on a real compare-mode prompt (case C1): reasoning
# alone used 11,584 tokens — an earlier value of 8000 left content
# empty (finish_reason="length", content=None) before any answer text
# was generated at all. 24000 gives real headroom above the worst case
# observed rather than the bare minimum that happened to pass once;
# still well under $0.06/query at this model's $2.30/M output price
# even if fully used, which most calls won't be.
LLM_MAX_TOKENS_REASONING = 24000

# --- Known regulations -----------------------------------------------------
REGULATIONS = {
    "DORA": {
        "full_name": "Regulation (EU) 2022/2554 (Digital Operational Resilience Act)",
        "celex": "32022R2554",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2554",
        "language": "en",
    },
    "GDPR": {
        # Sourced as data/raw/GDPR.txt (all 99 articles, verified sequential
        # 1-99) from the Publications Office OJ PDF — the original
        # as-published text, not including the 2018 corrigendum. The
        # earlier GDPR.html source was removed: ingest() prefers .html
        # over .txt when both exist, which would have silently kept
        # indexing the older source and shadowed this one.
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

# --- Board decisions, guidance & other non-statute documents ----------------
# The base statute text alone is not the full picture: the *operative*
# measures for several obligations (e.g. what "adequate security measures"
# means for special-category data under KVKK, or DORA's actual incident-
# reporting timeline) live in Board (Kurul) decisions, EU Level-2 RTS/ITS,
# or supervisory guidance — not the primary statute/regulation itself.
# Ingestion looks for data/raw/<id>.txt and routes it through one of two
# parsers depending on "parser" below (see ingest_document() in
# scripts/ingest.py for exactly how each is used):
#   "decision" — no "Article N" headings at all (Turkish Kurul kararı
#                numbered/lettered lists, or free-form guidance/opinions).
#   "articles" — DOES have "Article N" headings, but under a different
#                legal instrument than the base regulation (an EU Level-2
#                RTS/ITS/delegated act has its own Article 1, 2, 3...,
#                unrelated to the base regulation's own numbering).
# See data/raw/README.md for where each one comes from and what it does
# and doesn't cover. No entry for the KVKK 7499 amendment's substantive
# text: the current KVKK statute (data/raw/KVKK.txt) already carries
# those amendments inline as "(Değişik:2/3/2024-7499/... md.)" annotations
# on the affected articles — KVKK-7499-AMENDMENT-NOTES below is a
# separate, explicitly non-official cross-reference note, not statute text.
DOCUMENTS = {
    # -- KVKK (Turkish DPA): Board decisions, guidance, amendment notes --
    "KVKK-KK-2018-10": {
        "regulation": "KVKK",
        "parser": "decision",
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
        "parser": "decision",
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
        "parser": "decision",
        "doc_type": "board_decision",
        "doc_date": "2025-09-04",
        "in_force": True,
        "document_label": "Kurul Kararı 2025/1572",
        "full_name": "VERBİS kayıt istisna kriterlerinin güncellenmesi (04.09.2025)",
        "url": "https://www.kvkk.gov.tr/",
        "language": "tr",
    },
    "KVKK-KK-2019-10": {
        "regulation": "KVKK",
        "parser": "decision",
        "doc_type": "board_decision",
        "doc_date": "2019-01-24",
        "in_force": True,
        "document_label": "Kurul Kararı 2019/10",
        "full_name": ("Kişisel veri ihlallerinde Kurula ve ilgili kişilere "
                      "bildirim yükümlülüğü (24.01.2019) — the source of the "
                      "72-hour Board notification figure; not in KVKK's own "
                      "statute text (which says only \"en kısa sürede\")"),
        "url": "https://www.kvkk.gov.tr/",
        "language": "tr",
    },
    "KVKK-KK-2019-271": {
        "regulation": "KVKK",
        "parser": "decision",
        "doc_type": "board_decision",
        "doc_date": "2019-09-18",
        "in_force": True,
        "document_label": "Kurul Kararı 2019/271",
        "full_name": ("Veri sorumlusu tarafından ilgili kişiye yapılan veri "
                      "ihlali bildiriminde yer alması gereken asgari unsurlar "
                      "(18.09.2019) — the required minimum content of a breach "
                      "notification to affected individuals, complementing "
                      "2019/10's timing/procedure rules"),
        "url": "https://www.kvkk.gov.tr/",
        "language": "tr",
    },
    "KVKK-7499-AMENDMENT-NOTES": {
        "regulation": "KVKK",
        "parser": "decision",
        "doc_type": "notes",
        "doc_date": "",
        "in_force": True,
        "document_label": ("7499 Değişiklik Notu (gayriresmi — resmi metin için "
                           "KVKK madde metnine bakınız)"),
        "full_name": ("Derived note on what Law 7499 (12.03.2024, RG 32487) "
                      "changed in KVKK Madde 6, 9, 18 and Geçici Madde 3 — NOT "
                      "official statutory text; verified line-by-line against "
                      "data/raw/KVKK.txt. Kept only to answer 'what changed and "
                      "where', never as a citable source for operative wording."),
        "url": "",
        "language": "tr",
    },
    # -- DORA (EU): Level-2 RTS/ITS, delegated-act adoption texts ---------
    "DORA-RTS-2024-1772": {
        "regulation": "DORA",
        "parser": "articles",
        "doc_type": "rts",
        "doc_date": "2024-03-13",
        "in_force": True,
        "document_label": "Commission Delegated Regulation (EU) 2024/1772",
        "full_name": ("RTS specifying ICT-incident classification criteria and "
                      "materiality thresholds under DORA Art 18(4) — NOT the "
                      "reporting-timeline RTS (that's 2025/301); this one's own "
                      "text says so explicitly."),
        "url": "https://eur-lex.europa.eu/eli/reg_del/2024/1772/oj",
        "language": "en",
    },
    "DORA-RTS-2025-301-incident-reporting-timelines": {
        "regulation": "DORA",
        "parser": "articles",
        "doc_type": "rts",
        "doc_date": "",
        "in_force": True,
        "document_label": "Commission Delegated Regulation (EU) 2025/301",
        "full_name": ("RTS Art 5: the actual DORA incident-reporting deadlines — "
                      "initial notification (4h from classification / 24h from "
                      "awareness), intermediate report (72h from initial "
                      "notification), final report (1 month after intermediate)."),
        "url": "https://eur-lex.europa.eu/eli/reg_del/2025/301/oj",
        "language": "en",
    },
    "DORA-ITS-2025-302-incident-reporting-forms": {
        "regulation": "DORA",
        "parser": "articles",
        "doc_type": "its",
        "doc_date": "",
        "in_force": True,
        "document_label": "Commission Implementing Regulation (EU) 2025/302",
        "full_name": "Standard forms/templates for DORA incident reporting.",
        "url": "https://eur-lex.europa.eu/eli/reg_impl/2025/302/oj",
        "language": "en",
    },
    "DORA-RTS-2024-1773-ict-third-party-policy": {
        "regulation": "DORA",
        "parser": "articles",
        "doc_type": "rts",
        "doc_date": "",
        "in_force": True,
        "document_label": "Commission Delegated Regulation (EU) 2024/1773",
        "full_name": ("RTS on the policy for contractual arrangements for ICT "
                      "services supporting critical/important functions."),
        "url": "https://eur-lex.europa.eu/eli/reg_del/2024/1773/oj",
        "language": "en",
    },
    "DORA-RTS-2024-1774-ict-risk-management": {
        "regulation": "DORA",
        "parser": "articles",
        "doc_type": "rts",
        "doc_date": "",
        "in_force": True,
        "document_label": "Commission Delegated Regulation (EU) 2024/1774",
        "full_name": ("RTS on ICT risk management tools, methods, processes and "
                      "policies, incl. the simplified framework for smaller entities."),
        "url": "https://eur-lex.europa.eu/eli/reg_del/2024/1774/oj",
        "language": "en",
    },
    "DORA-RTS-2025-1190-TLPT": {
        "regulation": "DORA",
        "parser": "articles",
        "doc_type": "rts",
        "doc_date": "",
        "in_force": True,
        "document_label": "Commission Delegated Regulation (EU) 2025/1190",
        "full_name": "RTS on threat-led penetration testing (TLPT).",
        "url": "https://eur-lex.europa.eu/eli/reg_del/2025/1190/oj",
        "language": "en",
    },
    "DORA-RTS-2025-295-oversight-conduct": {
        "regulation": "DORA",
        "parser": "articles",
        "doc_type": "rts",
        "doc_date": "",
        "in_force": True,
        "document_label": "Commission Delegated Regulation (EU) 2025/295",
        "full_name": "RTS on the conditions for conducting oversight activities.",
        "url": "https://eur-lex.europa.eu/eli/reg_del/2025/295/oj",
        "language": "en",
    },
    "DORA-RTS-2025-420-joint-examination-teams": {
        "regulation": "DORA",
        "parser": "articles",
        "doc_type": "rts",
        "doc_date": "",
        "in_force": True,
        "document_label": "Commission Delegated Regulation (EU) 2025/420",
        "full_name": "RTS on the composition of joint examination teams.",
        "url": "https://eur-lex.europa.eu/eli/reg_del/2025/420/oj",
        "language": "en",
    },
    "DORA-RTS-2025-532-subcontracting": {
        "regulation": "DORA",
        "parser": "articles",
        "doc_type": "rts",
        "doc_date": "",
        "in_force": True,
        "document_label": "Commission Delegated Regulation (EU) 2025/532",
        "full_name": ("RTS on subcontracting ICT services supporting critical "
                      "or important functions."),
        "url": "https://eur-lex.europa.eu/eli/reg_del/2025/532/oj",
        "language": "en",
    },
    "DORA-ITS-2024-2956-register-of-information": {
        "regulation": "DORA",
        "parser": "articles",
        "doc_type": "its",
        "doc_date": "",
        "in_force": True,
        "document_label": "Commission Implementing Regulation (EU) 2024/2956",
        "full_name": "Standard templates for the register of information (Art 28(9) DORA).",
        "url": "https://eur-lex.europa.eu/eli/reg_impl/2024/2956/oj",
        "language": "en",
    },
    "DORA-DR-C2024-896-CTPP-designation-criteria": {
        "regulation": "DORA",
        "parser": "articles",
        "doc_type": "delegated_act_draft",
        "doc_date": "2024-02-22",
        "in_force": True,
        "document_label": ("Delegated Regulation C(2024) 896 final "
                           "(adoption text, pre-OJ numbering)"),
        "full_name": ("Criteria for designating ICT third-party providers as "
                      "critical (Art 31 DORA). Commission adoption text, not "
                      "yet the final OJ-numbered version — verify article "
                      "numbers against the published Regulation before citing."),
        "url": "",
        "language": "en",
    },
    "DORA-DR-C2024-902-oversight-fees": {
        "regulation": "DORA",
        "parser": "articles",
        "doc_type": "delegated_act_draft",
        "doc_date": "",
        "in_force": True,
        "document_label": ("Delegated Regulation C(2024) 902 final "
                           "(adoption text, pre-OJ numbering)"),
        "full_name": ("Oversight fees charged by the Lead Overseer. Commission "
                      "adoption text — same pre-OJ-numbering caveat as 2024/896."),
        "url": "",
        "language": "en",
    },
    "DORA-JC-GL-2024-36-oversight-cooperation": {
        "regulation": "DORA",
        "parser": "decision",
        "doc_type": "guideline",
        "doc_date": "2024-11-06",
        "in_force": True,
        "document_label": "Joint Guidelines JC/GL/2024/36",
        "full_name": ("ESA Joint Guidelines on oversight cooperation and "
                      "information exchange between the ESAs and competent "
                      "authorities under DORA Art 32(7)."),
        "url": "",
        "language": "en",
    },
    "DORA-JC-2024-34-guidelines-costs-and-losses": {
        "regulation": "DORA",
        "parser": "decision",
        "doc_type": "guideline",
        "doc_date": "",
        "in_force": False,
        "document_label": "Guidelines JC 2024 34 (not final as issued)",
        "full_name": ("Guidelines on estimating aggregated annual costs and "
                      "losses from major ICT incidents. Document states it was "
                      "not final at issue — in_force=False reflects that."),
        "url": "",
        "language": "en",
    },
    "DORA-ESA-2024-22-CTPP-reporting-decision": {
        "regulation": "DORA",
        "parser": "decision",
        "doc_type": "decision",
        "doc_date": "2024-11-08",
        "in_force": True,
        "document_label": "ESA Decision 2024/22",
        "full_name": ("EBA/ESMA/EIOPA joint decision on reporting by competent "
                      "authorities to the ESAs for critical ICT third-party "
                      "provider designation (Art 31(1)(a) DORA)."),
        "url": "",
        "language": "en",
    },
    "DORA-EIOPA-BOS-24-425-opinion-solvency-ii-scope": {
        "regulation": "DORA",
        "parser": "decision",
        "doc_type": "opinion",
        "doc_date": "2024-11-14",
        "in_force": True,
        "document_label": "EIOPA-BoS-24/425 Opinion",
        "full_name": ("EIOPA opinion on the impact of increased Solvency II size "
                      "thresholds on which insurance undertakings fall in scope "
                      "of DORA. Supervisory opinion, not binding legislation."),
        "url": "",
        "language": "en",
    },
}
