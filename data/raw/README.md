# KVKK corpus: statute + Board decisions & guidance

Status: **all four documents are now in place and indexed** (as of
2026-08-21). This note documents what's here and where it came from, for
anyone re-ingesting or extending the corpus later.

## What's indexed

```
python scripts/ingest.py            # re-ingest everything in this folder
```

- **`KVKK.txt`** (statute) — the current, consolidated KVKK text (33
  articles), including the Law No. 7499 (March 2024) amendments annotated
  inline on the articles they changed (e.g. Article 9 carries
  "(Değişik:2/3/2024-7499/34 md.)"). This replaced the earlier
  `KVKK.pdf` source, which didn't carry those amendments.
- **`KVKK-KK-2018-10.txt`** (`doc_type: board_decision`) — Kurul Kararı
  2018/10 (31.01.2018, RG 07.03.2018 No. 30353): the operative "yeterli
  önlemler" for processing special-category (özel nitelikli) personal
  data. Nests lettered sub-items under numbered top-level items ("1-",
  "2-"... each with its own "a) b) c)..."), which `parse_decision_text()`
  labels hierarchically ("2a", "3a", ...) so repeated letters across
  different numbered groups don't collide.
- **`KVKK-REHBER-VERI-GUVENLIGI.txt`** (`doc_type: guideline`) — Kişisel
  Veri Güvenliği Rehberi, KVKK's technical/administrative measures guide.
- **`KVKK-KK-2025-1572.txt`** (`doc_type: board_decision`) — Kurul Kararı
  2025/1572 (04.09.2025): revised VERBİS registration exemption criteria.

All are published by the Kişisel Verilerin Korunması Kurumu at
**kvkk.gov.tr** (Kararlar / Mevzuat sections).

## Format

Plain text, one file per document. Statute text (`doc_type: statute`,
`config.REGULATIONS`) goes through `parse_plain_text()` — "Madde N"
headings. Board decisions and guidance (`config.DOCUMENTS`) go through
`parse_decision_text()` — numbered/lettered list markers ("1-", "a)",
...). A document with fewer than two such markers is indexed as a single
section.

## Why this wasn't done automatically

This is regulatory/legal source material published by a Turkish
government authority — not something to reconstruct from a paraphrased
description. Fabricating "placeholder" legal text in a compliance tool's
corpus would be actively dangerous: it could get cited as if
authoritative. The ingestion code, metadata schema (`doc_type` /
`doc_date` / `in_force`), and decision-aware parser were built ahead of
time; the actual documents were sourced and dropped in afterward.
