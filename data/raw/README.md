# Missing corpus: KVKK Board decisions & guidance

The 20-case evaluation (`Reponses/RegIntel_Evaluation_Report_2026-08-20.md`)
found that roughly half the eval set's ground truth lives in documents this
corpus doesn't index yet. The statute (`KVKK.pdf`) alone doesn't say what
"adequate security measures" *means* for special-category data — that's in
Board (Kurul) decisions and official guidance. Obtain each document below
and save it as plain text at the path shown, then run:

```
python scripts/ingest.py KVKK-KK-2018-10
python scripts/ingest.py KVKK-REHBER-VERI-GUVENLIGI
python scripts/ingest.py KVKK-KK-2025-1572
python scripts/ingest.py KVKK-7499-AMENDMENT-NOTES
```

(or `python scripts/ingest.py` with no arguments to pick up everything
present in this folder, statutes and documents alike).

## Documents to obtain

All four are published by the Kişisel Verilerin Korunması Kurumu at
**kvkk.gov.tr** (Kararlar / Mevzuat sections) or **mevzuat.gov.tr** for the
7499 amendment. Save the plain text of each as the filename listed.

- **`KVKK-KK-2018-10.txt`** — Kurul Kararı 2018/10 (31.01.2018, RG
  07.03.2018 No. 30353): the operative "yeterli önlemler" for processing
  special-category (özel nitelikli) personal data — separate policy,
  access authorisations with periodic review, cryptographic encryption
  with keys held in a separate environment, encrypted transfer (KEP /
  corporate e-mail, VPN/sFTP), etc. This is the single most-cited-missing
  source in the eval report (Q4–Q6, Q8, Q11, S1, S3, S4).

- **`KVKK-REHBER-VERI-GUVENLIGI.txt`** — Kişisel Veri Güvenliği Rehberi
  (KVKK's technical/administrative measures guide). Needed for the
  "düzenli olarak" (regular, no fixed count) testing language that Q1/Q2
  depend on, and the firewall/backup measures list S4 probes.

- **`KVKK-KK-2025-1572.txt`** — Kurul Kararı 2025/1572 (04.09.2025):
  revised VERBİS registration exemption criteria. Needed for S8; without
  it, the system should say its VERBİS criteria may be stale rather than
  reciting old thresholds as current (see the out-of-corpus prompt rule
  in `regintel/generation/prompts.py`).

- **`KVKK-7499-AMENDMENT-NOTES.txt`** — a short summary of what Law No.
  7499 (March 2024) changed in KVKK Art 6 and Art 9. Needed for Q9. This
  one doesn't need to be the full amending law text — a clear summary of
  which articles changed and how is enough for the pipeline to answer
  "yes, 7499 changed X" instead of denying the amendment exists.

## Format

Plain text, one file per document. `parse_decision_text()`
(`regintel/ingestion/parser.py`) splits on lettered/numbered list markers
("a)", "b)", "1)"...) — the natural structure of a Kurul kararı's
operative measures list. A document with fewer than two such markers
(e.g. the short 7499 amendment notes) is indexed as a single section,
which is fine.

## Why this isn't done for you

This is regulatory/legal source material published by a Turkish
government authority — not something to reconstruct from a paraphrased
description. Fabricating "placeholder" legal text in a compliance tool's
corpus would be actively dangerous: it could get cited as if authoritative.
The ingestion code, metadata schema (`doc_type` / `doc_date` / `in_force`),
and decision-aware parser are ready; only the actual documents are missing.
