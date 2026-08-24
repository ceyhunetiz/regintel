# Corpus: statutes, EU Level-2 instruments, Board decisions & guidance

Status: 25 source documents indexed (as of 2026-08-21) across DORA, GDPR and
KVKK. This note documents what's here, where it came from, and how it's
parsed — for anyone re-ingesting or extending the corpus later.

```
python scripts/ingest.py            # re-ingest everything in this folder
python scripts/ingest.py DORA       # re-ingest one statute
python scripts/ingest.py KVKK-KK-2019-10   # re-ingest one document
```

## Statutes / base regulations (`config.REGULATIONS`, `doc_type: statute`)

Parsed with `parse_plain_text()`/`parse_eurlex_html()` — "Article N" /
"Madde N" headings split into individually citable articles.

| File | What it is | Notes |
|---|---|---|
| `KVKK.txt` | 6698 sayılı Kanun, consolidated (33 articles) | Carries the Law No. 7499 (12.3.2024, RG 32487) amendments annotated inline on the articles they changed, e.g. Article 9 carries "(Değişik:2/3/2024-7499/34 md.)". |
| `GDPR.txt` | Regulation (EU) 2016/679, OJ L 119, 4.5.2016 | All 99 articles, verified sequential 1–99. Publications Office OJ PDF — original as-published text, does **not** include the 2018 corrigendum. Replaces the earlier `GDPR.html` source (removed: `ingest()` prefers `.html` over `.txt` when both exist, which would have silently kept indexing the older source). |
| `DORA.html` | Regulation (EU) 2022/2554 (64 articles) | EUR-Lex Official Journal HTML. |

## KVKK: Board decisions, guidance & amendment notes (`config.DOCUMENTS`)

All published by the Kişisel Verilerin Korunması Kurumu at **kvkk.gov.tr**
(Kararlar / Mevzuat sections). Parsed with `parse_decision_text()` —
numbered/lettered list markers ("1-", "a)", ...), since these documents
have no "Madde N" headings.

| File | doc_type | Decision / doc | Date |
|---|---|---|---|
| `KVKK-KK-2018-10.txt` | `board_decision` | Özel nitelikli veriler için alınması gereken yeterli önlemler (RG 07.03.2018 No. 30353). Nests lettered sub-items under numbered top-level items ("1-", "2-"... each with its own "a) b) c)..."), which the parser labels hierarchically ("2a", "3a", ...) so repeated letters across different numbered groups don't collide. | 31.01.2018 |
| `KVKK-KK-2019-10.txt` | `board_decision` | Veri ihlali bildirim usul ve esasları — the source of the 72-hour Board-notification figure. The statute (Art 12(5)) only says "en kısa sürede"; without this decision the system has nothing to ground a breach-clock answer in. | 24.01.2019 |
| `KVKK-KK-2019-271.txt` | `board_decision` | Veri sorumlusu tarafından ilgili kişiye yapılan veri ihlali bildiriminde yer alması gereken asgari unsurlar — complements 2019/10 with the *content* (not timing) of a breach notification to affected individuals. Contains two separate flat a)/b)/c)-style lists with no numbered item above either; the parser detects the second list restarting at "a" and prefixes it ("g1a", "g1b"...) so it doesn't collide with and silently overwrite the first list's chunks. | 18.09.2019 |
| `KVKK-KK-2025-1572.txt` | `board_decision` | Revised VERBİS registration exemption criteria. | 04.09.2025 |
| `KVKK-REHBER-VERI-GUVENLIGI.txt` | `guideline` | Kişisel Veri Güvenliği Rehberi — technical/administrative measures guide. | Ocak 2018 |
| `KVKK-7499-AMENDMENT-NOTES.txt` | `notes` | **Not official statutory text.** A derived note explaining what Law 7499 changed in KVKK Madde 6, 9, 18 and Geçici Madde 3, with paragraph pinpoints and Türk alfabesi lettering (a, b, c, ç, d...); verified line-by-line against `KVKK.txt`. The document_label itself says "gayriresmi" (unofficial) and SYSTEM_PROMPT rule 13 tells the model never to cite it as operative wording — only KVKK.txt is authoritative for the amended articles' actual text. |

## DORA: Level-2 RTS/ITS, delegated-act adoption texts, supervisory guidance

Sourced from EUR-Lex (`eur-lex.europa.eu`) and the relevant ESA (EBA/ESMA/
EIOPA). These are legally distinct instruments from base DORA, some of
which *also* use "Article N" headings — their own numbering, unrelated to
base DORA's Article 1, 2, 3.... `parser: "articles"` entries go through
`parse_plain_text()` with `doc_type`/`document_label` set so chunker.py
never treats them as base DORA's own statute text (which would otherwise
collide chunk ids: both would produce a "DORA Article 5" chunk). Entries
without "Article N" headings (guidelines, opinions, decisions between
authorities) use `parser: "decision"` instead.

| File | doc_type / parser | Instrument | Subject |
|---|---|---|---|
| `DORA-RTS-2024-1772.txt` | `rts` / articles | (EU) 2024/1772 | Incident **classification criteria and materiality thresholds**. Does NOT set reporting time limits — its own text says so. |
| `DORA-RTS-2025-301-incident-reporting-timelines.txt` | `rts` / articles | (EU) 2025/301 | **The actual DORA reporting deadlines** (Art 5): initial notification 4h from classification / 24h from awareness; intermediate report 72h from initial notification; final report 1 month after intermediate. This is what DORA Art 19 delegates to — not 2024/1772. |
| `DORA-ITS-2025-302-incident-reporting-forms.txt` | `its` / articles | (EU) 2025/302 | Standard forms/templates for incident reporting. |
| `DORA-RTS-2024-1773-ict-third-party-policy.txt` | `rts` / articles | (EU) 2024/1773 | Policy on contractual arrangements for ICT services supporting critical/important functions. |
| `DORA-RTS-2024-1774-ict-risk-management.txt` | `rts` / articles | (EU) 2024/1774 | ICT risk management tools/methods/policies + simplified framework. |
| `DORA-RTS-2025-1190-TLPT.txt` | `rts` / articles | (EU) 2025/1190 | Threat-led penetration testing. |
| `DORA-RTS-2025-295-oversight-conduct.txt` | `rts` / articles | (EU) 2025/295 | Conditions for conducting oversight activities. |
| `DORA-RTS-2025-420-joint-examination-teams.txt` | `rts` / articles | (EU) 2025/420 | Joint examination team composition. |
| `DORA-RTS-2025-532-subcontracting.txt` | `rts` / articles | (EU) 2025/532 | Subcontracting ICT services supporting critical/important functions. |
| `DORA-ITS-2024-2956-register-of-information.txt` | `its` / articles | (EU) 2024/2956 | Register-of-information templates (Art 28(9) DORA). |
| `DORA-DR-C2024-896-CTPP-designation-criteria.txt` | `delegated_act_draft` / articles | C(2024) 896 final | Criteria for designating critical ICT third-party providers (Art 31 DORA). **Commission adoption text, not the final OJ-published version** — includes explanatory memorandum, lacks final OJ numbering; document_label says so, verify article numbers against the published Regulation before citing. |
| `DORA-DR-C2024-902-oversight-fees.txt` | `delegated_act_draft` / articles | C(2024) 902 final | Oversight fees charged by the Lead Overseer. Same pre-OJ-numbering caveat. |
| `DORA-JC-GL-2024-36-oversight-cooperation.txt` | `guideline` / decision | JC/GL/2024/36 | Oversight cooperation and information exchange between the ESAs and competent authorities (Art 32(7) DORA). Structured as recurring a)/b)/c) lists with no numbered item above them; the parser's flat-group-restart detection ("g1a", "g2a"...) keeps each restart distinct. |
| `DORA-JC-2024-34-guidelines-costs-and-losses.txt` | `guideline` / decision | JC 2024 34 | Estimating aggregated annual costs and losses from major ICT incidents. Document states it was **not final** at issue — `in_force: False` reflects that. |
| `DORA-ESA-2024-22-CTPP-reporting-decision.txt` | `decision` / decision | ESA 2024/22 | EBA/ESMA/EIOPA joint decision on reporting for critical ICT third-party provider designation (Art 31(1)(a) DORA). Mixes numbered section headings with independently-numbered points that don't reset per section — the parser's numeric-collision suffix ("2-2", "3-2"...) keeps every point distinct rather than letting a heading silently overwrite a real point sharing its number. |
| `DORA-EIOPA-BOS-24-425-opinion-solvency-ii-scope.txt` | `opinion` / decision | EIOPA-BoS-24/425 | Impact of Solvency II size thresholds on which insurers fall in DORA's scope. Supervisory opinion, not binding legislation. Same numbered-heading/point structure as the ESA decision above. |

## Excluded from the index

- **`_excluded/EU-DIRECTIVE-2022-2556.txt`** — the DORA "omnibus" Directive.
  Almost entirely amendment instructions ("in Article 74(1), the first
  subparagraph is replaced by the following..."), which would retrieve as
  DORA-adjacent text carrying no standalone obligation — a direct feeder
  for noise-padding. Kept out of `data/raw/` proper (in a subfolder) so
  `scripts/ingest.py`'s auto-discovery (which treats any stray `.txt`/
  `.html` file not in `config.DOCUMENTS` as a statute to ingest) doesn't
  pick it up by accident.

## Format

Plain text, one file per document. OJ page furniture (running headers,
`L 119/1` page markers, bare page numbers, `ELI:` footer lines) stripped;
leading whitespace stripped so heading lines sit at the start of the line
for the parser regexes.

## Why this wasn't done automatically

This is regulatory/legal source material — not something to reconstruct
from a paraphrased description. Fabricating "placeholder" legal text in a
compliance tool's corpus would be actively dangerous: it could get cited
as if authoritative. The ingestion code, metadata schema (`doc_type` /
`doc_date` / `in_force` / `document_label`), and parsers (statute /
decision / Level-2-articles) were built ahead of time; the actual
documents were sourced and dropped in afterward, and every parser was
dry-run against the real text (checking for label collisions / empty
sections) before anything was ingested for real.
