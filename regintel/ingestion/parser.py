"""Parse regulation documents into structured articles.

The unit of structure is the *article*: every chunk we index carries
regulation + article metadata so answers can cite "DORA, Article 17"
rather than an anonymous text block.

Two parsing strategies:
  1. EUR-Lex HTML (preferred): uses the 'ti-art' / 'oj-ti-art' CSS
     classes that mark article headings in Official Journal HTML.
  2. Plain text fallback: regex on 'Article N' headings. Works for
     any regulation you only have as extracted text (e.g. from a PDF).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from bs4 import BeautifulSoup


@dataclass
class Article:
    regulation: str          # e.g. "DORA"
    article_number: str      # e.g. "17" (statute) or item label, e.g. "a" (decision)
    article_title: str       # e.g. "ICT-related incident management process"
    chapter: str             # e.g. "Chapter III"
    text: str
    source_url: str = ""
    language: str = "en"
    # doc_type distinguishes statute text from every other kind of source
    # (Board/Kurul decisions, EU Level-2 RTS/ITS, delegated-act adoption
    # texts, supervisory guidelines/opinions/decisions, and non-official
    # derived notes) — see parse_decision_text() and parse_plain_text()'s
    # document_label/doc_type kwargs. doc_date/in_force let the pipeline
    # state temporal facts ("2018/10 remains in force") instead of
    # guessing. Only "statute" is special-cased in code (chunker id/label
    # format, citation rendering); the rest is free-form and just has to
    # be a stable, human-readable noun for the given source.
    doc_type: str = "statute"          # "statute" | "board_decision" | "guideline" |
                                        # "rts" | "its" | "delegated_act_draft" |
                                        # "opinion" | "decision" | "notes"
    doc_date: str = ""                 # ISO date, e.g. "2018-01-31"
    in_force: bool = True
    document_label: str = ""           # e.g. "Kurul Kararı 2018/10" — used for
                                        # citations instead of "Article N" when
                                        # doc_type != "statute"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Strategy 1: EUR-Lex Official Journal HTML
# --------------------------------------------------------------------------

_ART_CLASSES = ("ti-art", "oj-ti-art")
# A chapter/section heading can be split across TWO differently-classed
# paragraphs — "ti-section-1" carries the marker itself ("CHAPTER II",
# "Section I"), "ti-section-2" carries its descriptive title ("ICT risk
# management") as a separate element. Both must be recognised here: a
# class missing from this set falls through to the plain-body branch
# below and silently bleeds onto whatever article is still open — DORA
# Article 4's own indexed text ended with "ICT risk management" (its
# OWN body has nothing to do with ICT risk management; that's chapter
# II's title, meant for Article 5 onward) before "ti-section-2" was
# added here.
_CHAPTER_CLASSES = ("ti-section-1", "oj-ti-section-1", "ti-section-2", "oj-ti-section-2")


def parse_eurlex_html(html: str, regulation: str, source_url: str = "",
                      language: str = "en") -> list[Article]:
    soup = BeautifulSoup(html, "lxml")
    articles: list[Article] = []
    current_chapter = ""

    # Walk all paragraphs in document order, tracking chapter headings
    # and cutting a new Article at each article heading.
    #
    # Deliberately excludes "div": EUR-Lex wraps every article (and every
    # chapter/section) in a structural <div class="eli-subdivision">.
    # find_all() matches those wrapper divs too, at every nesting depth,
    # and visits each one *before* the <p class="oj-ti-art"> heading
    # nested inside it. A bare wrapper div carries none of the heading
    # classes below, so it used to fall into the plain-text branch and
    # its .get_text() would flatten the *entire* subtree — including the
    # next article's heading and full body — onto whatever article was
    # still open. The nested heading was then found moments later and
    # correctly started a new Article, so article counts came out right
    # (masking the bug) while large spans of text were duplicated onto
    # the wrong, preceding article.
    elements = soup.find_all(["p", "table"])
    current: Article | None = None
    buffer: list[str] = []
    # A chapter heading can span multiple consecutive elements (a
    # CHAPTER marker, its descriptive title, then a Section marker and
    # ITS descriptive title, all before the next article) — accumulated
    # here and joined into one current_chapter string once the next
    # article heading is reached, rather than each new element
    # silently overwriting whatever the previous one just set (which
    # would lose "CHAPTER II" the moment "Section I" followed it).
    chapter_parts: list[str] = []

    def flush() -> None:
        nonlocal current
        if current is not None:
            current.text = _clean("\n".join(buffer))
            if current.text:
                articles.append(current)
        buffer.clear()
        current = None

    for el in elements:
        cls = set(el.get("class") or [])

        if cls & set(_CHAPTER_CLASSES):
            chapter_text = _clean(el.get_text(" ", strip=True))
            if chapter_text:
                chapter_parts.append(chapter_text)

        elif cls & set(_ART_CLASSES):
            if chapter_parts:
                current_chapter = " — ".join(chapter_parts)
                chapter_parts = []
            flush()
            heading = _clean(el.get_text(" ", strip=True))
            m = re.match(r"Article\s+(\d+\w*)", heading, re.IGNORECASE)
            number = m.group(1) if m else heading
            current = Article(
                regulation=regulation,
                article_number=number,
                article_title="",  # filled by the next subtitle element
                chapter=current_chapter,
                text="",
                source_url=source_url,
                language=language,
            )

        elif current is not None:
            text = _clean(el.get_text(" ", strip=True))
            if not text:
                continue
            # First short line after the heading is the article title
            if not current.article_title and not buffer and len(text) < 200:
                current.article_title = text
            elif el.name != "table":  # table text comes via child <p> tags
                buffer.append(text)

    flush()
    return articles


# --------------------------------------------------------------------------
# Strategy 2: plain text fallback (PDF extracts, .txt files)
# --------------------------------------------------------------------------

# A running page-footer artifact from PDF-to-text extraction of the EU
# Level-2 instruments (RTS/ITS .txt sources) — a page number then "ELI:
# <url>", alone on its own line at every page break, unrelated to
# article/chapter structure (observed up to 18 times in one document).
# Stripped before any splitting, not just from each extracted body
# afterward: left in place it can (a) land in the middle of whatever
# article/section happens to be open at that page break, and (b) be
# mistaken for a genuine title/chapter candidate by the heading-
# recovery helpers below — it's short, has no trailing period, and
# doesn't start with "(", so it passes every other check they use.
_PAGE_FOOTER_RE = re.compile(r"^\s*\d+/\d+\s+ELI:\s*\S+\s*$", re.MULTILINE)


def _strip_page_footers(text: str) -> str:
    return _PAGE_FOOTER_RE.sub("", text)


# Standalone heading on its own line: "Article 5" / "Madde 5", plus
# Turkish transitional articles ("GEÇİCİ MADDE 3"). Without the optional
# GEÇİCİ prefix, transitional articles never matched and their full text
# was silently appended to the LAST regular article's body — KVKK's
# Geçici Madde 3 (the 7499 amendment's transition provision, a topic
# this corpus specifically curates) was buried inside the wrong article
# under the wrong citation.
_TEXT_ART_RE = re.compile(
    r"^(?P<gecici>(?:GE[ÇC][İI]C[İI]|Ge[çc]ici)\s+)?"
    r"(?:Article|Madde|MADDE)\s+(?P<num>\d+\w*)\s*[-–—]?\s*$",
    re.MULTILINE)
# Turkish inline style used by mevzuat.gov.tr: "MADDE 5- (1) Kişisel..."
# The heading and the first paragraph share a line.
_TEXT_ART_INLINE_RE = re.compile(
    r"^(?P<gecici>(?:GE[ÇC][İI]C[İI]|Ge[çc]ici)\s+)?"
    r"(?:Madde|MADDE)\s+(?P<num>\d+\w*)\s*[-–—]",
    re.MULTILINE)

# A Turkish statute's top-level chapter heading — "İKİNCİ BÖLÜM",
# "DÖRDÜNCÜ BÖLÜM"... — always ALL CAPS, always its own line. Used by
# _recover_heading to confirm a genuine chapter boundary rather than an
# incidental short line of body prose (see that function's docstring).
_CHAPTER_TR_RE = re.compile(r"^[A-ZÇĞİÖŞÜ]+\s+B[ÖO]L[ÜU]M$")

# EUR-lex standalone-style ("Article N" alone on its own line, GDPR.txt/
# DORA.html, and the DORA Level-2 RTS/ITS .txt sources) top-level
# heading markers: "CHAPTER II", "TITLE III". Several source .txt files
# carry a letter-spacing artifact inherited from their original PDF
# extraction that mangles these too ("Sec ti on 1", "C HAP TER IV") —
# _is_chapter_marker/_is_section_marker collapse all whitespace before
# matching rather than assuming clean spacing.
_CHAPTER_KEYWORD_RE = re.compile(r"^(CHAPTER|TITLE)[IVXLCDM]+$", re.IGNORECASE)


def _is_chapter_marker(cand: str) -> bool:
    return bool(_CHAPTER_KEYWORD_RE.match(re.sub(r"\s+", "", cand)))


def _preceding_line(text: str, pos: int) -> tuple[str, int]:
    """The stripped text of the line immediately before offset `pos`,
    and the offset where that line starts. Returns ("", pos) if that
    line is empty, too long to plausibly be a heading, starts a
    numbered clause ("(1)...") — i.e. it reads as ordinary body text —
    or is itself an inline "MADDE N-"/"Article N" match: a short
    article whose entire inline body fits on one physical line (a real
    risk for a short statute article, and the norm in this project's
    own test fixtures) looks exactly like a plausible title candidate
    by length and leading-character alone, and mistaking that whole
    line for the NEXT article's title would wrongly swallow this
    article's entire real body into the trim in parse_plain_text.

    Also rejects a line ending in "." — a genuine Turkish statute
    title is a bare noun phrase ("Kişisel verilerin aktarılması") and
    never carries a full stop; a wrapped sentence's last physical line
    before a paragraph break does ("...Başbakanlık tarafından yerine
    getirilir."), and is exactly the other shape that this heading
    recovery must not mistake for a title (observed on KVKK's Geçici 2
    and 3, each introduced by an amendment annotation with no title
    line of its own: the last line of the PRECEDING article's wrapped
    body was short enough to pass every other check here).
    """
    prefix = text[:pos].rstrip(" \t\n")
    line_start = prefix.rfind("\n") + 1
    cand = prefix[line_start:].strip()
    if (0 < len(cand) < 120 and not cand.startswith("(")
            and not cand.endswith(".")
            and not _TEXT_ART_INLINE_RE.search(cand)
            and not _TEXT_ART_RE.match(cand)):
        return cand, line_start
    return "", pos


def _is_section_marker(cand: str) -> bool:
    return bool(re.match(r"^Section\d+$", re.sub(r"\s+", "", cand), re.IGNORECASE))


def _recover_chapter_block(text: str, pos: int, max_lines: int = 4) -> tuple[str, int]:
    """Recover a CHAPTER/Section heading block bled into the tail of
    the article preceding an 'Article N' match at `pos`.

    Standalone style keeps an article's OWN title inline, right after
    its own heading line — no backward recovery needed for that part,
    unlike KVKK's inline style. But a CHAPTER/Section marker between
    one article and the next has nowhere else to attach and silently
    becomes trailing noise on the wrong (preceding) article instead:
    observed on GDPR, where 19 of 99 articles ended with exactly this
    (a chapter transition can stack up to 4 such lines — a CHAPTER
    marker, its descriptive title, a Section marker, and ITS
    descriptive title — all bled onto the article before it).

    Walks backward consuming up to `max_lines` short, non-body-looking
    lines (via _preceding_line's own safeguards), stopping early at a
    confirmed CHAPTER marker (always the outermost line when present).
    Only commits a trim if the run actually contains a confirmed
    CHAPTER or Section marker somewhere in it — an unconfirmed run of
    short lines is left alone rather than guessed at, same principle
    as _recover_heading's chapter check.
    """
    lines: list[str] = []
    starts: list[int] = []
    cursor = pos
    for _ in range(max_lines):
        cand, start = _preceding_line(text, cursor)
        if not cand:
            break
        lines.append(cand)
        starts.append(start)
        cursor = start
        if _is_chapter_marker(cand):
            break
    for idx in range(len(lines) - 1, -1, -1):
        if _is_chapter_marker(lines[idx]) or _is_section_marker(lines[idx]):
            return " — ".join(reversed(lines[:idx + 1])), starts[idx]
    return "", pos


def _recover_heading(text: str, pos: int) -> tuple[str, str, int]:
    """Recover a bled-in article title (and Turkish chapter heading, if
    any) from the line(s) immediately before an 'Article N'/'Madde N'
    match at `pos`.

    mevzuat.gov.tr's inline style puts an article's title on the line
    *before* "MADDE N-" rather than after it, and — at a chapter
    boundary — a two-line chapter heading the same way, immediately
    above that:
        İKİNCİ BÖLÜM
        Kişisel Verilerin İşlenmesi
        Genel ilkeler
        MADDE 4- (1) ...
    Returns (title, chapter, block_start): block_start is the text
    offset where the recovered block begins. The caller uses it to cut
    the tail of the PRECEDING article's body there instead of at this
    match's own start — every KVKK article otherwise silently absorbed
    the next article's title (and, at a chapter boundary, the chapter
    heading too) as if it were part of its own text, since that text
    sits between the previous "MADDE N-" match and this one and so
    fell inside the naive end=matches[i+1].start() slice. The chapter
    is only accepted when a line two above the title actually matches
    _CHAPTER_TR_RE — a short line in that position is otherwise treated
    as ordinary trailing body prose from the previous article and left
    alone, not guessed at as a chapter title.
    """
    title, title_start = _preceding_line(text, pos)
    if not title:
        return "", "", pos
    chapter_title, chapter_title_start = _preceding_line(text, title_start)
    if chapter_title:
        marker, marker_start = _preceding_line(text, chapter_title_start)
        if marker and _CHAPTER_TR_RE.match(marker):
            return title, f"{marker} — {chapter_title}", marker_start
    return title, "", title_start


def parse_plain_text(text: str, regulation: str, source_url: str = "",
                     language: str = "en", doc_type: str = "statute",
                     document_label: str = "", doc_date: str = "",
                     in_force: bool = True) -> list[Article]:
    """Split plain text on 'Article N' / 'Madde N' headings.

    'Madde' support means Turkish regulations (KVKK, BDDK) parse with
    the same code path. Both standalone headings and mevzuat.gov.tr's
    inline "MADDE 5- (1) ..." style are recognized.

    doc_type/document_label let this same "Article N" splitter serve EU
    Level-2 instruments (RTS/ITS/delegated acts) that are legally
    distinct from — but share article numbering 1, 2, 3... with — the
    base regulation they supplement (e.g. DORA's incident-reporting RTS
    has its own "Article 5", unrelated to base DORA's own Article 5).
    Passing doc_type != "statute" here is what keeps chunker.py from
    building a chunk id/citation as if this were the base regulation's
    own Article 5 — see chunk_article()'s is_statute branch.
    """
    text = _strip_page_footers(text)
    matches = list(_TEXT_ART_RE.finditer(text))
    if len(matches) < 3:  # standalone style not found; try inline style
        inline = list(_TEXT_ART_INLINE_RE.finditer(text))
        if len(inline) > len(matches):
            matches = inline

    # Pass 1: title/chapter for each match, recovering from the
    # preceding line(s) (mevzuat.gov.tr inline style) when the body
    # itself doesn't start with one. own_starts[i] records where that
    # recovered block begins — matches[i].start() when nothing was
    # recovered (standalone style, or no title at all) — so pass 2 can
    # trim the PRECEDING article's body there instead of at this
    # match's own start (see _recover_heading's docstring).
    titles: list[str] = []
    chapters: list[str] = []
    own_starts: list[int] = []
    for m in matches:
        first_line = text[m.end():].lstrip("\n \t").split("\n", 1)[0].strip()
        # A short first line is the article title — unless it starts with
        # a numbered paragraph like "(1)", which means the body began on
        # the heading line (Turkish inline style).
        if len(first_line) < 150 and not first_line.startswith("("):
            # Standalone style ("Article N" alone, title inline right
            # after it) never needs backward title-recovery, but a
            # CHAPTER/Section marker between the previous article and
            # this heading still bleeds onto the wrong article unless
            # recovered the same way (see _recover_chapter_block).
            chapter, block_start = _recover_chapter_block(text, m.start())
            titles.append(first_line)
            chapters.append(chapter)
            own_starts.append(block_start)
        else:
            title, chapter, block_start = _recover_heading(text, m.start())
            titles.append(title)
            chapters.append(chapter)
            own_starts.append(block_start)

    articles: list[Article] = []
    common = dict(doc_type=doc_type, document_label=document_label,
                 doc_date=doc_date, in_force=in_force)

    for i, m in enumerate(matches):
        start = m.end()
        end = own_starts[i + 1] if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue

        lines = body.split("\n", 1)
        first = lines[0].strip()
        has_inline_title = len(first) < 150 and not first.startswith("(")
        rest = lines[1].strip() if has_inline_title and len(lines) > 1 else body

        # A transitional article gets its own distinct number ("Geçici 3")
        # so it can never collide with the regular article of the same
        # digit ("Madde 3"), and stays out of the statute's numeric
        # article range (article_range() filters on isdigit()).
        number = m.group("num")
        if m.group("gecici"):
            number = f"Geçici {number}"
        articles.append(Article(
            regulation=regulation,
            article_number=number,
            article_title=titles[i],
            chapter=chapters[i],
            text=_clean(rest),
            source_url=source_url,
            language=language,
            **common,
        ))
    return articles


# --------------------------------------------------------------------------
# Strategy 3: decision / guideline documents (no "Madde N" headings)
# --------------------------------------------------------------------------

# Kurul kararları structure their operative content as numbered top-level
# items ("1- ...", "2- ...") each often containing its own lettered
# sub-items ("a) ...", "b) ..."). The letters restart at "a" under every
# numbered item (see Decision 2018/10: items 2, 3, 4 and 5 each run their
# own a/b/c/ç/... sequence) — so a bare letter is NOT a unique label
# within the document; it only becomes one once combined with the
# numbered item it falls under. Turkish lettered lists use Turkish
# alphabet order (a, b, c, ç, d, e, ...), hence the ç/ğ/ı/ö/ş/ü in the
# marker class.
_DECISION_NUM_RE = re.compile(r"^\s*(\d{1,2})[)\.\-–—]\s+", re.MULTILINE)
_DECISION_LETTER_RE = re.compile(
    r"^\s*([a-zçğıöşü])[)\.]\s+", re.MULTILINE | re.IGNORECASE)


def parse_decision_text(text: str, regulation: str, document_label: str,
                        doc_type: str = "board_decision", doc_date: str = "",
                        in_force: bool = True, source_url: str = "",
                        language: str = "tr") -> list[Article]:
    """Split a Board decision or guideline into indexable sections.

    These documents don't have "Madde N" headings — parse_plain_text's
    regexes never match — so they need their own strategy: split on the
    numbered/lettered list markers that carry the actual operative
    measures, labelling a lettered sub-item with the numbered item it
    falls under (e.g. "3a", "3b") so repeated letters across different
    numbered groups don't collide into the same label — and therefore
    the same chunk id, which would otherwise silently overwrite each
    other via add_chunks()'s upsert. If fewer than 2 markers are found
    (e.g. a short amendment-notes document with no list structure), the
    whole text is kept as a single section rather than dropped.

    Two further collision classes, both found by dry-running this parser
    against real documents before ingesting them:

    - A lettered list with no numbered item above it at all (current_num
      stays "" for the whole document) is NOT automatically safe just
      because there's no number to collide on — a document can contain
      more than one such flat list (e.g. one quoting the statute's a)/b)/
      c) items, then a separate operative a)/b)/c)/ç)/d) list further
      down). A repeated letter is detected as a *restart* of a new flat
      group and gets a synthetic "gN" prefix (second flat group onward;
      the first stays unprefixed so single-flat-list documents keep
      their original plain "a"/"b"/"c" labels).
    - A numbered top-level marker can also legitimately repeat when a
      document mixes section headings with independently numbered
      points that don't reset per section (observed in a supervisory
      opinion: "1. LEGAL BASIS" as a heading, then "1. The European..."
      as its first point) — this is always a genuine collision, never a
      real domain restart, so any repeat gets a disambiguating "-2",
      "-3"... suffix rather than overwriting the earlier occurrence.

    Returned Articles reuse article_number for the item label (e.g.
    "3a") so downstream chunking/retrieval code doesn't need a separate
    field — doc_type != "statute" is what tells citation formatting to
    render document_label ("Kurul Kararı 2018/10") instead of "Article N".
    """
    text = _strip_page_footers(text)
    tagged = sorted(
        [(m.start(), m.end(), m.group(1))
         for m in _DECISION_NUM_RE.finditer(text)] +
        [(m.start(), m.end(), m.group(1).lower())
         for m in _DECISION_LETTER_RE.finditer(text)],
        key=lambda t: t[0])
    common = dict(regulation=regulation, chapter="", source_url=source_url,
                 language=language, doc_type=doc_type, doc_date=doc_date,
                 in_force=in_force, document_label=document_label)

    if len(tagged) < 2:
        body = _clean(text)
        if not body:
            return []
        return [Article(article_number="", article_title="", text=body, **common)]

    articles: list[Article] = []
    current_num = ""
    numeric_seen: dict[str, int] = {}
    flat_group = 0
    seen_in_group: set[str] = set()
    for i, (start, end, raw_label) in enumerate(tagged):
        seg_end = tagged[i + 1][0] if i + 1 < len(tagged) else len(text)
        body = _clean(text[end:seg_end])
        is_numeric = raw_label.isdigit()
        if is_numeric:
            current_num = raw_label
            seen_in_group = set()
            numeric_seen[raw_label] = numeric_seen.get(raw_label, 0) + 1
            count = numeric_seen[raw_label]
            label = raw_label if count == 1 else f"{raw_label}-{count}"
        else:
            if raw_label in seen_in_group:
                flat_group += 1
                seen_in_group = set()
            seen_in_group.add(raw_label)
            prefix = current_num or (f"g{flat_group}" if flat_group else "")
            label = f"{prefix}{raw_label}"
        if not body:
            continue
        articles.append(Article(article_number=label, article_title="",
                                text=body, **common))
    return articles


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
