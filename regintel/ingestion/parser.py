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
    # doc_type distinguishes statute text from Board (Kurul) decisions and
    # guidance documents, whose operative content isn't "Article N" — see
    # parse_decision_text(). doc_date/in_force let the pipeline state
    # temporal facts ("2018/10 remains in force") instead of guessing.
    doc_type: str = "statute"          # "statute" | "board_decision" | "guideline"
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
_CHAPTER_CLASSES = ("ti-section-1", "oj-ti-section-1")


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
                current_chapter = chapter_text

        elif cls & set(_ART_CLASSES):
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

# Standalone heading on its own line: "Article 5" / "Madde 5"
_TEXT_ART_RE = re.compile(r"^(?:Article|Madde|MADDE)\s+(\d+\w*)\s*[-–—]?\s*$",
                          re.MULTILINE)
# Turkish inline style used by mevzuat.gov.tr: "MADDE 5- (1) Kişisel..."
# The heading and the first paragraph share a line.
_TEXT_ART_INLINE_RE = re.compile(r"^(?:Madde|MADDE)\s+(\d+\w*)\s*[-–—]",
                                 re.MULTILINE)


def parse_plain_text(text: str, regulation: str, source_url: str = "",
                     language: str = "en") -> list[Article]:
    """Split plain text on 'Article N' / 'Madde N' headings.

    'Madde' support means Turkish regulations (KVKK, BDDK) parse with
    the same code path. Both standalone headings and mevzuat.gov.tr's
    inline "MADDE 5- (1) ..." style are recognized.
    """
    matches = list(_TEXT_ART_RE.finditer(text))
    if len(matches) < 3:  # standalone style not found; try inline style
        inline = list(_TEXT_ART_INLINE_RE.finditer(text))
        if len(inline) > len(matches):
            matches = inline
    articles: list[Article] = []

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue

        lines = body.split("\n", 1)
        first = lines[0].strip()
        # A short first line is the article title — unless it starts with
        # a numbered paragraph like "(1)", which means the body began on
        # the heading line (Turkish inline style).
        is_title = len(first) < 150 and not first.startswith("(")
        title = first if is_title else ""
        rest = lines[1].strip() if title and len(lines) > 1 else body

        # mevzuat.gov.tr style: the heading sits on the line *before*
        # "MADDE N-" (e.g. "İlgili kişinin hakları\nMADDE 11- (1) ...").
        # That heading is the strongest retrieval signal, so recover it
        # from the text preceding the match when the body has no title.
        if not title:
            preceding = text[:m.start()].rstrip().split("\n")
            if preceding:
                cand = preceding[-1].strip()
                if 0 < len(cand) < 120 and not cand.startswith("("):
                    title = cand

        articles.append(Article(
            regulation=regulation,
            article_number=m.group(1),
            article_title=title,
            chapter="",
            text=_clean(rest),
            source_url=source_url,
            language=language,
        ))
    return articles


# --------------------------------------------------------------------------
# Strategy 3: decision / guideline documents (no "Madde N" headings)
# --------------------------------------------------------------------------

# Kurul kararları and guidance documents structure their operative content
# as a lettered or numbered list ("a) ...", "b) ...", "1) ...") rather than
# statute articles. Turkish lettered lists use the Turkish alphabet order
# (a, b, c, ç, d, e, ...), so the marker class includes ç/ğ/ı/ö/ş/ü.
_DECISION_ITEM_RE = re.compile(
    r"^\s*([a-zçğıöşü]|\d{1,2})[)\.]\s+", re.MULTILINE | re.IGNORECASE)


def parse_decision_text(text: str, regulation: str, document_label: str,
                        doc_type: str = "board_decision", doc_date: str = "",
                        in_force: bool = True, source_url: str = "",
                        language: str = "tr") -> list[Article]:
    """Split a Board decision or guideline into indexable sections.

    These documents don't have "Madde N" headings — parse_plain_text's
    regexes never match — so they need their own strategy: split on the
    lettered/numbered list markers ("a)", "b)", "1)"...) that carry the
    actual operative measures. If fewer than 2 markers are found (e.g. a
    short amendment-notes document with no list structure), the whole
    text is kept as a single section rather than dropped.

    Returned Articles reuse article_number for the item label (e.g. "a")
    so downstream chunking/retrieval code doesn't need a separate field —
    doc_type != "statute" is what tells citation formatting to render
    document_label ("Kurul Kararı 2018/10") instead of "Article N".
    """
    matches = list(_DECISION_ITEM_RE.finditer(text))
    common = dict(regulation=regulation, chapter="", source_url=source_url,
                 language=language, doc_type=doc_type, doc_date=doc_date,
                 in_force=in_force, document_label=document_label)

    if len(matches) < 2:
        body = _clean(text)
        if not body:
            return []
        return [Article(article_number="", article_title="", text=body, **common)]

    articles: list[Article] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = _clean(text[start:end])
        if not body:
            continue
        articles.append(Article(
            article_number=m.group(1).lower(), article_title="",
            text=body, **common))
    return articles


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
