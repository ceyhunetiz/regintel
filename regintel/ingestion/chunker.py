"""Split parsed articles into indexable chunks.

Chunks never cross article boundaries — that is what keeps citations
honest. Long articles are split at paragraph boundaries with a small
overlap so no requirement is cut in half.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from regintel import config
from regintel.ingestion.parser import Article


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_article(article: Article,
                  max_chars: int = config.MAX_CHUNK_CHARS,
                  overlap: int = config.CHUNK_OVERLAP_CHARS) -> list[Chunk]:
    pieces = _split_text(article.text, max_chars, overlap)
    is_statute = article.doc_type == "statute"
    # Statute ids/text stay exactly as before (ids are stable across
    # re-ingests and existing tests pin this format). Non-statute
    # documents (Board decisions, guidelines) use their document_label —
    # "Article N" would be actively wrong for those.
    if is_statute:
        # Slugify for the id only ("Geçici 3" -> "Gecici-3"): plain
        # numeric/alphanumeric article numbers pass through unchanged, so
        # existing statute chunk ids stay byte-identical across
        # re-ingests.
        num_slug = re.sub(r"[^0-9A-Za-z]+", "-",
                          article.article_number.replace("ç", "c").replace("Ç", "C")
                          .replace("ı", "i").replace("İ", "I")).strip("-")
        id_prefix = f"{article.regulation}-art{num_slug}"
        label = f"Article {article.article_number}"
        if article.article_title:
            label += f" — {article.article_title}"
    else:
        slug = re.sub(r"[^a-z0-9]+", "-", article.document_label.lower()).strip("-") or "doc"
        item = f"-item-{article.article_number}" if article.article_number else ""
        id_prefix = f"{article.regulation}-{slug}{item}"
        label = article.document_label or "Document"
        if article.article_number:
            label += f" ({article.article_number})"

    chunks = []
    for i, piece in enumerate(pieces):
        chunks.append(Chunk(
            id=f"{id_prefix}-{i}",
            # Prepend context so the embedding "knows" where this text lives
            text=f"{article.regulation}, {label}:\n{piece}",
            metadata={
                "regulation": article.regulation,
                "article_number": article.article_number,
                "article_title": article.article_title,
                "chapter": article.chapter,
                "chunk_index": i,
                "total_chunks": len(pieces),
                "source_url": article.source_url,
                "language": article.language,
                "doc_type": article.doc_type,
                "doc_date": article.doc_date,
                "in_force": article.in_force,
                "document_label": article.document_label,
            },
        ))
    return chunks


def chunk_articles(articles: list[Article]) -> list[Chunk]:
    out: list[Chunk] = []
    for a in articles:
        out.extend(chunk_article(a))
    return out


# EU-legislative articles number their top-level paragraphs on their own
# line ("1.\xa0\xa0\xa0As part of..."), with lettered sub-items ("(a)")
# and their content each split across separate lines too. A numbered
# top-level paragraph is an independently citable provision.
_NUMBERED_PARA_RE = re.compile(r"^\d{1,2}[.\xa0]")


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text else []

    lines = [p for p in text.split("\n") if p.strip()]

    if any(_NUMBERED_PARA_RE.match(l) for l in lines):
        return _split_by_numbered_paragraphs(lines, max_chars, overlap)
    return _pack_paragraphs(lines, max_chars, overlap)


def _split_by_numbered_paragraphs(lines: list[str], max_chars: int,
                                  overlap: int) -> list[str]:
    """One chunk per numbered top-level paragraph, never packing several
    together into a shared character budget (unlike _pack_paragraphs
    below). A lettered sub-item line ("(a)") and its content line merge
    into the numbered paragraph they belong to rather than starting a
    new unit.

    Packing multiple numbered paragraphs together by character count
    alone diluted a short, specific provision's retrieval signal with
    its neighbours' unrelated vocabulary — observed directly: DORA
    Art 11(10)'s "financial entities... shall report... upon [the
    competent authority's] request an estimation of aggregated annual
    costs and losses" sat in a chunk shared with earlier paragraphs
    about business-continuity test result copies, and failed to surface
    in the top 8 results even for a query using its own exact wording.
    One paragraph per chunk keeps each provision's own vocabulary as
    the chunk's dominant signal.
    """
    units: list[str] = []
    for line in lines:
        if _NUMBERED_PARA_RE.match(line) or not units:
            units.append(line)
        else:
            units[-1] = units[-1] + "\n" + line

    pieces: list[str] = []
    for unit in units:
        while len(unit) > max_chars:
            cut = unit.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 2 else max_chars
            pieces.append(unit[:cut].strip())
            unit = unit[max(cut - overlap, 0):].strip()
        if unit:
            pieces.append(unit)
    return pieces


def _pack_paragraphs(paragraphs: list[str], max_chars: int,
                     overlap: int) -> list[str]:
    """General-purpose strategy for text with no numbered-paragraph
    structure: greedily pack lines up to a character budget."""
    pieces: list[str] = []
    current = ""

    for para in paragraphs:
        # A single paragraph longer than max_chars gets hard-split on sentences
        while len(para) > max_chars:
            cut = para.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 2 else max_chars
            if current:
                pieces.append(current)
                current = ""
            pieces.append(para[:cut].strip())
            para = para[max(cut - overlap, 0):].strip()

        if len(current) + len(para) + 1 > max_chars:
            pieces.append(current)
            # start next piece with tail of previous for continuity
            current = (current[-overlap:] + "\n" + para).strip() if overlap else para
        else:
            current = f"{current}\n{para}".strip()

    if current:
        pieces.append(current)
    return pieces
