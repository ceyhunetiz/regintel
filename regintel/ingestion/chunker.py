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
        id_prefix = f"{article.regulation}-art{article.article_number}"
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


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text else []

    paragraphs = [p for p in text.split("\n") if p.strip()]
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
