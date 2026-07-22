"""Split parsed articles into indexable chunks.

Chunks never cross article boundaries — that is what keeps citations
honest. Long articles are split at paragraph boundaries with a small
overlap so no requirement is cut in half.
"""

from __future__ import annotations

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
    chunks = []
    for i, piece in enumerate(pieces):
        chunks.append(Chunk(
            id=f"{article.regulation}-art{article.article_number}-{i}",
            # Prepend context so the embedding "knows" where this text lives
            text=(f"{article.regulation}, Article {article.article_number}"
                  f"{' — ' + article.article_title if article.article_title else ''}:\n"
                  f"{piece}"),
            metadata={
                "regulation": article.regulation,
                "article_number": article.article_number,
                "article_title": article.article_title,
                "chapter": article.chapter,
                "chunk_index": i,
                "total_chunks": len(pieces),
                "source_url": article.source_url,
                "language": article.language,
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
