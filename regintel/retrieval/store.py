"""Vector store + hybrid retrieval.

Semantic search (ChromaDB + multilingual embeddings) is fused with
keyword search (BM25) using Reciprocal Rank Fusion. Regulatory text is
full of exact tokens — "Article 5", "madde 12", "ICT third-party risk" —
where BM25 outperforms embeddings, and conceptual questions where
embeddings win. RRF gets the best of both without score calibration.

With use_embeddings=False the store runs BM25-only on a pure-Python
JSON document store — no ChromaDB or embedding model needed. Useful for
tests, CI, and trying the pipeline before installing heavy dependencies.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from regintel import config
from regintel.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    id: str
    text: str
    metadata: dict
    score: float

    @property
    def citation(self) -> str:
        m = self.metadata
        part = ""
        if m.get("total_chunks", 1) > 1:
            part = f" (part {m['chunk_index'] + 1} of {m['total_chunks']})"
        if m.get("doc_type", "statute") != "statute" and m.get("document_label"):
            item = f" ({m['article_number']})" if m.get("article_number") else ""
            return f"{m['regulation']}, {m['document_label']}{item}{part}"
        title = f" — {m['article_title']}" if m.get("article_title") else ""
        return f"{m['regulation']}, Article {m['article_number']}{title}{part}"


def _tokenize(text: str) -> list[str]:
    # Python's str.lower() maps Turkish "İ" to "i" + a combining dot
    # (U+0307): "İlgili".lower() != "ilgili". Every indexed Turkish word
    # containing İ ("İlgili", "BİLDİRİM", "MADDE" headings in caps...)
    # therefore tokenized to a form no typed query could ever match,
    # silently blinding the BM25 half of hybrid retrieval to a Turkish
    # chunk's most distinctive terms. Dropping the combining dot after
    # lowering restores the match; the same fold applies at index and
    # query time, so tokens stay consistent.
    return re.findall(r"\w+", text.lower().replace("\u0307", ""))


# Matches a document's own distinctive numeric identifier as it appears in
# document_label ("Kurul Kararı 2019/10" -> "2019/10", "Commission Delegated
# Regulation (EU) 2024/1772" -> "2024/1772"). A bare law number like "7499"
# doesn't fit this shape, so it's handled separately in
# _document_number_map below.
_DOC_NUMBER_RE = re.compile(r"\d{4}/\d{1,5}")

# Matches an explicit "Article N" / "Madde N" mention in EITHER word
# order: English and formal Turkish put the word first ("Article 1",
# "Madde 1"), but natural Turkish puts the number first ("1. Madde",
# "1. Maddeyi", "1. Maddesi") — a request phrased that way ("KVKK 1.
# Maddeyi gösterebilir misin?") matched neither an existing detector
# nor, worse, ordinary hybrid search: a bare digit like "1" is one of
# the least discriminating tokens in this corpus (nearly every
# paragraph starts "(1)..."), so even this exact, unambiguous request
# never reached the top of a ranked search. See _named_article_chunks.
_ARTICLE_NUMBER_RE = re.compile(
    r"\b(?:article|art\.?|madde)\s+(?P<num1>\d+\w*)\b"
    r"|\b(?P<num2>\d+\w*)\.\s*madde\w*",
    re.IGNORECASE)


def _requested_article_numbers(query: str) -> set[str]:
    """Article numbers explicitly named in the query — see
    _ARTICLE_NUMBER_RE. Normalized to the plain digit form used in
    article_number metadata; a transitional "Geçici N" article is
    matched by its own number too since "Geçici" itself doesn't appear
    immediately next to the number in most real phrasings, and a
    request for the wrong (non-Geçici) sibling with the same digits is
    harmless — it just adds one more, genuinely on-topic article.
    """
    return {(m.group("num1") or m.group("num2"))
           for m in _ARTICLE_NUMBER_RE.finditer(query)}


def _where_clause(regulation: str | None, doc_type: str | list[str] | None) -> dict | None:
    """Build a ChromaDB `where` filter from regulation + doc_type, combining
    both with $and when both are given (Chroma rejects a flat dict with
    more than one top-level key)."""
    clauses = []
    if regulation:
        clauses.append({"regulation": regulation})
    if doc_type:
        types = [doc_type] if isinstance(doc_type, str) else list(doc_type)
        clauses.append({"doc_type": {"$in": types}} if len(types) > 1
                       else {"doc_type": types[0]})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


@lru_cache(maxsize=1)
def _embedding_function():
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL)


class RegulationStore:
    def __init__(self, persist_dir: str | None = None, use_embeddings: bool = True):
        self.persist_dir = Path(persist_dir or config.CHROMA_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.use_embeddings = use_embeddings

        if use_embeddings:
            import chromadb  # lazy: only needed for semantic mode
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self.client.get_or_create_collection(
                config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
                embedding_function=_embedding_function(),
            )
        else:
            self.collection = None
            self._docs_path = self.persist_dir / "docs.json"
            self._json_docs: dict[str, tuple[str, dict]] = {}
            if self._docs_path.exists():
                raw = json.loads(self._docs_path.read_text(encoding="utf-8"))
                self._json_docs = {k: (v["text"], v["metadata"])
                                   for k, v in raw.items()}

        self._bm25 = None
        self._bm25_ids: list[str] = []
        self._bm25_docs: dict[str, tuple[str, dict]] = {}
        self._doc_number_map: dict[str, tuple[str, str]] | None = None

    # -- indexing --------------------------------------------------------

    def clear_regulation(self, regulation: str) -> None:
        """Delete all indexed STATUTE chunks for one regulation, before
        re-ingesting the statute after a chunking/parsing change:
        add_chunks() upserts by id, so if a fix changes how many chunks
        an article produces, old chunk ids that are no longer generated
        (e.g. article-N-15..39) would otherwise stay orphaned in the
        index alongside the corrected ones.

        Deliberately statute-only, not "every chunk tagged with this
        regulation": a KVKK Board decision, RTS, or guideline shares
        regulation="KVKK"/"DORA" with the base statute, and a plain
        `python scripts/ingest.py <regulation>` re-ingest only re-adds
        the statute — a blanket delete here silently wiped every one of
        those sibling documents out of the index with no re-add and no
        warning (observed: a routine KVKK re-ingest deleted all six
        KVKK Board decisions/guidelines). clear_document() already
        protects the statute from a document re-ingest; this is the
        same protection in the other direction.
        """
        if self.collection is not None:
            self.collection.delete(where={"$and": [
                {"regulation": regulation}, {"doc_type": "statute"}]})
        else:
            self._json_docs = {k: v for k, v in self._json_docs.items()
                               if not (v[1].get("regulation") == regulation and
                                       v[1].get("doc_type", "statute") == "statute")}
            self._docs_path.write_text(json.dumps(
                {k: {"text": t, "metadata": m}
                 for k, (t, m) in self._json_docs.items()},
                ensure_ascii=False), encoding="utf-8")
        self._bm25 = None
        self._doc_number_map = None

    def clear_document(self, regulation: str, document_label: str) -> None:
        """Delete all indexed chunks for one non-statute document (a Board
        decision or guideline), without touching the statute or any other
        document indexed under the same regulation.

        clear_regulation() is too broad for this: a KVKK Board decision
        shares regulation="KVKK" with the KVKK statute, so re-ingesting the
        decision with clear_regulation() would silently wipe the statute
        out of the index too.
        """
        if self.collection is not None:
            self.collection.delete(where={"$and": [
                {"regulation": regulation}, {"document_label": document_label}]})
        else:
            self._json_docs = {k: v for k, v in self._json_docs.items()
                               if not (v[1].get("regulation") == regulation and
                                       v[1].get("document_label") == document_label)}
            self._docs_path.write_text(json.dumps(
                {k: {"text": t, "metadata": m}
                 for k, (t, m) in self._json_docs.items()},
                ensure_ascii=False), encoding="utf-8")
        self._bm25 = None
        self._doc_number_map = None

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        if self.collection is not None:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                self.collection.upsert(
                    ids=[c.id for c in batch],
                    documents=[c.text for c in batch],
                    metadatas=[c.metadata for c in batch],
                )
        else:
            for c in chunks:
                self._json_docs[c.id] = (c.text, c.metadata)
            self._docs_path.write_text(json.dumps(
                {k: {"text": t, "metadata": m}
                 for k, (t, m) in self._json_docs.items()},
                ensure_ascii=False), encoding="utf-8")
        self._bm25 = None  # force rebuild
        self._doc_number_map = None

    # -- document access ---------------------------------------------------

    def _all_docs(self) -> dict[str, tuple[str, dict]]:
        if self.collection is not None:
            data = self.collection.get(include=["documents", "metadatas"])
            return {i: (d, m) for i, d, m in
                    zip(data["ids"], data["documents"], data["metadatas"])}
        return dict(self._json_docs)

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        self._bm25_docs = self._all_docs()
        self._bm25_ids = list(self._bm25_docs)
        corpus = [_tokenize(self._bm25_docs[i][0]) for i in self._bm25_ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def _document_number_map(self) -> dict[str, tuple[str, str]]:
        """Number string -> (regulation, document_label) for every
        distinctively-numbered non-statute document in the corpus, built
        from document_label itself (e.g. "2024/1772" ->
        ("DORA", "Commission Delegated Regulation (EU) 2024/1772")).

        Lets a question that names a specific decision/regulation by
        number route straight to that document's own chunks (see
        _named_document_chunks) instead of competing on equal footing,
        diluted across dozens of small chunks, in ordinary hybrid search
        — a citation number is the single strongest identifying signal a
        legal question can give, stronger than any topical phrasing.
        """
        if self._doc_number_map is None:
            self._ensure_bm25()
            mapping: dict[str, tuple[str, str]] = {}
            for _, meta in self._bm25_docs.values():
                label = meta.get("document_label")
                if not label:
                    continue
                reg = meta["regulation"]
                for num in _DOC_NUMBER_RE.findall(label):
                    mapping[num] = (reg, label)
                # Bare law numbers ("7499") don't fit the YYYY/NNN shape
                # above but appear at the very start of their label.
                lead = label.split()[0]
                if lead.isdigit() and len(lead) >= 3:
                    mapping[lead] = (reg, label)
            self._doc_number_map = mapping
        return self._doc_number_map

    def _named_document_chunks(self, query: str, regulation: str | None,
                                n: int) -> list[SearchResult]:
        """Chunks of a document the query names explicitly by number,
        ranked by keyword match within just that document — bypasses the
        cross-corpus dilution that lets a single relevant chunk crowd out
        the rest of its own document (see _document_number_map). Returns
        [] if the query names no known document number.
        """
        doc_map = self._document_number_map()
        wanted = {(reg, label) for num, (reg, label) in doc_map.items()
                 if (regulation is None or reg == regulation)
                 and re.search(rf"\b{re.escape(num)}\b", query)}
        if not wanted or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        candidates = [
            SearchResult(id=id_, text=self._bm25_docs[id_][0],
                        metadata=self._bm25_docs[id_][1], score=score)
            for id_, score in zip(self._bm25_ids, scores)
            if (self._bm25_docs[id_][1].get("regulation"),
                self._bm25_docs[id_][1].get("document_label")) in wanted
        ]
        candidates.sort(key=lambda r: -r.score)
        return candidates[:n]

    def _named_article_chunks(self, query: str, regulation: str | None,
                              n: int) -> list[SearchResult]:
        """Chunks of a STATUTE article the query names explicitly by
        number ("Madde 1", "1. Maddeyi", "Article 33") — returned
        directly rather than ranked, since a bare article number is one
        of the least discriminating tokens in this corpus (nearly every
        paragraph starts "(1)...") and even an exact, unambiguous ask
        for a specific article can otherwise lose to unrelated articles
        in ordinary hybrid search (observed live: "KVKK 1. Maddeyi
        gösterebilir misin?" never surfaced KVKK Article 1 in the top 6
        of an otherwise well-formed English retrieval query). Mirrors
        _named_document_chunks's reasoning for non-statute documents,
        including the same `n` cap: a question can name one article by
        number while the actually-correct answer lives in a DIFFERENT
        article entirely (eval case Q7, deliberately: "Does KVKK Article
        9 cover special-category data?" — no, that's Article 6 — needs
        both retrieved to say so). Reserving this article's ENTIRE text
        unconditionally starved that other, uncited article of room in
        the rest of the result set; capping to at most half of top_k
        leaves that room back.

        Only fires when exactly one regulation is already resolved — a
        bare number is ambiguous across regulations, and this makes no
        attempt to disambiguate. Checks both the plain number and its
        "Geçici N" (transitional-article) form, since the word "Geçici"
        doesn't reliably sit next to the number in real phrasing —
        pulling in both when genuinely ambiguous is harmless (one more
        on-topic article), unlike guessing wrong and returning neither.
        """
        if not regulation:
            return []
        numbers = _requested_article_numbers(query)
        if not numbers:
            return []
        wanted = numbers | {f"Geçici {n}" for n in numbers}
        self._ensure_bm25()
        matches = [
            SearchResult(id=id_, text=text, metadata=meta, score=1.0)
            for id_, (text, meta) in self._bm25_docs.items()
            if meta.get("regulation") == regulation
            and meta.get("doc_type", "statute") == "statute"
            and meta.get("article_number") in wanted
        ]
        matches.sort(key=lambda r: (r.metadata.get("article_number", ""),
                                    r.metadata.get("chunk_index", 0)))
        return matches[:n]

    # -- search ------------------------------------------------------------

    def search(self, query: str, top_k: int = config.DEFAULT_TOP_K,
               regulation: str | None = None,
               doc_type: str | list[str] | None = None,
               extra_query: str | None = None) -> list[SearchResult]:
        """Hybrid search with optional regulation and doc_type filters.

        doc_type restricts to "statute" | "board_decision" | "guideline"
        (or a list of those) — e.g. a comparison-mode caller that only
        wants statute text, or a follow-up that specifically wants Board
        decisions once the corpus has them.

        extra_query: an additional phrasing of the same information need,
        searched and fused in alongside `query` rather than replacing it —
        for a same-language corpus (KVKK, entirely Turkish) searched with
        an English-translated query, cross-lingual embedding similarity
        can be markedly weaker than same-language similarity for the
        exact same content, even when the translation is accurate
        (observed: KVKK Article 5's processing conditions, retrievable
        instantly by its own Turkish wording, never surfaced by any
        English rephrasing tried, including the pipeline's real query-
        rewrite output). Passing the original, untranslated question here
        gives that same-language signal a chance to win without
        discarding the translated query's own (real, needed) contribution
        for cross-lingual and BM25-exact-term matches.
        """
        where = _where_clause(regulation, doc_type)
        semantic = self._semantic_search(query, top_k * 3, where) \
            if self.use_embeddings else []
        keyword = self._bm25_search(query, top_k * 3, regulation, doc_type)

        if not extra_query or extra_query == query:
            fused = _rrf([semantic, keyword]) if semantic else keyword
        else:
            extra_semantic = self._semantic_search(extra_query, top_k * 3, where) \
                if self.use_embeddings else []
            extra_keyword = self._bm25_search(extra_query, top_k * 3, regulation, doc_type)
            fused = _rrf([lst for lst in
                         (semantic, keyword, extra_semantic, extra_keyword) if lst])

        number_probe = query if not extra_query else f"{query} {extra_query}"

        # A question naming a specific STATUTE ARTICLE by number ("Madde
        # 1", "1. Maddeyi") gets that article's own chunks directly,
        # ahead of the document-number check below — an article ask is
        # the more specific of the two, and a bare number is too common
        # a token (every paragraph starts "(1)...") to trust to ranking
        # at all. See _named_article_chunks.
        #
        # Additive, not subtracted from top_k, unlike the document-name
        # reservation below: a question can name one article while the
        # actually-correct answer lives in a DIFFERENT article entirely
        # (eval case Q7, deliberately: "Does KVKK Article 9 cover
        # special-category data?" — no, that's Article 6, and the model
        # needs both to say so correctly). Article 6 already ranks into
        # the ordinary top-k on its own for that query; reserving room
        # OUT of top_k for Article 9 pushed it back out. A named
        # document, by contrast, is normally supplementary detail the
        # question wasn't independently asking about, so trimming rest's
        # budget there hasn't caused this problem.
        article_named = self._named_article_chunks(
            number_probe, regulation, n=min(max(top_k // 2, 1), 4))
        if article_named:
            article_ids = {r.id for r in article_named}
            rest = _apply_diversity_cap(
                [r for r in fused if r.id not in article_ids],
                top_k,
                config.MAX_CHUNKS_PER_ARTICLE, config.MAX_CHUNKS_PER_DOCUMENT)
            return article_named + rest

        # A question naming a specific document by number ("7499",
        # "2019/10"...) gets that document's own best-matching chunks
        # reserved up front, since ordinary hybrid ranking can bury all
        # but one chunk of a finely-split document under unrelated
        # higher-scoring content from the rest of the corpus (observed:
        # a 17-chunk amendment note where only its single top chunk ever
        # reached the final result, even though a different one of its
        # own chunks held the actual answer). Reserves at most half of
        # top_k so there's still room for complementary context.
        named = self._named_document_chunks(number_probe, regulation,
                                             n=min(max(top_k // 2, 1), 4))
        if named:
            named_ids = {r.id for r in named}
            rest = _apply_diversity_cap(
                [r for r in fused if r.id not in named_ids],
                max(top_k - len(named), 0),
                config.MAX_CHUNKS_PER_ARTICLE, config.MAX_CHUNKS_PER_DOCUMENT)
            return named + rest

        return _apply_diversity_cap(fused, top_k, config.MAX_CHUNKS_PER_ARTICLE,
                                    config.MAX_CHUNKS_PER_DOCUMENT)

    def _semantic_search(self, query, n, where) -> list[SearchResult]:
        n = min(n, max(self.collection.count(), 1))
        res = self.collection.query(query_texts=[query], n_results=n, where=where)
        kept, dropped = [], 0
        for i, d, m, dist in zip(res["ids"][0], res["documents"][0],
                                 res["metadatas"][0], res["distances"][0]):
            score = 1 - dist
            if score >= config.MIN_SEMANTIC_SCORE:
                kept.append(SearchResult(id=i, text=d, metadata=m, score=score))
            else:
                dropped += 1
        if dropped:
            logger.debug("semantic: dropped %d chunk(s) below MIN_SEMANTIC_SCORE "
                        "(%.2f) for query %r", dropped, config.MIN_SEMANTIC_SCORE, query)
        return kept

    def _bm25_search(self, query, n, regulation, doc_type=None) -> list[SearchResult]:
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        allowed_types = {doc_type} if isinstance(doc_type, str) else \
            set(doc_type) if doc_type else None
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._bm25_ids, scores), key=lambda x: -x[1])
        out = []
        top_score = None
        for id_, score in ranked:
            doc, meta = self._bm25_docs[id_]
            if regulation and meta.get("regulation") != regulation:
                continue
            if allowed_types and meta.get("doc_type", "statute") not in allowed_types:
                continue
            if score <= 0:
                break
            # BM25 scores aren't comparable across queries, so the floor
            # is relative to this query's own top (allowed) match — see
            # config.MIN_BM25_RELATIVE_SCORE. Results stay sorted
            # descending, so once one candidate falls below the floor,
            # every remaining candidate would too.
            if top_score is None:
                top_score = score
            elif score < top_score * config.MIN_BM25_RELATIVE_SCORE:
                logger.debug("bm25: truncating at relative-score floor "
                            "(top=%.3f, cutoff=%.3f) for query %r",
                            top_score, score, query)
                break
            out.append(SearchResult(id=id_, text=doc, metadata=meta, score=score))
            if len(out) >= n:
                break
        return out

    # The three corpus-metadata helpers below iterate the BM25 doc cache
    # (self._bm25_docs, refreshed via _ensure_bm25) instead of calling
    # _all_docs() directly: _all_docs() dumps the ENTIRE Chroma
    # collection over the wire, and _scope_note() calls two of these per
    # question — that was two to three full-collection fetches on every
    # single ask. The cache is already invalidated correctly on any
    # index change (add_chunks/clear_* set _bm25 = None), so this is
    # pure saved work.

    def regulations(self) -> list[str]:
        self._ensure_bm25()
        return sorted({m["regulation"] for _, m in self._bm25_docs.values()})

    def doc_types(self, regulation: str) -> set[str]:
        """Which doc_types are actually indexed for a regulation — lets the
        answer prompt state corpus scope truthfully (e.g. "statute only,
        no Board decisions indexed") instead of the model guessing."""
        self._ensure_bm25()
        return {m.get("doc_type", "statute") for _, m in self._bm25_docs.values()
               if m.get("regulation") == regulation}

    def article_range(self, regulation: str) -> tuple[int, int] | None:
        """Min/max indexed STATUTE article number for a regulation, or
        None if nothing's indexed. Retrieval only ever hands the model
        snippets, not corpus-wide facts — this lets the answer prompt
        state the instrument's real article range explicitly, so the
        model can correctly resolve "does Article N exist" questions
        instead of having no basis to say anything but "not in the
        retrieved sources" either way.

        Restricted to doc_type == "statute": the range is presented (and
        used by _out_of_range_article) as the STATUTE's article range,
        but numeric labels from Kurul-decision items and RTS/ITS
        articles used to leak into it — harmless today only because no
        sub-document's numbering happens to exceed any statute's, i.e.
        by luck rather than by construction.
        """
        self._ensure_bm25()
        nums = [int(m["article_number"]) for _, m in self._bm25_docs.values()
               if m["regulation"] == regulation
               and m.get("doc_type", "statute") == "statute"
               and m["article_number"].isdigit()]
        return (min(nums), max(nums)) if nums else None


def _rrf(result_lists: list[list[SearchResult]],
         k: int = config.RRF_K) -> list[SearchResult]:
    """Reciprocal Rank Fusion across ranked lists."""
    scores: dict[str, float] = {}
    seen: dict[str, SearchResult] = {}
    for results in result_lists:
        for rank, r in enumerate(results):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (k + rank + 1)
            seen.setdefault(r.id, r)
    fused = [SearchResult(id=i, text=seen[i].text, metadata=seen[i].metadata,
                          score=s) for i, s in scores.items()]
    return sorted(fused, key=lambda r: -r.score)


def _apply_diversity_cap(results: list[SearchResult], top_k: int,
                         max_per_article: int,
                         max_per_document: int | None = None) -> list[SearchResult]:
    """Limit how many chunks from a single article/section — AND from a
    single non-statute document as a whole — can occupy the final result
    list, so one long, well-scoring source can't crowd out every other
    relevant one (observed: a single KVKK Art 9 chunk filling 5 of a
    scenario's context slots). The article-level cap alone doesn't bound
    a multi-section Board decision or RTS: each of its sections has a
    distinct article_number label, so none of them ever collide under
    that key even though they all come from the same document — the
    document-level cap closes that gap. Backfills from the capped-out
    overflow if the quota would otherwise leave top_k under-filled —
    diversity is a preference, not a reason to return fewer results than
    asked for when nothing else is available.
    """
    article_counts: dict[tuple, int] = {}
    doc_counts: dict[tuple, int] = {}
    kept, overflow = [], []
    for r in results:
        m = r.metadata
        article_key = (m.get("regulation"), m.get("article_number"))
        doc_label = m.get("document_label")
        doc_key = (m.get("regulation"), doc_label) if doc_label else None
        over_article = article_counts.get(article_key, 0) >= max_per_article
        over_doc = (doc_key is not None and max_per_document is not None and
                   doc_counts.get(doc_key, 0) >= max_per_document)
        if over_article or over_doc:
            overflow.append(r)
        else:
            article_counts[article_key] = article_counts.get(article_key, 0) + 1
            if doc_key is not None:
                doc_counts[doc_key] = doc_counts.get(doc_key, 0) + 1
            kept.append(r)
        if len(kept) >= top_k:
            return kept[:top_k]
    kept.extend(overflow[:max(0, top_k - len(kept))])
    return kept[:top_k]
