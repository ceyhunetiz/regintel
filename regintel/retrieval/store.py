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
    return re.findall(r"\w+", text.lower())


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

    # -- indexing --------------------------------------------------------

    def clear_regulation(self, regulation: str) -> None:
        """Delete all indexed chunks for one regulation.

        Needed before re-ingesting after a chunking/parsing change:
        add_chunks() upserts by id, so if a fix changes how many chunks
        an article produces, old chunk ids that are no longer generated
        (e.g. article-N-15..39) would otherwise stay orphaned in the
        index alongside the corrected ones.
        """
        if self.collection is not None:
            self.collection.delete(where={"regulation": regulation})
        else:
            self._json_docs = {k: v for k, v in self._json_docs.items()
                               if v[1].get("regulation") != regulation}
            self._docs_path.write_text(json.dumps(
                {k: {"text": t, "metadata": m}
                 for k, (t, m) in self._json_docs.items()},
                ensure_ascii=False), encoding="utf-8")
        self._bm25 = None

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

    # -- search ------------------------------------------------------------

    def search(self, query: str, top_k: int = config.DEFAULT_TOP_K,
               regulation: str | None = None,
               doc_type: str | list[str] | None = None) -> list[SearchResult]:
        """Hybrid search with optional regulation and doc_type filters.

        doc_type restricts to "statute" | "board_decision" | "guideline"
        (or a list of those) — e.g. a comparison-mode caller that only
        wants statute text, or a follow-up that specifically wants Board
        decisions once the corpus has them.
        """
        where = _where_clause(regulation, doc_type)
        semantic = self._semantic_search(query, top_k * 3, where) \
            if self.use_embeddings else []
        keyword = self._bm25_search(query, top_k * 3, regulation, doc_type)
        fused = _rrf([semantic, keyword]) if semantic else keyword
        return _apply_diversity_cap(fused, top_k, config.MAX_CHUNKS_PER_ARTICLE)

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

    def regulations(self) -> list[str]:
        return sorted({m["regulation"] for _, m in self._all_docs().values()})

    def doc_types(self, regulation: str) -> set[str]:
        """Which doc_types are actually indexed for a regulation — lets the
        answer prompt state corpus scope truthfully (e.g. "statute only,
        no Board decisions indexed") instead of the model guessing."""
        return {m.get("doc_type", "statute") for _, m in self._all_docs().values()
               if m.get("regulation") == regulation}

    def article_range(self, regulation: str) -> tuple[int, int] | None:
        """Min/max indexed article number for a regulation, or None if
        nothing's indexed. Retrieval only ever hands the model snippets,
        not corpus-wide facts — this lets the answer prompt state the
        instrument's real article range explicitly, so the model can
        correctly resolve "does Article N exist" questions instead of
        having no basis to say anything but "not in the retrieved
        sources" either way.
        """
        nums = [int(m["article_number"]) for _, (_, m) in self._all_docs().items()
               if m["regulation"] == regulation and m["article_number"].isdigit()]
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
                         max_per_article: int) -> list[SearchResult]:
    """Limit how many chunks from a single article/section can occupy the
    final result list, so one long, well-scoring article can't crowd out
    every other relevant one (observed: a single KVKK Art 9 chunk filling
    5 of a scenario's context slots). Backfills from the capped-out
    overflow if the quota would otherwise leave top_k under-filled —
    diversity is a preference, not a reason to return fewer results than
    asked for when nothing else is available.
    """
    counts: dict[tuple, int] = {}
    kept, overflow = [], []
    for r in results:
        key = (r.metadata.get("regulation"), r.metadata.get("article_number"))
        if counts.get(key, 0) < max_per_article:
            counts[key] = counts.get(key, 0) + 1
            kept.append(r)
        else:
            overflow.append(r)
        if len(kept) >= top_k:
            return kept[:top_k]
    kept.extend(overflow[:max(0, top_k - len(kept))])
    return kept[:top_k]
