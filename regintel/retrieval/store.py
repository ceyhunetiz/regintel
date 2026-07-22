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
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from regintel import config
from regintel.ingestion.chunker import Chunk


@dataclass
class SearchResult:
    id: str
    text: str
    metadata: dict
    score: float

    @property
    def citation(self) -> str:
        m = self.metadata
        title = f" — {m['article_title']}" if m.get("article_title") else ""
        return f"{m['regulation']}, Article {m['article_number']}{title}"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


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
               regulation: str | None = None) -> list[SearchResult]:
        """Hybrid search with optional regulation filter."""
        where = {"regulation": regulation} if regulation else None
        semantic = self._semantic_search(query, top_k * 3, where) \
            if self.use_embeddings else []
        keyword = self._bm25_search(query, top_k * 3, regulation)
        fused = _rrf([semantic, keyword]) if semantic else keyword
        return fused[:top_k]

    def _semantic_search(self, query, n, where) -> list[SearchResult]:
        n = min(n, max(self.collection.count(), 1))
        res = self.collection.query(query_texts=[query], n_results=n, where=where)
        return [
            SearchResult(id=i, text=d, metadata=m, score=1 - dist)
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0],
                                     res["metadatas"][0], res["distances"][0])
        ]

    def _bm25_search(self, query, n, regulation) -> list[SearchResult]:
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._bm25_ids, scores), key=lambda x: -x[1])
        out = []
        for id_, score in ranked:
            doc, meta = self._bm25_docs[id_]
            if regulation and meta.get("regulation") != regulation:
                continue
            if score <= 0:
                break
            out.append(SearchResult(id=id_, text=doc, metadata=meta, score=score))
            if len(out) >= n:
                break
        return out

    def regulations(self) -> list[str]:
        return sorted({m["regulation"] for _, m in self._all_docs().values()})


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
