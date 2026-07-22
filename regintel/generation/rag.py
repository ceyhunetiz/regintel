"""RAG pipeline: retrieve -> assemble prompt -> generate -> return
answer with the sources that back it."""

from __future__ import annotations

from dataclasses import dataclass, field

from regintel import config
from regintel.generation import prompts
from regintel.generation.llm import LLM, EchoLLM, get_llm
from regintel.retrieval.store import RegulationStore, SearchResult


@dataclass
class RagResponse:
    answer: str
    sources: list[SearchResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [
                {"citation": s.citation, "text": s.text,
                 "metadata": s.metadata, "score": s.score}
                for s in self.sources
            ],
        }


class RagPipeline:
    def __init__(self, store: RegulationStore | None = None,
                 llm: LLM | None = None):
        self.store = store or RegulationStore()
        self.llm = llm or get_llm()

    def _retrieval_query(self, question: str) -> str:
        """Rewrite the question into a short English search query.

        Regulations are indexed in English; a Turkish question would miss
        BM25 entirely and weaken semantic search. Falls back to the raw
        question if rewriting fails.
        """
        if isinstance(self.llm, EchoLLM):
            return question
        try:
            q = self.llm.chat(prompts.QUERY_REWRITE_PROMPT, question).strip()
            return q[:300] if q else question
        except Exception:
            return question

    def ask(self, question: str, regulation: str | None = None,
            top_k: int = config.DEFAULT_TOP_K) -> RagResponse:
        """Answer a question, optionally restricted to one regulation."""
        query = self._retrieval_query(question)
        results = self.store.search(query, top_k=top_k, regulation=regulation)
        prompt = prompts.ANSWER_TEMPLATE.format(
            sources=prompts.format_sources(results), question=question)
        answer = self.llm.chat(prompts.SYSTEM_PROMPT, prompt)
        return RagResponse(answer=answer, sources=results)

    def compare(self, question: str, reg_a: str, reg_b: str,
                top_k_each: int = 4) -> RagResponse:
        """Compare two regulations on a topic.

        Retrieval runs separately per regulation (metadata-filtered) so
        both sides are actually represented in the context — a single
        unfiltered search often returns chunks from only one framework.
        """
        query = self._retrieval_query(question)
        results_a = self.store.search(query, top_k=top_k_each, regulation=reg_a)
        results_b = self.store.search(query, top_k=top_k_each, regulation=reg_b)
        prompt = prompts.COMPARE_TEMPLATE.format(
            reg_a=reg_a, reg_b=reg_b,
            sources_a=prompts.format_sources(results_a),
            sources_b=prompts.format_sources(results_b, start=len(results_a) + 1),
            question=question,
        )
        answer = self.llm.chat(prompts.SYSTEM_PROMPT, prompt)
        return RagResponse(answer=answer, sources=results_a + results_b)
