"""FastAPI backend.

Run:  uvicorn regintel.api.main:app --reload
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from regintel.generation.rag import RagPipeline

app = FastAPI(
    title="Regulatory Intelligence Assistant",
    description="Source-backed regulatory Q&A for cybersecurity "
                "governance and compliance. Runs fully locally.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def pipeline() -> RagPipeline:
    return RagPipeline()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    regulation: str | None = Field(None, description="Restrict to one regulation, e.g. 'DORA'")
    top_k: int = Field(6, ge=1, le=20)


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=3)
    regulation_a: str
    regulation_b: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/regulations")
def regulations() -> dict:
    return {"regulations": pipeline().store.regulations()}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    resp = pipeline().ask(req.question, regulation=req.regulation, top_k=req.top_k)
    return resp.to_dict()


@app.post("/compare")
def compare(req: CompareRequest) -> dict:
    available = pipeline().store.regulations()
    for reg in (req.regulation_a, req.regulation_b):
        if reg not in available:
            raise HTTPException(404, f"Regulation '{reg}' not indexed. "
                                     f"Available: {available}")
    resp = pipeline().compare(req.question, req.regulation_a, req.regulation_b)
    return resp.to_dict()


@app.get("/search")
def search(q: str, regulation: str | None = None, top_k: int = 6) -> dict:
    results = pipeline().store.search(q, top_k=top_k, regulation=regulation)
    return {"results": [
        {"citation": r.citation, "text": r.text, "score": r.score}
        for r in results
    ]}
