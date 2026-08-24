"""LLM layer — Ollama by default, swappable behind a small interface.

Everything runs against localhost; no data leaves the machine.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator

import requests

from regintel import config


def _max_tokens(reasoning: bool) -> int:
    """A thinking model can spend several thousand tokens reasoning
    before its actual answer starts — config.LLM_MAX_TOKENS is sized for
    a non-reasoning response and would truncate the answer to nothing
    once reasoning is on."""
    return config.LLM_MAX_TOKENS_REASONING if reasoning else config.LLM_MAX_TOKENS


# Returned by _strip_thinking when content is None (see its docstring).
# Exported as a named constant, not just an inline string, so a caller
# that needs to tell "the model genuinely produced this sentence" apart
# from "there is no real content at all" can compare against it rather
# than re-typing (and risking drifting out of sync with) the literal
# text. Concretely needed by rag.py's auxiliary calls (decompose, query
# rewrite, regulation triage): these request reasoning=False, but a
# "hybrid" model's own reasoning toggle isn't a hard guarantee — it can
# still spend tokens reasoning anyway (confirmed directly: a decompose
# call against qwen3-32b with reasoning explicitly disabled still used
# 1,375 reasoning tokens) and exhaust config.LLM_MAX_TOKENS, which is
# sized for a short non-reasoning response and has no headroom for
# that. Without this constant, the resulting placeholder sentence was
# silently parsed as if it were a real (single, garbled) sub-question
# or rewritten query, rather than recognised as the failure it is.
REASONING_EXHAUSTED_MESSAGE = (
    "[The model ran out of its reasoning budget before writing an answer. "
    "Try a shorter or simpler question, or retry.]")


def _strip_thinking(content: str | None) -> str:
    """Thinking models (e.g. qwen3, on Ollama or an API host) wrap
    reasoning in <think> tags — strip it so only the final answer
    reaches the user. Shared by every backend that might serve a
    thinking-mode model, not just Ollama's: an API host serving the
    same model family (e.g. OpenRouter's qwen/qwen3-14b) has no
    equivalent to Ollama's "think": False request flag, so the same
    stripping is needed as a safety net there too.

    content can be None: some OpenRouter-hosted thinking models return
    their reasoning in a separate `message.reasoning` field entirely
    (not <think> tags inside content at all) and `content` itself comes
    back null if the token budget runs out mid-reasoning, before any
    answer text was generated (confirmed directly: a real compare-mode
    prompt used 11,584 reasoning tokens against an 8,000-token budget —
    finish_reason "length", content None). A larger budget
    (LLM_MAX_TOKENS_REASONING) is the real fix, but no fixed budget can
    guarantee covering every prompt, so this returns a clear, visible
    signal (REASONING_EXHAUSTED_MESSAGE) instead of crashing the whole
    pipeline on `None` — a user seeing this message can retry or
    simplify the question; a stack trace just looks like the tool is
    broken.
    """
    if content is None:
        return REASONING_EXHAUSTED_MESSAGE
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def _stream_strip_thinking(chunks: Iterator[str]) -> Iterator[str]:
    """Streaming counterpart of _strip_thinking: filters <think>...</think>
    spans out of a token stream as it flows, holding back just enough text
    to handle a tag split across chunk boundaries.

    _strip_thinking only ever ran on the non-streaming chat() path — the
    Streamlit UI consumes stream_chat(), where a thinking model's
    reasoning used to reach the user verbatim (and then flow into
    cited_sources(), which would try to ground stray [n] markers inside
    the reasoning against real sources).
    """
    open_tag, close_tag = "<think>", "</think>"
    buf, thinking = "", False
    for piece in chunks:
        buf += piece
        out = ""
        while buf:
            if thinking:
                end = buf.find(close_tag)
                if end == -1:
                    # keep a tail in case "</think>" spans two chunks
                    buf = buf[-(len(close_tag) - 1):]
                    break
                buf = buf[end + len(close_tag):]
                thinking = False
            else:
                start = buf.find(open_tag)
                if start == -1:
                    # hold back a possible partial "<think" at the end
                    safe = len(buf)
                    for k in range(len(open_tag) - 1, 0, -1):
                        if buf.endswith(open_tag[:k]):
                            safe = len(buf) - k
                            break
                    out += buf[:safe]
                    buf = buf[safe:]
                    break
                out += buf[:start]
                buf = buf[start + len(open_tag):]
                thinking = True
        if out:
            yield out
    if buf and not thinking:
        yield buf


class LLM(ABC):
    @abstractmethod
    def chat(self, system: str, user: str, reasoning: bool = False) -> str:
        """reasoning: let a thinking-capable model actually think before
        answering, for backends that support toggling it (currently
        Ollama's "think" and OpenRouter's "reasoning" fields). Defaults
        to off: query rewriting, scenario decomposition, and regulation
        classification are short, mechanical transformation tasks that
        gain little from extended reasoning but would each separately
        pay its latency and token cost if left on by default — a single
        ask() already makes 2-3 of these auxiliary calls before the one
        call that should actually reason (the final answer generation).
        Callers doing that final generation pass reasoning=True
        explicitly; every auxiliary call site is left at the default.
        """
        ...

    def stream_chat(self, system: str, user: str, reasoning: bool = False) -> Iterator[str]:
        """Default: no real streaming, yield the full reply as one chunk.

        Subclasses override this when the backend supports token streaming.
        """
        yield self.chat(system, user, reasoning=reasoning)


class OllamaLLM(LLM):
    def __init__(self, model: str = config.OLLAMA_MODEL,
                 base_url: str = config.OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(self, system: str, user: str, reasoning: bool = False) -> str:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                # "think": False by default (qwen3 etc.): reasoning
                # tokens would eat the whole output budget and add
                # latency for the short, mechanical auxiliary calls this
                # defaults to serving — see LLM.chat's reasoning
                # docstring. The caller doing final answer generation
                # opts in explicitly.
                "think": reasoning,
                "options": {
                    "temperature": config.LLM_TEMPERATURE,
                    "seed": config.LLM_SEED,
                    "num_predict": _max_tokens(reasoning),
                    # Discourage the model from restating the same point.
                    # Ollama's default repeat_last_n=64 wasn't enough to
                    # break a real observed failure mode: greedy decoding
                    # (temperature=0, deliberate — see LLM_TEMPERATURE's
                    # comment) degenerating into ~20 verbatim repetitions
                    # of one ~30-word sentence until cut off by
                    # num_predict (v3 eval, case D4) — a 30-word sentence
                    # is ~40-50 tokens, right at the edge of the default
                    # window. repeat_last_n=256 (widening the lookback,
                    # not the penalty strength) fixed that case cleanly.
                    # repeat_penalty itself must stay moderate: raising it
                    # to 1.3 was tried and made things WORSE on a
                    # different case — instead of literal repetition, the
                    # model spiralled into incoherent unrelated-vocabulary
                    # word-salad for the rest of the token budget, a
                    # known llama.cpp/Ollama failure mode when the
                    # penalty is too aggressive. 1.15 was the original,
                    # untuned value; both values stay deterministic
                    # (still greedy — this adjusts logits, not sampling).
                    "repeat_penalty": 1.15,
                    "repeat_last_n": 256,
                    # Ollama's default context (4096) can silently truncate
                    # comparison prompts; qwen3:8b supports up to 32K.
                    "num_ctx": 16384,
                },
            },
            # 300s wasn't enough for a real compare-mode call (longer
            # prompt, more sources) on this machine — observed a hard
            # ReadTimeout mid-eval. Larger local models are slower per
            # token, so this only gets tighter, not looser, as the model
            # size grows; 600s gives real headroom without masking a
            # genuinely hung request forever.
            timeout=600,
        )
        resp.raise_for_status()
        return _strip_thinking(resp.json()["message"]["content"])

    def stream_chat(self, system: str, user: str, reasoning: bool = False) -> Iterator[str]:
        # Safety net regardless of the reasoning flag: strips any
        # <think> span that reaches the stream either way.
        return _stream_strip_thinking(self._stream_raw(system, user, reasoning))

    def _stream_raw(self, system: str, user: str, reasoning: bool = False) -> Iterator[str]:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": True,
                "think": reasoning,
                "options": {
                    "temperature": config.LLM_TEMPERATURE,
                    "seed": config.LLM_SEED,
                    "num_predict": _max_tokens(reasoning),
                    # 1.15, matching chat() exactly — this method briefly
                    # diverged at 1.3, the value chat()'s own comment
                    # documents as tried-and-rejected (word-salad failure
                    # mode), so the streaming UI ran with the known-bad
                    # setting while evals (which call chat()) ran with the
                    # good one.
                    "repeat_penalty": 1.15,
                    "repeat_last_n": 256,
                    "num_ctx": 16384,
                },
            },
            # 300s wasn't enough for a real compare-mode call (longer
            # prompt, more sources) on this machine — observed a hard
            # ReadTimeout mid-eval. Larger local models are slower per
            # token, so this only gets tighter, not looser, as the model
            # size grows; 600s gives real headroom without masking a
            # genuinely hung request forever.
            timeout=600,
            stream=True,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece
            if chunk.get("done"):
                break

    def is_available(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/api/tags", timeout=3).ok
        except requests.RequestException:
            return False


class ApiLLM(LLM):
    """OpenAI-compatible API client (OpenAI, Azure OpenAI, OpenRouter, Groq...).

    NOTE: using a public API sends query text and retrieved regulation
    excerpts to an external provider. For bank deployment, only use an
    enterprise endpoint (e.g. Azure OpenAI in the org's own tenant) that
    contractually keeps data inside the organization.
    """

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None):
        import os
        self.base_url = (base_url or config.API_BASE_URL).rstrip("/")
        self.model = model or config.API_MODEL
        self.api_key = api_key or os.environ.get("REGINTEL_API_KEY", "")

    def _payload(self, system: str, user: str, reasoning: bool = False) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": _max_tokens(reasoning),
            "seed": config.LLM_SEED,
        }
        # The API has no equivalent of Ollama's "think" flag; "reasoning"
        # is OpenRouter's own field for the same toggle — only send it
        # there, since other OpenAI-compatible hosts can reject unknown
        # request fields. Off by default: without this, a thinking-
        # capable model burns tokens/latency reasoning on every call,
        # including the short mechanical ones (query rewrite, scenario
        # decomposition, regulation classification) that don't need it —
        # see LLM.chat's reasoning docstring for why only the final
        # answer call opts in.
        if "openrouter" in self.base_url:
            payload["reasoning"] = {"enabled": reasoning}
        return payload

    def chat(self, system: str, user: str, reasoning: bool = False) -> str:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=self._payload(system, user, reasoning),
            # A thinking model can spend a long time reasoning before the
            # answer starts — 120s was already none too generous for a
            # non-reasoning compare-mode call; give reasoning real room.
            timeout=300 if reasoning else 120,
        )
        resp.raise_for_status()
        return _strip_thinking(resp.json()["choices"][0]["message"]["content"])

    def stream_chat(self, system: str, user: str, reasoning: bool = False) -> Iterator[str]:
        # _stream_strip_thinking filters <think> spans regardless of the
        # reasoning flag, as a safety net either way.
        return _stream_strip_thinking(self._stream_raw(system, user, reasoning))

    def _stream_raw(self, system: str, user: str, reasoning: bool = False) -> Iterator[str]:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={**self._payload(system, user, reasoning), "stream": True},
            timeout=300 if reasoning else 120,
            stream=True,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = line[len(b"data: "):]
            if data.strip() == b"[DONE]":
                break
            delta = json.loads(data)["choices"][0]["delta"].get("content")
            if delta:
                yield delta


class EchoLLM(LLM):
    """No-LLM fallback: returns the retrieved sources verbatim.

    Lets you develop and demo retrieval quality before Ollama is set up.
    """

    def chat(self, system: str, user: str, reasoning: bool = False) -> str:
        return ("[No LLM configured — showing retrieved context only]\n\n" + user)


def get_llm() -> LLM:
    if config.LLM_PROVIDER == "api":
        return ApiLLM()
    llm = OllamaLLM()
    return llm if llm.is_available() else EchoLLM()
