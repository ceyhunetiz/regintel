"""LLM layer — Ollama by default, swappable behind a small interface.

Everything runs against localhost; no data leaves the machine.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import requests

from regintel import config


class LLM(ABC):
    @abstractmethod
    def chat(self, system: str, user: str) -> str:
        ...


class OllamaLLM(LLM):
    def __init__(self, model: str = config.OLLAMA_MODEL,
                 base_url: str = config.OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(self, system: str, user: str) -> str:
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                # Disable "thinking" mode (qwen3 etc.): reasoning tokens
                # would eat the whole output budget and add latency.
                "think": False,
                "options": {
                    "temperature": config.LLM_TEMPERATURE,
                    "num_predict": config.LLM_MAX_TOKENS,
                    # Discourage the model from restating the same point
                    "repeat_penalty": 1.15,
                    # Ollama's default context (4096) can silently truncate
                    # comparison prompts; qwen3:8b supports up to 32K.
                    "num_ctx": 16384,
                },
            },
            timeout=300,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        # Thinking models (e.g. qwen3) wrap reasoning in <think> tags —
        # strip it so only the final answer reaches the user.
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        return content.strip()

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

    def chat(self, system: str, user: str) -> str:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": config.LLM_TEMPERATURE,
                "max_tokens": config.LLM_MAX_TOKENS,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


class EchoLLM(LLM):
    """No-LLM fallback: returns the retrieved sources verbatim.

    Lets you develop and demo retrieval quality before Ollama is set up.
    """

    def chat(self, system: str, user: str) -> str:
        return ("[No LLM configured — showing retrieved context only]\n\n" + user)


def get_llm() -> LLM:
    if config.LLM_PROVIDER == "api":
        return ApiLLM()
    llm = OllamaLLM()
    return llm if llm.is_available() else EchoLLM()
