"""LLM provider boundary (spec section 11). Implemented in M6; the protocol
lives here so later milestones depend on the abstraction, not an SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class LLMAnswer:
    text: str
    model: str
    usage: dict


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str) -> LLMAnswer:
        ...


class OpenAIProvider:
    """Chat Completions via httpx (no openai SDK needed)."""

    def __init__(self, api_key: str = "", model: str = "gpt-4o-mini",
                 base_url: str = "https://api.openai.com/v1") -> None:
        self.name = "openai"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.base_url = base_url

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str) -> LLMAnswer:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                  "temperature": 0.0},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]["content"]
        return LLMAnswer(text=msg, model=data.get("model", self.model),
                         usage=data.get("usage", {}))


class AnthropicProvider:
    """Messages API via httpx (no anthropic SDK needed)."""

    def __init__(self, api_key: str = "", model: str = "claude-sonnet-4-20250514") -> None:
        self.name = "anthropic"
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str) -> LLMAnswer:
        import httpx

        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            json={"model": self.model, "max_tokens": 1024, "temperature": 0.0,
                  "system": system,
                  "messages": [{"role": "user", "content": user}]},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []))
        return LLMAnswer(text=text, model=data.get("model", self.model),
                         usage=data.get("usage", {}))


def get_llm_provider() -> LLMProvider | None:
    """Configured provider, or None when no key is set (template mode)."""
    which = os.environ.get("LLM_PROVIDER", "openai").lower()
    provider = AnthropicProvider() if which == "anthropic" else OpenAIProvider()
    return provider if provider.configured else None
