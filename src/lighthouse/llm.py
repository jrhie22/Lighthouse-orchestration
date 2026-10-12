"""A thin seam between the agents and whatever language model is behind them.

Agents only know the ``LLMClient`` protocol (``chat(messages, tools, format) -> LLMResponse``).
That gives three implementations with identical behaviour from the agent's point of view:

* ``OllamaClient``  - a local model served by Ollama (the real thing).
* ``ScriptedLLM``   - canned responses, used by tests so the agent loop is verified without a model.
* anything else you want (a hosted API) - implement ``chat`` and pass it in.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .settings import settings


@dataclass
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_message: Any = None  # what to append to the conversation history

    def __post_init__(self):
        if self.raw_message is None:
            self.raw_message = {"role": "assistant", "content": self.content}


class LLMClient(Protocol):
    def chat(self, messages: list, tools: list[Callable] | None = None, format: dict | None = None) -> LLMResponse: ...


class OllamaClient:
    """Local model via Ollama. Tools are passed as plain Python functions; the library derives the
    JSON tool schema from each function's signature and docstring. ``format`` is a JSON schema that
    constrains the reply (structured outputs - local Ollama only)."""

    def __init__(self, host: str | None = None, model: str | None = None):
        import ollama  # imported lazily so the rest of the app runs without the package

        self.model = model or settings.ollama_model
        self._client = ollama.Client(host=host or settings.ollama_host)

    def available(self) -> bool:
        try:
            names = [m.model for m in self._client.list().models]
            return any(n == self.model or n.split(":")[0] == self.model.split(":")[0] for n in names)
        except Exception:
            return False

    def chat(self, messages, tools=None, format=None) -> LLMResponse:
        kwargs: dict[str, Any] = dict(model=self.model, messages=messages, options={"temperature": 0})
        if tools:
            kwargs["tools"] = tools
        if format:
            kwargs["format"] = format
        think = os.environ.get("OLLAMA_THINK")
        if think is not None:
            kwargs["think"] = think == "1"
        msg = self._client.chat(**kwargs).message
        calls = [ToolCall(c.function.name, dict(c.function.arguments or {})) for c in (msg.tool_calls or [])]
        return LLMResponse(content=msg.content or "", tool_calls=calls, raw_message=msg)


class ScriptedLLM:
    """Plays back a fixed list of responses in order. ``calls`` records what the agent sent."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, format=None) -> LLMResponse:
        self.calls.append({"n_messages": len(messages), "tools": [t.__name__ for t in tools or []], "format": bool(format)})
        if not self._responses:
            raise RuntimeError("ScriptedLLM ran out of responses")
        return self._responses.pop(0)


def default_client() -> LLMClient:
    return OllamaClient()
