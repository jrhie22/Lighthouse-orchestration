"""Runtime settings. One mutable singleton so tests can point the app at a temp database."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # SQLite file that holds *everything*: pilot data, cases, outputs, rules, audit log.
    db_path: str = field(default_factory=lambda: _env("LIGHTHOUSE_DB", "data/lighthouse.db"))

    # Local LLM (only used in mode="agent"). Any Ollama model that supports tool calling works.
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen3:8b"))

    # Per-agent time budget. Rules agents finish in milliseconds; LLM agents are slower.
    # When an agent exceeds it, the orchestrator records status=TIMEOUT and moves on
    # (Deliverable 1, section 1.1.6: "or after a defined timeout if an agent has not responded").
    agent_timeout_s: float = field(default_factory=lambda: float(_env("AGENT_TIMEOUT_S", "20")))
    llm_agent_timeout_s: float = field(default_factory=lambda: float(_env("LLM_AGENT_TIMEOUT_S", "180")))

    # How many tool calls an LLM agent may make before it must answer.
    max_tool_calls: int = 5


settings = Settings()
