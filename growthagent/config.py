"""Runtime configuration, read from environment variables.

All values have safe defaults so the project runs with zero configuration
(the default ``scripted`` provider needs no API key).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    llm_provider: str = "scripted"  # scripted | openai | ollama
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    temperature: float = 0.0
    timeout_s: float = 60.0

    # agent loop guardrails
    max_steps: int = 8
    token_budget: int = 8000

    # observability
    trace_dir: str = "logs/traces"
    ledger_path: str = "logs/ledger.jsonl"

    # cost estimates (USD per 1k tokens) used only for the ledger report
    price_prompt_per_1k: float = 0.0
    price_completion_per_1k: float = 0.0

    @classmethod
    def from_env(cls) -> "Settings":
        def f(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None or raw == "":
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        def i(name: str, default: int) -> int:
            raw = os.getenv(name)
            if raw is None or raw == "":
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "scripted").strip().lower(),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            temperature=f("LLM_TEMPERATURE", 0.0),
            timeout_s=f("LLM_TIMEOUT_S", 60.0),
            max_steps=i("MAX_STEPS", 8),
            token_budget=i("TOKEN_BUDGET", 8000),
            trace_dir=os.getenv("TRACE_DIR", "logs/traces"),
            ledger_path=os.getenv("LEDGER_PATH", "logs/ledger.jsonl"),
            price_prompt_per_1k=f("PRICE_PROMPT_PER_1K", 0.0),
            price_completion_per_1k=f("PRICE_COMPLETION_PER_1K", 0.0),
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reset_settings() -> None:
    """For tests: clear the cached settings."""
    global _settings
    _settings = None
