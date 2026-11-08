"""LLM provider abstraction.

Three implementations:

* ``ScriptedLLM`` — deterministic, pops responses from a pre-written
  script.  Lets the whole agent stack run and be tested with **zero API
  keys** and **zero network**.
* ``OpenAIProvider`` — talks to any OpenAI-compatible ``/chat/completions``
  endpoint over plain ``urllib`` (no SDK dependency).
* ``OllamaProvider`` — talks to a local Ollama ``/api/chat`` endpoint over
  ``urllib``.

Providers are interchangeable: the agent loop only depends on
``.complete(messages, tools) -> LLMResponse``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .config import Settings
from .schema import LLMResponse, Message, ToolCall, Usage


class LLMError(RuntimeError):
    """Raised when a provider call fails (network / auth / bad response)."""


class LLMProvider:
    model: str = "unknown"

    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Deterministic scripted provider (used for tests + no-key demo)
# --------------------------------------------------------------------------- #
class ScriptedLLM(LLMProvider):
    """Pops one response per ``complete`` call, in order.

    When the script is exhausted it returns an empty completion, which the
    agent loop treats as "stop".
    """

    model = "scripted"

    def __init__(self, responses: List[LLMResponse]):
        self._responses = list(responses)
        self.model = "scripted"

    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        if not self._responses:
            return LLMResponse(content="", finish_reason="stop")
        return self._responses.pop(0)


# --------------------------------------------------------------------------- #
# Helpers shared by the HTTP providers
# --------------------------------------------------------------------------- #
def _messages_to_openai(messages: List[Message]) -> List[Dict[str, Any]]:
    return [m.to_dict() for m in messages]


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout_s: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise LLMError(f"HTTP {e.code} from {url}: {detail[:500]}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"connection error to {url}: {e.reason}") from e
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise LLMError(f"non-JSON response from {url}: {body[:200]}") from e


def _parse_openai_tool_calls(msg: Dict[str, Any]) -> List[ToolCall]:
    out: List[ToolCall] = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        args: Dict[str, Any] = {}
        raw = fn.get("arguments")
        if isinstance(raw, str):
            try:
                args = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw": raw}
        elif isinstance(raw, dict):
            args = raw
        out.append(ToolCall(name=fn.get("name", ""), arguments=args, id=tc.get("id") or f"call_{i}"))
    return out


# --------------------------------------------------------------------------- #
# OpenAI-compatible provider (plain urllib, no SDK)
# --------------------------------------------------------------------------- #
class OpenAIProvider(LLMProvider):
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        self._api_key = settings.openai_api_key
        self._base = settings.openai_base_url
        self.model = settings.openai_model
        self._temperature = settings.temperature
        self._timeout = settings.timeout_s

    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": _messages_to_openai(messages),
            "temperature": self._temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        start = time.time()
        data = _post_json(f"{self._base}/chat/completions", payload, headers, self._timeout)
        latency = (time.time() - start) * 1000.0

        try:
            choice = data["choices"][0]
            msg = choice.get("message") or {}
            usage = Usage(**data.get("usage", {}))
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected OpenAI response shape: {str(data)[:300]}") from e

        return LLMResponse(
            content=msg.get("content") or "",
            tool_calls=_parse_openai_tool_calls(msg),
            usage=usage,
            latency_ms=latency,
            model=self.model,
            finish_reason=choice.get("finish_reason", ""),
        )


# --------------------------------------------------------------------------- #
# Ollama provider (plain urllib)
# --------------------------------------------------------------------------- #
class OllamaProvider(LLMProvider):
    def __init__(self, settings: Settings):
        self._base = settings.ollama_base_url
        self.model = settings.ollama_model
        self._temperature = settings.temperature
        self._timeout = settings.timeout_s

    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": _messages_to_openai(messages),
            "stream": False,
            "options": {"temperature": self._temperature},
        }
        if tools:
            payload["tools"] = tools
        headers = {"Content-Type": "application/json"}
        start = time.time()
        data = _post_json(f"{self._base}/api/chat", payload, headers, self._timeout)
        latency = (time.time() - start) * 1000.0

        msg = data.get("message") or {}
        usage = Usage(
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
            total_tokens=int(data.get("prompt_eval_count", 0)) + int(data.get("eval_count", 0)),
        )
        return LLMResponse(
            content=msg.get("content") or "",
            tool_calls=_parse_openai_tool_calls(msg),
            usage=usage,
            latency_ms=latency,
            model=self.model,
            finish_reason="stop",
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_provider(settings: Optional[Settings] = None) -> LLMProvider:
    from .config import get_settings

    settings = settings or get_settings()
    if settings.llm_provider == "openai":
        return OpenAIProvider(settings)
    if settings.llm_provider == "ollama":
        return OllamaProvider(settings)
    if settings.llm_provider == "scripted":
        # A bare scripted provider answers nothing — callers (tests / demo)
        # should construct ScriptedLLM with an explicit script instead.
        return ScriptedLLM([])
    raise LLMError(f"unknown LLM_PROVIDER: {settings.llm_provider!r}")
