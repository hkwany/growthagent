"""Token estimation for context budgeting.

Uses ``tiktoken`` when available (accurate for OpenAI models) and falls
back to a deterministic heuristic that is *approximately* correct but has
zero dependencies — enough for enforcing a token budget in the agent loop.
"""

from __future__ import annotations

import re
from typing import List

try:  # optional, accurate
    import tiktoken  # type: ignore

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken is optional
    _ENC = None

_CJK_RE = re.compile(r"[一-鿿぀-ヿ가-힯]")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in ``text``."""
    if not text:
        return 0
    if _ENC is not None:
        try:
            return len(_ENC.encode(text))
        except Exception:  # pragma: no cover
            pass
    # Fallback heuristic:
    # - CJK characters ~ 1 token each
    # - latin words ~ 1 token per 4 chars (min 1)
    # - remaining non-alphanumeric non-space chars ~ 1 token each
    cjk = len(_CJK_RE.findall(text))
    latin_tokens = 0
    for w in _WORD_RE.findall(text):
        latin_tokens += max(1, (len(w) + 3) // 4)
    others = sum(
        1 for ch in text if not (ch.isalnum() or ch.isspace()) and not _CJK_RE.match(ch)
    )
    return cjk + latin_tokens + others


def estimate_messages_tokens(messages: List) -> int:
    """Estimate total tokens for a list of Message objects."""
    total = 0
    for m in messages:
        total += 4  # per-message framing overhead (roughly)
        total += estimate_tokens(getattr(m, "content", "") or "")
        for tc in getattr(m, "tool_calls", None) or []:
            total += estimate_tokens(getattr(tc, "name", "") or "")
            total += estimate_tokens(str(getattr(tc, "arguments", {})))
        name = getattr(m, "name", None)
        if name:
            total += estimate_tokens(str(name))
    return total
