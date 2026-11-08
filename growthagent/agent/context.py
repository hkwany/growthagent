"""Context management: token budgeting, sliding window and summarization.

This is the "Context Engineering" piece — enforcing a token budget and
compressing history so long agent runs don't blow the model context.
"""

from __future__ import annotations

from typing import List, Optional

from ..schema import Message
from ..tokenizer import estimate_messages_tokens


def summarize(text: str, max_chars: int = 240) -> str:
    """Naive deterministic summarizer (head + tail).

    A real deployment swaps this for an LLM summarizer — the interface is
    intentionally trivial so it can be replaced without touching the loop.
    """
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


class ContextManager:
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens

    def total_tokens(self, messages: List[Message]) -> int:
        return estimate_messages_tokens(messages)

    def over_budget(self, messages: List[Message]) -> bool:
        return self.total_tokens(messages) > self.max_tokens

    def trim(
        self,
        messages: List[Message],
        keep_recent: int = 6,
    ) -> List[Message]:
        """Drop the middle of the history, keeping system + task + recent.

        Keeps: every ``system`` message, the first ``user`` message (the
        task), and the last ``keep_recent`` messages.  Replaces whatever
        was dropped with a single summary marker message.
        """
        if not messages:
            return messages

        system_msgs = [m for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]

        first_user_idx = next((i for i, m in enumerate(non_system) if m.role == "user"), 0)
        task_msg = non_system[first_user_idx]
        recent = non_system[-(keep_recent):]

        dropped = [m for m in non_system if m is not task_msg and m not in recent]
        if not dropped:
            return messages

        marker = Message(
            role="system",
            content=(
                f"[history compressed: {len(dropped)} earlier messages removed; "
                "their tool results are summarized below]\n"
                + summarize(" ; ".join(m.content[:200] for m in dropped if m.content))
            ),
        )
        return system_msgs + [task_msg, marker] + recent
