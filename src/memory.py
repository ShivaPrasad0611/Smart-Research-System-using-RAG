"""
memory.py
---------
Lightweight multi-turn conversation memory.

Keeps a bounded history of (question, answer) pairs and formats them into
a short "conversation so far" block that gets prepended to prompts, so the
assistant can resolve follow-up questions like "what about part-time staff?"
without needing a heavyweight memory framework.
"""

from __future__ import annotations

from typing import List, Tuple


class ConversationMemory:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history: List[Tuple[str, str]] = []

    def add(self, question: str, answer: str) -> None:
        self.history.append((question, answer))
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns :]

    def as_prompt_block(self) -> str:
        if not self.history:
            return ""
        lines = ["Previous conversation:"]
        for q, a in self.history:
            lines.append(f"User: {q}")
            lines.append(f"Assistant: {a}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.history = []
