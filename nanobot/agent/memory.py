"""Per-user memory system for persistent agent memory."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.utils.helpers import ensure_dir, safe_filename

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session


class MemoryStore:
    """Per-user memory stored as markdown files."""

    def __init__(self, workspace: Path):
        self.users_dir = ensure_dir(workspace / "memory" / "users")

    def _user_memory_path(self, sender_id: str) -> Path:
        """Return the MEMORY.md path for a specific user."""
        user_dir = ensure_dir(self.users_dir / safe_filename(sender_id))
        return user_dir / "MEMORY.md"

    def read_user_memory(self, sender_id: str) -> str:
        """Read a user's long-term memory."""
        path = self._user_memory_path(sender_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def write_user_memory(self, sender_id: str, content: str) -> None:
        """Write a user's long-term memory."""
        self._user_memory_path(sender_id).write_text(content, encoding="utf-8")

    def get_memory_context(self, sender_id: str | None = None) -> str:
        """Return formatted memory for system prompt injection."""
        if not sender_id:
            return ""
        memory = self.read_user_memory(sender_id)
        return f"## Long-term Memory\n{memory}" if memory else ""

    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        sender_id: str,
        topic: str | None = None,
    ) -> bool:
        """Consolidate session messages into per-user MEMORY.md via LLM.

        The LLM returns the updated memory as plain text (no tool calling).
        Returns True on success, False on failure.
        """
        if not session.messages:
            return True

        logger.info("Memory consolidation for user {}: {} messages", sender_id, len(session.messages))

        lines = []
        for m in session.messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")

        current_memory = self.read_user_memory(sender_id)

        topic_instruction = ""
        if topic:
            topic_instruction = f"\nFocus on: {topic}. Extract and preserve details related to this topic while keeping existing memory intact."

        prompt = f"""Consolidate the conversation below into an updated long-term memory document.
Return ONLY the updated markdown memory — no explanation, no preamble, no code fences.
Include all existing facts plus any new ones from the conversation.
If nothing new is worth remembering, return the existing memory unchanged.{topic_instruction}

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{chr(10).join(lines)}"""

        try:
            response = await provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Return ONLY the updated memory as markdown. No explanation, no wrapping."},
                    {"role": "user", "content": prompt},
                ],
                model=model,
            )

            update = (response.content or "").strip()
            if not update:
                logger.warning("Memory consolidation: LLM returned empty response")
                return False

            if update != current_memory.strip():
                self.write_user_memory(sender_id, update)

            logger.info("Memory consolidation done for user {}", sender_id)
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return False
