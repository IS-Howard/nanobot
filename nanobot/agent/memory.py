"""Memory system: global (IDENTITY.md) + per-user (MEMORY.md)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from nanobot.utils.helpers import ensure_dir, safe_filename

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session

_MEMORIZE_GLOBAL_RE = re.compile(r"<memorize_global>(.*?)</memorize_global>", re.DOTALL)
_MEMORIZE_USER_RE = re.compile(r"<memorize_user>(.*?)</memorize_user>", re.DOTALL)


def _strip_memorize_tags(text: str) -> str:
    """Remove any <memorize_global>/<memorize_user> tags from text."""
    text = _MEMORIZE_GLOBAL_RE.sub("", text)
    text = _MEMORIZE_USER_RE.sub("", text)
    return text.strip()


def extract_memorize_tags(text: str) -> tuple[str, list[str], list[str]]:
    """Extract <memorize_global> and <memorize_user> tags from LLM response.

    Returns (clean_text, global_facts, user_facts).
    Tags are stripped from the returned text.
    """
    global_facts = [f.strip() for f in _MEMORIZE_GLOBAL_RE.findall(text) if f.strip()]
    user_facts = [f.strip() for f in _MEMORIZE_USER_RE.findall(text) if f.strip()]
    clean = _MEMORIZE_GLOBAL_RE.sub("", text)
    clean = _MEMORIZE_USER_RE.sub("", clean).strip()
    return clean, global_facts, user_facts


class MemoryStore:
    """Global memory (IDENTITY.md) + per-user memory (MEMORY.md)."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.users_dir = ensure_dir(workspace / "memory" / "users")

    # ── Global memory (IDENTITY.md) ──────────────────────────────

    def _global_memory_path(self) -> Path:
        return self.workspace / "IDENTITY.md"

    def read_global_memory(self) -> str:
        path = self._global_memory_path()
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write_global_memory(self, content: str) -> None:
        self._global_memory_path().write_text(content, encoding="utf-8")

    def append_global_memory(self, fact: str) -> None:
        """Append a single fact line to global memory."""
        current = self.read_global_memory()
        updated = current.rstrip("\n") + f"\n- {fact}\n" if current else f"- {fact}\n"
        self.write_global_memory(updated)

    # ── Per-user memory (MEMORY.md) ──────────────────────────────

    def _user_memory_path(self, sender_id: str) -> Path:
        user_dir = ensure_dir(self.users_dir / safe_filename(sender_id))
        return user_dir / "MEMORY.md"

    def read_user_memory(self, sender_id: str) -> str:
        path = self._user_memory_path(sender_id)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write_user_memory(self, sender_id: str, content: str) -> None:
        self._user_memory_path(sender_id).write_text(content, encoding="utf-8")

    def append_user_memory(self, sender_id: str, fact: str) -> None:
        """Append a single fact line to a user's private memory."""
        current = self.read_user_memory(sender_id)
        updated = current.rstrip("\n") + f"\n- {fact}\n" if current else f"- {fact}\n"
        self.write_user_memory(sender_id, updated)

    def get_memory_context(self, sender_id: str | None = None) -> str:
        """Return formatted per-user memory for system prompt injection."""
        if not sender_id:
            return ""
        memory = self.read_user_memory(sender_id)
        return f"## Long-term Memory\n{memory}" if memory else ""

    # ── Helpers ──────────────────────────────────────────────────

    def _format_conversation(self, session: Session) -> str:
        lines = []
        for m in session.messages:
            if not m.get("content"):
                continue
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
        return "\n".join(lines)

    @staticmethod
    def _parse_json_response(raw: str) -> dict | None:
        """Parse LLM JSON response, stripping markdown code fences if present."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    # ── Consolidation ────────────────────────────────────────────

    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        sender_id: str,
        topic: str | None = None,
        system_prompt: str = "",
    ) -> bool:
        """Consolidate session into global IDENTITY.md + per-user MEMORY.md.

        Returns True on success, False on failure.
        """
        if not session.messages:
            return True

        logger.info("Memory consolidation for user {}: {} messages", sender_id, len(session.messages))

        conversation = self._format_conversation(session)
        current_global = self.read_global_memory()
        current_user = self.read_user_memory(sender_id)

        topic_instruction = ""
        if topic:
            topic_instruction = f"\nFocus on: {topic}. Extract and preserve details related to this topic while keeping existing memory intact."

        system_context = (
            f"\n\n## System Prompt (already provided every conversation — do NOT duplicate this)\n{system_prompt}"
            if system_prompt else ""
        )

        prompt = f"""Consolidate the conversation below into two memory documents.

Return ONLY a JSON object with two keys:
- "global": updated global memory (shared facts, project info, general context — applies to all users)
- "user": updated per-user memory (this user's preferences, personal details, user-specific context)

Rules:
- Include existing memory facts plus any new ones from the conversation.
- If nothing new is worth remembering for a section, return that section unchanged.
- Do NOT add facts to User Memory that are already in Global Memory.
- PRESERVE the original language of each fact — do NOT translate.
- Do NOT store tool names, skill names, bot capabilities, or system features.
- Do NOT duplicate anything already in the System Prompt section below.
- Return ONLY valid JSON, no explanation, no code fences.{topic_instruction}{system_context}

## Current Global Memory (IDENTITY.md)
{current_global or "(empty)"}

## Current User Memory (MEMORY.md)
{current_user or "(empty)"}

## Conversation to Process
{conversation}"""

        try:
            response = await provider.chat(
                messages=[
                    {"role": "system", "content": 'You are a memory consolidation agent. Return ONLY a JSON object with "global" and "user" keys, each containing markdown text. No explanation, no code fences. CRITICAL: Keep each fact in its original language — never translate.'},
                    {"role": "user", "content": prompt},
                ],
                model=model,
            )

            raw = (response.content or "").strip()
            if not raw:
                logger.warning("Memory consolidation: LLM returned empty response")
                return False

            result = self._parse_json_response(raw)
            if result is None:
                logger.error("Memory consolidation: failed to parse JSON: {}", raw[:200])
                return False

            global_update = _strip_memorize_tags(result.get("global", "")).strip()
            user_update = _strip_memorize_tags(result.get("user", "")).strip()

            if global_update and global_update != current_global.strip():
                self.write_global_memory(global_update)
                logger.info("Global memory (IDENTITY.md) updated")

            if user_update and user_update != current_user.strip():
                self.write_user_memory(sender_id, user_update)
                logger.info("User memory updated for {}", sender_id)

            logger.info("Memory consolidation done for user {}", sender_id)
            return True
        except Exception:
            logger.exception("Memory consolidation failed")
            return False

    # ── Cleanup ──────────────────────────────────────────────────

    async def cleanup(
        self,
        provider: LLMProvider,
        model: str,
        sender_id: str,
        system_prompt: str = "",
    ) -> bool:
        """Compact and deduplicate both memory files.

        Returns True on success, False on failure.
        """
        current_global = self.read_global_memory()
        current_user = self.read_user_memory(sender_id)

        if not current_global and not current_user:
            return True

        logger.info("Memory cleanup for user {}", sender_id)

        system_context = (
            f"\n\n## System Prompt (do NOT keep anything already covered here)\n{system_prompt}"
            if system_prompt else ""
        )

        prompt = f"""Compact and deduplicate the memory files below.

Return ONLY a JSON object with two keys:
- "global": compacted global memory
- "user": compacted per-user memory

Rules:
- Remove entries from User Memory that duplicate info already in Global Memory.
- Remove entries that duplicate info in the System Prompt section below.
- Merge related facts into concise statements.
- Remove outdated or contradicted entries.
- Preserve all unique, valuable facts.
- PRESERVE the original language of each fact — do NOT translate.
- If a section becomes empty after cleanup, return "(empty)" for that key.
- Return ONLY valid JSON, no explanation, no code fences.{system_context}

## Current Global Memory (IDENTITY.md)
{current_global or "(empty)"}

## Current User Memory (MEMORY.md)
{current_user or "(empty)"}"""

        try:
            response = await provider.chat(
                messages=[
                    {"role": "system", "content": 'You are a memory cleanup agent. Return ONLY a JSON object with "global" and "user" keys, each containing compacted markdown text. No explanation, no code fences. CRITICAL: Keep each fact in its original language — never translate.'},
                    {"role": "user", "content": prompt},
                ],
                model=model,
            )

            raw = (response.content or "").strip()
            if not raw:
                logger.warning("Memory cleanup: LLM returned empty response")
                return False

            result = self._parse_json_response(raw)
            if result is None:
                logger.error("Memory cleanup: failed to parse JSON: {}", raw[:200])
                return False

            global_update = _strip_memorize_tags(result.get("global", "")).strip()
            user_update = _strip_memorize_tags(result.get("user", "")).strip()

            # Treat "(empty)" as blank
            if global_update.lower() == "(empty)":
                global_update = ""
            if user_update.lower() == "(empty)":
                user_update = ""

            if global_update != current_global.strip():
                self.write_global_memory(global_update)
                logger.info("Global memory (IDENTITY.md) cleaned up")

            if user_update != current_user.strip():
                self.write_user_memory(sender_id, user_update)
                logger.info("User memory cleaned up for {}", sender_id)

            logger.info("Memory cleanup done for user {}", sender_id)
            return True
        except Exception:
            logger.exception("Memory cleanup failed")
            return False
