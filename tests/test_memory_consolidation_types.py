"""Test MemoryStore.consolidate() with plain-text LLM responses."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.providers.base import LLMResponse


def _make_session(message_count: int = 10):
    """Create a mock session with messages."""
    session = MagicMock()
    session.messages = [
        {"role": "user", "content": f"msg{i}", "timestamp": "2026-01-01 00:00"}
        for i in range(message_count)
    ]
    return session


def _make_provider(response_text: str) -> AsyncMock:
    """Create a mock provider that returns plain text."""
    provider = AsyncMock()
    provider.chat = AsyncMock(
        return_value=LLMResponse(content=response_text, tool_calls=[])
    )
    return provider


class TestMemoryConsolidation:
    """Test that consolidation works with plain-text LLM responses."""

    @pytest.mark.asyncio
    async def test_basic_consolidation(self, tmp_path: Path) -> None:
        """LLM returns updated memory as plain text."""
        store = MemoryStore(tmp_path)
        provider = _make_provider("# Memory\nUser likes testing.")
        session = _make_session(message_count=5)

        result = await store.consolidate(session, provider, "test-model", sender_id="user1")

        assert result is True
        memory_path = store._user_memory_path("user1")
        assert memory_path.exists()
        assert "User likes testing." in memory_path.read_text()

    @pytest.mark.asyncio
    async def test_empty_response_returns_false(self, tmp_path: Path) -> None:
        """When LLM returns empty text, return False."""
        store = MemoryStore(tmp_path)
        provider = _make_provider("")
        session = _make_session(message_count=5)

        result = await store.consolidate(session, provider, "test-model", sender_id="user1")

        assert result is False

    @pytest.mark.asyncio
    async def test_empty_session_is_noop(self, tmp_path: Path) -> None:
        """Consolidation should be a no-op when session has no messages."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        session = _make_session(message_count=0)

        result = await store.consolidate(session, provider, "test-model", sender_id="user1")

        assert result is True
        provider.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_per_user_isolation(self, tmp_path: Path) -> None:
        """Different users should have separate memory files."""
        store = MemoryStore(tmp_path)

        # Consolidate for user1
        provider1 = _make_provider("User1 facts.")
        session1 = _make_session(message_count=3)
        await store.consolidate(session1, provider1, "test-model", sender_id="user1")

        # Consolidate for user2
        provider2 = _make_provider("User2 facts.")
        session2 = _make_session(message_count=3)
        await store.consolidate(session2, provider2, "test-model", sender_id="user2")

        assert "User1 facts." in store.read_user_memory("user1")
        assert "User2 facts." in store.read_user_memory("user2")
        assert "User2" not in store.read_user_memory("user1")

    @pytest.mark.asyncio
    async def test_topic_passed_to_prompt(self, tmp_path: Path) -> None:
        """When topic is provided, it should appear in the LLM prompt."""
        store = MemoryStore(tmp_path)
        provider = _make_provider("API design notes.")
        session = _make_session(message_count=5)

        await store.consolidate(
            session, provider, "test-model", sender_id="user1", topic="API design"
        )

        call_args = provider.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages") or call_args[0][0]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "Focus on: API design" in user_msg["content"]

    @pytest.mark.asyncio
    async def test_unchanged_memory_not_rewritten(self, tmp_path: Path) -> None:
        """If LLM returns same content, file should not be rewritten."""
        store = MemoryStore(tmp_path)
        store.write_user_memory("user1", "Existing memory.")

        provider = _make_provider("Existing memory.")
        session = _make_session(message_count=5)

        # Get mtime before
        path = store._user_memory_path("user1")
        mtime_before = path.stat().st_mtime

        result = await store.consolidate(session, provider, "test-model", sender_id="user1")

        assert result is True
        # Content unchanged, file should not have been rewritten
        assert path.stat().st_mtime == mtime_before
