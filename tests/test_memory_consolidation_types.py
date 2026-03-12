"""Test MemoryStore consolidation, cleanup, and memorize tag extraction."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import MemoryStore, extract_memorize_tags
from nanobot.providers.base import LLMResponse


def _make_session(message_count: int = 10):
    """Create a mock session with messages."""
    session = MagicMock()
    session.messages = [
        {"role": "user", "content": f"msg{i}", "timestamp": "2026-01-01 00:00"}
        for i in range(message_count)
    ]
    return session


def _make_provider(global_text: str = "", user_text: str = "") -> AsyncMock:
    """Create a mock provider that returns JSON with global/user keys."""
    payload = json.dumps({"global": global_text, "user": user_text})
    provider = AsyncMock()
    provider.chat = AsyncMock(return_value=LLMResponse(content=payload, tool_calls=[]))
    return provider


class TestMemoryConsolidation:
    """Test consolidation writes to both global and per-user memory."""

    @pytest.mark.asyncio
    async def test_basic_consolidation_user(self, tmp_path: Path) -> None:
        """LLM response updates per-user MEMORY.md."""
        store = MemoryStore(tmp_path)
        provider = _make_provider(user_text="User likes testing.")
        session = _make_session(message_count=5)

        result = await store.consolidate(session, provider, "test-model", sender_id="user1")

        assert result is True
        assert "User likes testing." in store.read_user_memory("user1")

    @pytest.mark.asyncio
    async def test_basic_consolidation_global(self, tmp_path: Path) -> None:
        """LLM response updates global IDENTITY.md."""
        store = MemoryStore(tmp_path)
        provider = _make_provider(global_text="Project uses Python.", user_text="")
        session = _make_session(message_count=5)

        result = await store.consolidate(session, provider, "test-model", sender_id="user1")

        assert result is True
        assert "Project uses Python." in store.read_global_memory()

    @pytest.mark.asyncio
    async def test_empty_response_returns_false(self, tmp_path: Path) -> None:
        """When LLM returns empty text, return False."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(return_value=LLMResponse(content="", tool_calls=[]))
        session = _make_session(message_count=5)

        result = await store.consolidate(session, provider, "test-model", sender_id="user1")

        assert result is False

    @pytest.mark.asyncio
    async def test_invalid_json_returns_false(self, tmp_path: Path) -> None:
        """When LLM returns non-JSON, return False."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(return_value=LLMResponse(content="not json at all", tool_calls=[]))
        session = _make_session(message_count=5)

        result = await store.consolidate(session, provider, "test-model", sender_id="user1")

        assert result is False

    @pytest.mark.asyncio
    async def test_json_with_code_fences(self, tmp_path: Path) -> None:
        """LLM response wrapped in markdown code fences should be parsed correctly."""
        store = MemoryStore(tmp_path)
        payload = '```json\n' + json.dumps({"global": "", "user": "User fact."}) + '\n```'
        provider = AsyncMock()
        provider.chat = AsyncMock(return_value=LLMResponse(content=payload, tool_calls=[]))
        session = _make_session(message_count=5)

        result = await store.consolidate(session, provider, "test-model", sender_id="user1")

        assert result is True
        assert "User fact." in store.read_user_memory("user1")

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

        provider1 = _make_provider(user_text="User1 facts.")
        await store.consolidate(_make_session(3), provider1, "test-model", sender_id="user1")

        provider2 = _make_provider(user_text="User2 facts.")
        await store.consolidate(_make_session(3), provider2, "test-model", sender_id="user2")

        assert "User1 facts." in store.read_user_memory("user1")
        assert "User2 facts." in store.read_user_memory("user2")
        assert "User2" not in store.read_user_memory("user1")

    @pytest.mark.asyncio
    async def test_topic_passed_to_prompt(self, tmp_path: Path) -> None:
        """When topic is provided, it should appear in the LLM prompt."""
        store = MemoryStore(tmp_path)
        provider = _make_provider(user_text="API design notes.")
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
        """If LLM returns same content as existing, file should not be rewritten."""
        store = MemoryStore(tmp_path)
        store.write_user_memory("user1", "Existing memory.")

        provider = _make_provider(user_text="Existing memory.")
        session = _make_session(message_count=5)

        path = store._user_memory_path("user1")
        mtime_before = path.stat().st_mtime

        result = await store.consolidate(session, provider, "test-model", sender_id="user1")

        assert result is True
        assert path.stat().st_mtime == mtime_before

    @pytest.mark.asyncio
    async def test_system_prompt_passed_to_consolidation(self, tmp_path: Path) -> None:
        """system_prompt parameter should appear in the LLM prompt for deduplication."""
        store = MemoryStore(tmp_path)
        provider = _make_provider(user_text="Some fact.")
        session = _make_session(message_count=3)

        await store.consolidate(
            session, provider, "test-model", sender_id="user1",
            system_prompt="Tools: web_search, exec",
        )

        call_args = provider.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[0][0]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "Tools: web_search, exec" in user_msg["content"]


class TestMemoryCleanup:
    """Test MemoryStore.cleanup()."""

    @pytest.mark.asyncio
    async def test_cleanup_compacts_memory(self, tmp_path: Path) -> None:
        """Cleanup updates both global and user memory."""
        store = MemoryStore(tmp_path)
        store.write_global_memory("Global fact A.\nGlobal fact A duplicate.")
        store.write_user_memory("user1", "User pref A.\nUser pref A duplicate.")

        provider = AsyncMock()
        provider.chat = AsyncMock(return_value=LLMResponse(
            content=json.dumps({"global": "Global fact A.", "user": "User pref A."}),
            tool_calls=[],
        ))

        result = await store.cleanup(provider, "test-model", sender_id="user1")

        assert result is True
        assert store.read_global_memory() == "Global fact A."
        assert store.read_user_memory("user1") == "User pref A."

    @pytest.mark.asyncio
    async def test_cleanup_noop_when_empty(self, tmp_path: Path) -> None:
        """Cleanup should be a no-op when both files are empty."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()

        result = await store.cleanup(provider, "test-model", sender_id="user1")

        assert result is True
        provider.chat.assert_not_called()


class TestMemorizeTagExtraction:
    """Test extract_memorize_tags() utility."""

    def test_no_tags(self) -> None:
        text = "Hello, how are you?"
        clean, global_facts, user_facts = extract_memorize_tags(text)
        assert clean == text
        assert global_facts == []
        assert user_facts == []

    def test_global_tag(self) -> None:
        text = "The project uses Python.<memorize_global>Project uses Python</memorize_global>"
        clean, global_facts, user_facts = extract_memorize_tags(text)
        assert "Project uses Python" in global_facts
        assert "<memorize_global>" not in clean
        assert "The project uses Python." in clean

    def test_user_tag(self) -> None:
        text = "Got it!<memorize_user>User prefers dark mode</memorize_user>"
        clean, global_facts, user_facts = extract_memorize_tags(text)
        assert "User prefers dark mode" in user_facts
        assert "<memorize_user>" not in clean

    def test_multiple_tags(self) -> None:
        text = (
            "Sure!<memorize_global>Fact A</memorize_global>"
            "<memorize_global>Fact B</memorize_global>"
            "<memorize_user>User likes coffee</memorize_user>"
        )
        clean, global_facts, user_facts = extract_memorize_tags(text)
        assert global_facts == ["Fact A", "Fact B"]
        assert user_facts == ["User likes coffee"]
        assert clean == "Sure!"

    def test_multiline_tag_content(self) -> None:
        text = "Done.<memorize_user>User name: Alice\nUser timezone: UTC+8</memorize_user>"
        clean, global_facts, user_facts = extract_memorize_tags(text)
        assert len(user_facts) == 1
        assert "Alice" in user_facts[0]


class TestMemoryAppend:
    """Test direct append methods."""

    def test_append_global_to_empty(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.append_global_memory("First fact")
        assert "- First fact" in store.read_global_memory()

    def test_append_global_to_existing(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.write_global_memory("# Existing\nSome content.")
        store.append_global_memory("New fact")
        content = store.read_global_memory()
        assert "Some content." in content
        assert "- New fact" in content

    def test_append_user_to_empty(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.append_user_memory("user1", "User fact")
        assert "- User fact" in store.read_user_memory("user1")

    def test_append_user_isolated(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path)
        store.append_user_memory("user1", "Fact for user1")
        store.append_user_memory("user2", "Fact for user2")
        assert "Fact for user1" in store.read_user_memory("user1")
        assert "Fact for user2" not in store.read_user_memory("user1")
