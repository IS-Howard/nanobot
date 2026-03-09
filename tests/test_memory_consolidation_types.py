"""Test MemoryStore.consolidate() handles non-string tool call arguments.

Regression test: when memory consolidation receives dict values instead of strings
from the LLM tool call response, it should serialize them to JSON instead of raising TypeError.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.providers.base import LLMResponse, ToolCallRequest


def _make_session(message_count: int = 10):
    """Create a mock session with messages."""
    session = MagicMock()
    session.messages = [
        {"role": "user", "content": f"msg{i}", "timestamp": "2026-01-01 00:00"}
        for i in range(message_count)
    ]
    return session


def _make_tool_response(memory_update):
    """Create an LLMResponse with a save_memory tool call."""
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(
                id="call_1",
                name="save_memory",
                arguments={"memory_update": memory_update},
            )
        ],
    )


class TestMemoryConsolidationTypeHandling:
    """Test that consolidation handles various argument types correctly."""

    @pytest.mark.asyncio
    async def test_string_arguments_work(self, tmp_path: Path) -> None:
        """Normal case: LLM returns string arguments."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=_make_tool_response(
                memory_update="# Memory\nUser likes testing.",
            )
        )
        session = _make_session(message_count=5)

        result = await store.consolidate(session, provider, "test-model", sender_id="user1")

        assert result is True
        memory_path = store._user_memory_path("user1")
        assert memory_path.exists()
        assert "User likes testing." in memory_path.read_text()

    @pytest.mark.asyncio
    async def test_dict_arguments_serialized_to_json(self, tmp_path: Path) -> None:
        """LLM returns dict instead of string — must not raise TypeError."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=_make_tool_response(
                memory_update={"facts": ["User likes testing"], "topics": ["testing"]},
            )
        )
        session = _make_session(message_count=5)

        result = await store.consolidate(session, provider, "test-model", sender_id="user1")

        assert result is True
        memory_content = store._user_memory_path("user1").read_text()
        parsed_mem = json.loads(memory_content)
        assert "User likes testing" in parsed_mem["facts"]

    @pytest.mark.asyncio
    async def test_string_arguments_as_raw_json(self, tmp_path: Path) -> None:
        """Some providers return arguments as a JSON string instead of parsed dict."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()

        response = LLMResponse(
            content=None,
            tool_calls=[
                ToolCallRequest(
                    id="call_1",
                    name="save_memory",
                    arguments=json.dumps({
                        "memory_update": "# Memory\nUser likes testing.",
                    }),
                )
            ],
        )
        provider.chat = AsyncMock(return_value=response)
        session = _make_session(message_count=5)

        result = await store.consolidate(session, provider, "test-model", sender_id="user1")

        assert result is True
        assert "User likes testing." in store._user_memory_path("user1").read_text()

    @pytest.mark.asyncio
    async def test_no_tool_call_returns_false(self, tmp_path: Path) -> None:
        """When LLM doesn't use the save_memory tool, return False."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=LLMResponse(content="I summarized the conversation.", tool_calls=[])
        )
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
        provider = AsyncMock()

        # Consolidate for user1
        provider.chat = AsyncMock(
            return_value=_make_tool_response(memory_update="User1 facts.")
        )
        session1 = _make_session(message_count=3)
        await store.consolidate(session1, provider, "test-model", sender_id="user1")

        # Consolidate for user2
        provider.chat = AsyncMock(
            return_value=_make_tool_response(memory_update="User2 facts.")
        )
        session2 = _make_session(message_count=3)
        await store.consolidate(session2, provider, "test-model", sender_id="user2")

        assert "User1 facts." in store.read_user_memory("user1")
        assert "User2 facts." in store.read_user_memory("user2")
        assert "User2" not in store.read_user_memory("user1")

    @pytest.mark.asyncio
    async def test_topic_passed_to_prompt(self, tmp_path: Path) -> None:
        """When topic is provided, it should appear in the LLM prompt."""
        store = MemoryStore(tmp_path)
        provider = AsyncMock()
        provider.chat = AsyncMock(
            return_value=_make_tool_response(memory_update="API design notes.")
        )
        session = _make_session(message_count=5)

        await store.consolidate(
            session, provider, "test-model", sender_id="user1", topic="API design"
        )

        # Verify the topic was included in the prompt sent to the LLM
        call_args = provider.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages") or call_args[0][0]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "Focus the consolidation on: API design" in user_msg["content"]
