"""PostgreSQL storage for chat history and files."""

from __future__ import annotations

import json
from typing import Any

import asyncpg
from loguru import logger

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    session_key VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL DEFAULT '',
    user_name VARCHAR(255) NOT NULL DEFAULT '',
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    tool_calls JSONB DEFAULT NULL,
    tool_call_id VARCHAR(255) DEFAULT NULL,
    tool_name VARCHAR(100) DEFAULT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history(session_key, created_at DESC);

CREATE TABLE IF NOT EXISTS stored_files (
    id SERIAL PRIMARY KEY,
    session_key VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL DEFAULT '',
    file_type VARCHAR(50) NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    file_data BYTEA NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    original_message_id VARCHAR(255) DEFAULT '',
    uploaded_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_files_session ON stored_files(session_key, uploaded_at DESC);
"""


class PostgresStorage:
    """Async PostgreSQL storage for chat history and file uploads."""

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None

    async def connect(self, url: str) -> None:
        self._pool = await asyncpg.create_pool(url, min_size=1, max_size=5)
        async with self._pool.acquire() as conn:
            await conn.execute(_SCHEMA)
        logger.info("PostgreSQL storage connected")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    # -- Chat history --

    async def get_history(self, session_key: str, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent messages for a session, ordered oldest-first, aligned to a user turn."""
        assert self._pool
        rows = await self._pool.fetch(
            "SELECT role, content, tool_calls, tool_call_id, tool_name "
            "FROM chat_history WHERE session_key = $1 "
            "ORDER BY created_at DESC LIMIT $2",
            session_key, limit,
        )
        rows = list(reversed(rows))  # oldest first

        # Align to first user message to avoid orphaned tool results
        for i, r in enumerate(rows):
            if r["role"] == "user":
                rows = rows[i:]
                break

        messages = []
        for r in rows:
            msg: dict[str, Any] = {"role": r["role"], "content": r["content"]}
            if r["tool_calls"]:
                msg["tool_calls"] = json.loads(r["tool_calls"]) if isinstance(r["tool_calls"], str) else r["tool_calls"]
            if r["tool_call_id"]:
                msg["tool_call_id"] = r["tool_call_id"]
            if r["tool_name"]:
                msg["name"] = r["tool_name"]
            messages.append(msg)
        return messages

    async def save_message(
        self,
        session_key: str,
        role: str,
        content: str,
        *,
        user_id: str = "",
        user_name: str = "",
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Save a single message. Returns the row id."""
        assert self._pool
        tc_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
        row = await self._pool.fetchrow(
            "INSERT INTO chat_history (session_key, user_id, user_name, role, content, "
            "tool_calls, tool_call_id, tool_name, metadata) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id",
            session_key, user_id, user_name, role, content,
            tc_json, tool_call_id, tool_name,
            json.dumps(metadata or {}, ensure_ascii=False),
        )
        return row["id"]

    async def save_messages(self, session_key: str, messages: list[dict[str, Any]]) -> None:
        """Bulk-save multiple messages (e.g. a full turn)."""
        assert self._pool
        async with self._pool.acquire() as conn:
            for m in messages:
                tc = m.get("tool_calls")
                await conn.execute(
                    "INSERT INTO chat_history (session_key, role, content, "
                    "tool_calls, tool_call_id, tool_name) "
                    "VALUES ($1, $2, $3, $4, $5, $6)",
                    session_key, m["role"], m.get("content") or "",
                    json.dumps(tc, ensure_ascii=False) if tc else None,
                    m.get("tool_call_id"), m.get("name"),
                )

    async def clear_session(self, session_key: str) -> None:
        assert self._pool
        await self._pool.execute(
            "DELETE FROM chat_history WHERE session_key = $1", session_key,
        )

    # -- File storage --

    async def save_file(
        self,
        session_key: str,
        user_id: str,
        file_type: str,
        mime_type: str,
        data: bytes,
        message_id: str = "",
    ) -> int:
        """Save a file. Returns the row id."""
        assert self._pool
        row = await self._pool.fetchrow(
            "INSERT INTO stored_files (session_key, user_id, file_type, mime_type, "
            "file_data, file_size_bytes, original_message_id) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id",
            session_key, user_id, file_type, mime_type,
            data, len(data), message_id,
        )
        return row["id"]

    async def get_latest_file(
        self, session_key: str, file_type: str | None = None,
    ) -> dict[str, Any] | None:
        """Get the most recent file for a session."""
        assert self._pool
        if file_type:
            row = await self._pool.fetchrow(
                "SELECT id, file_type, mime_type, file_data, file_size_bytes, uploaded_at "
                "FROM stored_files WHERE session_key = $1 AND file_type = $2 "
                "ORDER BY uploaded_at DESC LIMIT 1",
                session_key, file_type,
            )
        else:
            row = await self._pool.fetchrow(
                "SELECT id, file_type, mime_type, file_data, file_size_bytes, uploaded_at "
                "FROM stored_files WHERE session_key = $1 "
                "ORDER BY uploaded_at DESC LIMIT 1",
                session_key,
            )
        if not row:
            return None
        return dict(row)

    async def cleanup_files(self, session_key: str, max_files: int = 2) -> None:
        """Keep only the N most recent files per session (FIFO cleanup)."""
        assert self._pool
        await self._pool.execute(
            "DELETE FROM stored_files WHERE session_key = $1 AND id NOT IN ("
            "  SELECT id FROM stored_files WHERE session_key = $1 "
            "  ORDER BY uploaded_at DESC LIMIT $2"
            ")",
            session_key, max_files,
        )
