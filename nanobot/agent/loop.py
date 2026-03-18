"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.access import AccessManager
from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore, extract_memorize_tags
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig
    from nanobot.cron.service import CronService
    from nanobot.storage.postgres import PostgresStorage


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 2000

    _NEED_TOOLS_RE = re.compile(r"<need_tools>(.*?)</need_tools>", re.DOTALL)

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        tool_model: str = "",
        tool_provider: LLMProvider | None = None,
        auto_escalate: bool = True,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_context_items: int = 20,
        reasoning_effort: str | None = None,
        brave_api_key: str | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        storage: PostgresStorage | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        parallel: bool = False,
        access: AccessManager | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.access = access
        self.bus = bus
        self.parallel = parallel
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.tool_model = tool_model
        self.tool_provider = tool_provider
        self.auto_escalate = auto_escalate
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_context_items = max_context_items
        self.reasoning_effort = reasoning_effort
        self.brave_api_key = brave_api_key
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.storage = storage
        self.tools = ToolRegistry()
        self._tool_mode: dict[str, bool] = {}  # session_key -> persistent tool mode
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=reasoning_effort,
            brave_api_key=brave_api_key,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._global_lock = asyncio.Lock()
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._busy_sessions: set[str] = set()
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        self.tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

        # File analysis tools (require PostgreSQL storage)
        if self.storage:
            from nanobot.agent.tools.files import FileAnalysisTool, FileInfoTool
            self.tools.register(FileInfoTool(storage=self.storage))
            analysis_provider = self.tool_provider or self.provider
            analysis_model = self.tool_model or self.model
            transcription = None
            try:
                import os
                groq_key = os.environ.get("GROQ_API_KEY")
                if groq_key:
                    from nanobot.providers.transcription import GroqTranscriptionProvider
                    transcription = GroqTranscriptionProvider(api_key=groq_key)
            except ImportError:
                pass
            self.tools.register(FileAnalysisTool(
                storage=self.storage,
                provider=analysis_provider,
                model=analysis_model,
                transcription=transcription,
            ))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))
        # File tools need session_key
        session_key = f"{channel}:{chat_id}"
        for name in ("get_file_info", "analyze_file"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(session_key)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        use_tools: bool = True,
        model: str | None = None,
        allowed_tools: list[str] | None = None,
        on_busy: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        active_model = model or self.model
        # Use tool_provider when switching to tool_model (may need different provider type)
        active_provider = (self.tool_provider or self.provider) if (model and model == self.tool_model and self.tool_provider) else self.provider
        tool_defs = self.tools.get_definitions(allowed=allowed_tools) if use_tools else None

        while iteration < self.max_iterations:
            iteration += 1

            # Refresh busy indicator each iteration (loading animation expires)
            if on_busy and iteration > 1:
                try:
                    await on_busy()
                except Exception:
                    pass

            msg_chars = sum(
                len(m.get("content") or "") if isinstance(m.get("content"), str)
                else sum(len(c.get("text", "")) for c in m["content"] if isinstance(c, dict))
                if isinstance(m.get("content"), list) else 0
                for m in messages
            )
            logger.info("LLM request: ~{} chars, {} messages ({})", msg_chars, len(messages), active_model)

            response = await active_provider.chat(
                messages=messages,
                tools=tool_defs,
                model=active_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )

            if response.has_tool_calls:
                if on_progress:
                    clean = self._strip_think(response.content)
                    if clean:
                        await on_progress(clean)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    # Guard: reject tool calls not in the allowed list
                    if allowed_tools is not None and tool_call.name not in allowed_tools:
                        result = f"Error: Tool '{tool_call.name}' is not permitted for this user."
                    else:
                        result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                clean = self._strip_think(response.content)
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            cmd = msg.content.strip().lower()
            if cmd == "/stop":
                await self._handle_stop(msg)
                continue

            # /new cancels active tasks before clearing session
            if cmd == "/new":
                await self._handle_stop(msg, silent=True)

            # Notify user if agent is busy processing
            is_busy = (msg.session_key in self._busy_sessions
                       if self.parallel else bool(self._busy_sessions))
            if is_busy:
                logger.info("Session {} is busy, sending busy notification", msg.session_key)
                await self._send_busy_notification(msg)
            # Mark session as busy before dispatching to avoid race conditions
            self._busy_sessions.add(msg.session_key)
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(msg.session_key, []).append(task)
            task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _handle_stop(self, msg: InboundMessage, *, silent: bool = False) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        if silent:
            return
        total = cancelled + sub_cancelled
        content = f"⏹ Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    def _get_lock(self, session_key: str) -> asyncio.Lock:
        """Return the appropriate lock based on parallel mode."""
        if self.parallel:
            return self._session_locks.setdefault(session_key, asyncio.Lock())
        return self._global_lock

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under lock, coalescing queued messages from the same user."""
        async with self._get_lock(msg.session_key):
            try:
                msg = self._coalesce(msg)
                # Show busy indicator when processing starts (skip CLI and slash commands)
                if msg.channel != "cli" and not msg.content.strip().startswith("/"):
                    await self._send_busy_notification(msg)
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))
            finally:
                self._busy_sessions.discard(msg.session_key)
                if msg.channel != "cli":
                    self.bus.outbound.put_nowait(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata={"_cancel_busy": True},
                    ))

    def _coalesce(self, msg: InboundMessage) -> InboundMessage:
        """Drain the inbound queue and coalesce messages from the same user.

        Slash commands and messages from other users/sessions are re-queued.
        """
        extra: list[InboundMessage] = []
        requeue: list[InboundMessage] = []

        while not self.bus.inbound.empty():
            try:
                queued = self.bus.inbound.get_nowait()
            except asyncio.QueueEmpty:
                break

            if queued.content.strip().startswith("/"):
                requeue.append(queued)
            elif (queued.session_key == msg.session_key
                    and queued.sender_id == msg.sender_id):
                extra.append(queued)
            else:
                requeue.append(queued)

        for m in requeue:
            self.bus.inbound.put_nowait(m)

        if not extra:
            return msg

        all_msgs = [msg] + extra
        combined_content = "\n".join(m.content for m in all_msgs if m.content)
        combined_media: list[str] = []
        for m in all_msgs:
            combined_media.extend(m.media)
        combined_metadata: dict = {}
        for m in all_msgs:
            combined_metadata.update(m.metadata)

        logger.info("Coalesced {} messages from {}:{}", len(all_msgs), msg.session_key, msg.sender_id)

        return InboundMessage(
            channel=msg.channel,
            sender_id=msg.sender_id,
            chat_id=msg.chat_id,
            content=combined_content,
            timestamp=all_msgs[-1].timestamp,
            media=combined_media,
            metadata=combined_metadata,
            session_key_override=msg.session_key_override,
        )

    async def _send_busy_notification(self, msg: InboundMessage) -> None:
        """Notify the channel that the agent is busy processing."""
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="",
            metadata={"_busy": True},
        ))

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _get_history(self, key: str, session: Session) -> list[dict[str, Any]]:
        """Get history from DB if available, otherwise from session."""
        if self.storage:
            return await self.storage.get_history(key, limit=self.max_context_items)
        return session.get_history(max_messages=self.max_context_items)

    async def _save_turn_to_storage(self, key: str, session: Session, all_msgs: list[dict], skip: int) -> None:
        """Save turn to DB and/or session file."""
        self._save_turn(session, all_msgs, skip)
        self.sessions.save(session)
        if self.storage:
            new_msgs = []
            for m in all_msgs[skip:]:
                role = m.get("role")
                content = m.get("content", "")
                # Skip intermediate tool messages
                if role == "tool":
                    continue
                if role == "assistant" and m.get("tool_calls"):
                    continue
                if role == "assistant" and not content:
                    continue
                if role == "user" and isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        m = {**m, "content": parts[1]}
                    else:
                        continue
                new_msgs.append(m)
            if new_msgs:
                await self.storage.save_messages(key, new_msgs)

    def _should_use_tools(self, key: str, content: str, sender_id: str | None = None) -> tuple[bool, str]:
        """Determine if tools should be used. Returns (use_tools, cleaned_content)."""
        # Non-admins cannot manually activate tool mode (strip ! prefix so LLM sees clean text)
        if sender_id and self.access and not self.access.is_admin(sender_id):
            if content.startswith("!") and len(content) > 1:
                content = content[1:].strip()
            return False, content
        # Persistent tool mode
        if self._tool_mode.get(key, False):
            return True, content
        # One-shot tool mode with ! prefix
        if content.startswith("!") and len(content) > 1:
            return True, content[1:].strip()
        return False, content

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = await self._get_history(key, session)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
                sender_id=msg.sender_id,
            )
            final_content, _, all_msgs = await self._run_agent_loop(messages, use_tools=True)
            await self._save_turn_to_storage(key, session, all_msgs, 1 + len(history))
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            if self.storage:
                await self.storage.clear_session(key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/consolidate" or cmd.startswith("/consolidate "):
            topic = msg.content.strip()[len("/consolidate"):].strip() or None
            try:
                success = await self._consolidate_memory(session, sender_id=msg.sender_id, topic=topic)
                if success:
                    result = f"Memory consolidated{f' (focus: {topic})' if topic else ''}."
                    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                          content=result)
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Consolidation failed. Please try again.")
            except Exception:
                logger.exception("Consolidation failed for {}", session.key)
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Consolidation failed. Please try again.")
        if cmd == "/cleanup":
            try:
                success = await self._cleanup_memory(sender_id=msg.sender_id)
                if success:
                    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                          content="Memory cleaned up.")
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Cleanup failed. Please try again.")
            except Exception:
                logger.exception("Cleanup failed for {}", session.key)
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Cleanup failed. Please try again.")
        if cmd == "/tool":
            # Non-admins cannot toggle tool mode
            if self.access and not self.access.is_admin(msg.sender_id):
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Access denied. Only admins can toggle tool mode.")
            self._tool_mode[key] = not self._tool_mode.get(key, False)
            mode = "ON" if self._tool_mode[key] else "OFF"
            model_info = f" (model: {self.tool_model or self.model})" if self._tool_mode[key] else ""
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=f"Tool mode {mode}{model_info}")

        # /admin commands
        if cmd.startswith("/admin") and self.access:
            return self._handle_admin_command(msg)

        if cmd == "/help":
            help_text = (
                "nanobot commands:\n"
                "/new - Start a new conversation\n"
                "/consolidate [topic] - Save conversation to memory (optionally focused on a topic)\n"
                "/cleanup - Compact and deduplicate memory files\n"
                "/tool - Toggle tool mode\n"
                "/stop - Stop the current task\n"
                "/admin <passphrase> - Authenticate as admin\n"
                "/admin panel - Show permissions panel\n"
                "/help - Show available commands\n"
                "! prefix - One-shot tool mode"
            )
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=help_text)

        # Determine tool usage
        use_tools, content = self._should_use_tools(key, msg.content, sender_id=msg.sender_id)

        # Resolve access-control filtered tool/skill lists
        allowed_tools = self.access.get_allowed_tools(msg.sender_id) if self.access else None
        allowed_skills = self.access.get_allowed_skills(msg.sender_id) if self.access else None

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = await self._get_history(key, session)

        # Choose model based on tool mode
        if use_tools and self.tool_model:
            active_model = self.tool_model
        else:
            active_model = self.model

        # Determine prompt flags based on tool usage
        can_escalate = not use_tools and bool(self.tool_model) and self.auto_escalate
        include_skills = use_tools
        include_escalation = can_escalate

        initial_messages = self.context.build_messages(
            history=history,
            current_message=content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
            include_skills=include_skills,
            include_escalation=include_escalation,
            sender_id=msg.sender_id,
            allowed_skills=allowed_skills,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        progress_cb = on_progress or _bus_progress

        async def _refresh_busy() -> None:
            await self._send_busy_notification(msg)

        busy_cb = _refresh_busy if msg.channel != "cli" else None

        # Auto-escalation: free model detects tool need -> switch to tool model
        if can_escalate:
            # Pass 1: free model, no tools, lightweight prompt
            final_content, _, all_msgs = await self._run_agent_loop(
                initial_messages, on_progress=progress_cb, use_tools=False, model=self.model,
                on_busy=busy_cb,
            )
            # Check for escalation trigger
            if final_content:
                match = self._NEED_TOOLS_RE.search(final_content)
                if match:
                    reason = match.group(1).strip()
                    logger.info("Auto-escalating to tool model: {}", reason)
                    # Pass 2: tool model with FC + skills (filtered for normal users)
                    initial_messages = self.context.build_messages(
                        history=history,
                        current_message=content,
                        media=msg.media if msg.media else None,
                        channel=msg.channel, chat_id=msg.chat_id,
                        include_skills=True, include_escalation=False,
                        sender_id=msg.sender_id,
                        allowed_skills=allowed_skills,
                    )
                    final_content, _, all_msgs = await self._run_agent_loop(
                        initial_messages, on_progress=progress_cb,
                        use_tools=True, model=self.tool_model,
                        allowed_tools=allowed_tools,
                        on_busy=busy_cb,
                    )
        else:
            final_content, _, all_msgs = await self._run_agent_loop(
                initial_messages, on_progress=progress_cb,
                use_tools=use_tools, model=active_model,
                allowed_tools=allowed_tools if use_tools else None,
                on_busy=busy_cb,
            )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # Extract and persist any memorize tags the LLM emitted
        final_content, global_facts, user_facts = extract_memorize_tags(final_content)
        if global_facts or user_facts:
            store = MemoryStore(self.workspace)
            for fact in global_facts:
                store.append_global_memory(fact)
                logger.info("Global memory appended: {}", fact[:80])
            for fact in user_facts:
                store.append_user_memory(msg.sender_id, fact)
                logger.info("User memory appended for {}: {}", msg.sender_id, fact[:80])

        await self._save_turn_to_storage(key, session, all_msgs, 1 + len(history))

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    def _handle_admin_command(self, msg: InboundMessage) -> OutboundMessage:
        """Handle /admin slash commands."""
        assert self.access is not None
        raw = msg.content.strip()
        parts = raw.split(None, 2)  # ["/admin", subcommand?, arg?]
        sub = parts[1] if len(parts) > 1 else ""
        arg = parts[2] if len(parts) > 2 else ""

        # /admin <passphrase> — anyone can attempt authentication
        if sub and sub not in ("panel", "toggle_tool", "toggle_skill", "passphrase", "revoke"):
            ok = self.access.authenticate(msg.sender_id, sub)
            status = "Authenticated as admin." if ok else "Invalid passphrase."
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=status,
                                  metadata={"_admin_auth": ok, **(msg.metadata or {})})

        # Everything below requires admin
        if not self.access.is_admin(msg.sender_id):
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="Access denied. Authenticate first with /admin <passphrase>")

        if sub == "panel":
            all_tools = self.tools.tool_names
            all_skills = [s["name"] for s in self.context.skills.list_skills()]
            allowed_tools = self.access.get_allowed_tools("__normal__") or []
            allowed_skills = self.access.get_allowed_skills("__normal__") or []
            lines = ["**Admin Panel — Normal User Permissions**\n"]
            lines.append("**Tools:**")
            for t in all_tools:
                icon = "ON" if t in allowed_tools else "OFF"
                lines.append(f"  {icon} `{t}`")
            lines.append("\n**Skills:**")
            for s in all_skills:
                icon = "ON" if s in allowed_skills else "OFF"
                lines.append(f"  {icon} `{s}`")
            lines.append(f"\nUse `/admin toggle_tool <name>` or `/admin toggle_skill <name>` to change.")
            content = "\n".join(lines)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=content,
                                  metadata={"_admin_panel": True,
                                            "tools": all_tools,
                                            "skills": all_skills,
                                            "allowed_tools": allowed_tools,
                                            "allowed_skills": allowed_skills,
                                            **(msg.metadata or {})})

        if sub == "toggle_tool":
            if not arg:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Usage: /admin toggle_tool <name>")
            new_state = self.access.toggle_tool(arg)
            status = "ON" if new_state else "OFF"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=f"Tool `{arg}` for normal users: {status}")

        if sub == "toggle_skill":
            if not arg:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Usage: /admin toggle_skill <name>")
            new_state = self.access.toggle_skill(arg)
            status = "ON" if new_state else "OFF"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=f"Skill `{arg}` for normal users: {status}")

        if sub == "passphrase":
            if not arg:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Usage: /admin passphrase <new>")
            self.access.set_passphrase(arg)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="Admin passphrase updated.")

        if sub == "revoke":
            target = arg or msg.sender_id
            ok = self.access.revoke_admin(target)
            status = f"Admin access revoked for `{target}`." if ok else f"`{target}` is not an admin."
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=status)

        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                              content="Unknown /admin subcommand. Try: panel, toggle_tool, toggle_skill, passphrase, revoke")

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save user messages and final assistant reply into session.

        Intermediate tool round-trips (assistant with tool_calls, tool results)
        are internal to a single turn and not persisted — avoids breaking
        models without function-calling support.
        """
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")

            # Skip intermediate tool messages
            if role == "tool":
                continue
            if role == "assistant" and entry.get("tool_calls"):
                continue
            if role == "assistant" and not content:
                continue

            if role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if c.get("type") == "text" and isinstance(c.get("text"), str) and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                            continue
                        if (c.get("type") == "image_url"
                                and c.get("image_url", {}).get("url", "").startswith("data:image/")):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _consolidate_memory(self, session, sender_id: str, topic: str | None = None) -> bool:
        """Delegate to MemoryStore.consolidate(). Returns True on success."""
        system_prompt = self.context.build_system_prompt(include_memory=False, sender_id=sender_id)
        return await MemoryStore(self.workspace).consolidate(
            session, self.provider, self.model,
            sender_id=sender_id, topic=topic,
            system_prompt=system_prompt,
        )

    async def _cleanup_memory(self, sender_id: str) -> bool:
        """Compact and deduplicate memory files. Returns True on success."""
        system_prompt = self.context.build_system_prompt(include_memory=False, sender_id=sender_id)
        return await MemoryStore(self.workspace).cleanup(
            self.provider, self.model,
            sender_id=sender_id,
            system_prompt=system_prompt,
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
