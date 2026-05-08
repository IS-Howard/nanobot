"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.access import AccessManager
from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore, extract_memorize_tags
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.context import current_sender
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebBrowseTool, WebFetchTool, WebSearchTool, _DevBrowserRunner
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
        self._pending_media: dict[str, dict[str, str]] = {}  # session_key -> {path, mime}
        self._attach_next: dict[str, bool] = {}  # session_key -> True when /a was sent
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
            access=self.access,
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

    def _is_call_restricted(self) -> bool:
        """Resolve workspace restriction for the in-flight tool call.

        Reads the active sender_id from :data:`current_sender` and consults the
        :class:`AccessManager` (if present). Falls back to the static config flag
        when no access manager is configured or no sender is in scope.
        """
        if self.access is None:
            return self.restrict_to_workspace
        return self.access.is_workspace_restricted(
            current_sender.get(), default=self.restrict_to_workspace
        )

    def _allowed_dir_for_call(self) -> Path | None:
        return self.workspace if self._is_call_restricted() else None

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(
                workspace=self.workspace,
                allowed_dir_resolver=self._allowed_dir_for_call,
            ))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            restrict_resolver=self._is_call_restricted,
            path_append=self.exec_config.path_append,
            python_via_uv=self.exec_config.python_via_uv,
        ))
        self.tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy))
        browser_runner = _DevBrowserRunner()
        self.tools.register(WebFetchTool(runner=browser_runner))
        self.tools.register(WebBrowseTool(runner=browser_runner, workspace=self.workspace))
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

    def _set_tool_context(
        self,
        channel: str,
        chat_id: str,
        message_id: str | None = None,
        sender_id: str | None = None,
    ) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if not hasattr(tool, "set_context"):
                    continue
                if name == "message":
                    tool.set_context(channel, chat_id, message_id)
                elif name == "spawn":
                    tool.set_context(channel, chat_id, sender_id)
                else:
                    tool.set_context(channel, chat_id)
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
        sender_id: str | None = None,
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
                        result = await self.tools.execute(
                            tool_call.name, tool_call.arguments, sender_id=sender_id
                        )
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
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"), msg.sender_id)
            history = await self._get_history(key, session)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
                sender_id=msg.sender_id,
            )
            final_content, _, all_msgs = await self._run_agent_loop(
                messages, use_tools=True, sender_id=msg.sender_id,
            )
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
            self._pending_media.pop(key, None)
            self._attach_next.pop(key, None)
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
                "/a - Process attached file (image/audio/text/PDF)\n"
                "/consolidate [topic] - Save conversation to memory (optionally focused on a topic)\n"
                "/cleanup - Compact and deduplicate memory files\n"
                "/tool - Toggle tool mode\n"
                "/stop - Stop the current task\n"
                "/admin <passphrase> - Authenticate as admin\n"
                "/admin panel - Show permissions panel (workspace + your + normal-user)\n"
                "/admin workspace on|off|reset - Set/clear workspace restriction (caller only)\n"
                "/admin self_toggle_tool|self_toggle_skill <name> - Restrict your own tools/skills\n"
                "/admin self_reset - Clear your tool/skill self-restrictions\n"
                "/help - Show available commands\n"
                "! prefix - One-shot tool mode"
            )
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=help_text)

        _attach_mode = False  # True when processing attached file content

        # Stash media when messages arrive (latest file only), don't send to LLM
        if msg.media:
            path = msg.media[-1]
            mime, _ = mimetypes.guess_type(path)
            self._pending_media[key] = {"path": path, "mime": mime or "application/octet-stream"}
            # If message is ONLY a file marker (no real user text), acknowledge and return
            stripped = msg.content.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                file_name = Path(path).name
                meta = dict(msg.metadata or {})
                # For audio/video: offer transcription confirmation
                if mime and (mime.startswith("audio/") or mime.startswith("video/")):
                    meta["_transcribe_confirm"] = True
                    meta["_file_name"] = file_name
                    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                          content=f"Audio received: {file_name}\nTranscribe?",
                                          metadata=meta)
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content=f"File received: {file_name}. Send /a to process.")

        # /transcribe — transcribe pending audio and save .txt
        if cmd == "/transcribe":
            pending = self._pending_media.get(key)
            if not pending:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="No pending audio file.")
            p = Path(pending["path"])
            mime = pending["mime"]
            if not (mime.startswith("audio/") or mime.startswith("video/")):
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Pending file is not audio/video.")
            txt_path = p.with_suffix(".txt")
            if txt_path.is_file():
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content=f"Already transcribed ({len(txt_path.read_text(encoding='utf-8'))} chars). Send /a to attach.")
            transcriber = self._get_transcription()
            if not transcriber:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Transcription unavailable (GROQ_API_KEY not set).")
            try:
                logger.info("/transcribe: transcribing {}...", p.name)
                transcript = await transcriber.transcribe(pending["path"])
                if transcript:
                    txt_path.write_text(transcript, encoding="utf-8")
                    logger.info("/transcribe: saved {} ({} chars)", txt_path, len(transcript))
                    return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                          content=f"Transcribed ({len(transcript)} chars). Send /a to attach.")
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Transcription returned empty.")
            except Exception as e:
                logger.error("/transcribe: failed: {}", e)
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content=f"Transcription failed: {e}")

        # /a (attach) — flag that the next message will include processed file content
        if cmd in ("/a", "/attach"):
            pending = self._pending_media.get(key)
            if not pending:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="No pending attachments.")
            self._attach_next[key] = True
            mime = pending["mime"]
            file_name = Path(pending["path"]).name
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=f"Attaching {file_name} ({mime.split('/')[0]}). Send your message now.")

        # If /a was used previously, process pending media and prepend to this message
        if self._attach_next.pop(key, False):
            pending = self._pending_media.get(key)
            if pending:
                processed_text, media_content = await self._process_attachment(pending)
                if media_content is not None:
                    # Image: build multimodal message and send to LLM directly
                    user_text = msg.content.strip() or processed_text
                    history = await self._get_history(key, session)
                    self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"), msg.sender_id)
                    runtime_ctx = ContextBuilder._build_runtime_context(msg.channel, msg.chat_id)
                    user_content = [{"type": "text", "text": runtime_ctx}] + media_content + [{"type": "text", "text": user_text}]
                    initial_messages = [
                        {"role": "system", "content": self.context.build_system_prompt(
                            sender_id=msg.sender_id,
                        )},
                        *history,
                        {"role": "user", "content": user_content},
                    ]
                    async def _bus_progress_img(text: str, *, tool_hint: bool = False) -> None:
                        meta = dict(msg.metadata or {})
                        meta["_progress"] = True
                        meta["_tool_hint"] = tool_hint
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id, content=text, metadata=meta,
                        ))
                    final_content, _, all_msgs = await self._run_agent_loop(
                        initial_messages, on_progress=_bus_progress_img, use_tools=False,
                        sender_id=msg.sender_id,
                    )
                    if final_content is None:
                        final_content = "I've completed processing but have no response to give."
                    final_content, global_facts, user_facts = extract_memorize_tags(final_content)
                    if global_facts or user_facts:
                        store = MemoryStore(self.workspace)
                        for fact in global_facts:
                            store.append_global_memory(fact)
                        for fact in user_facts:
                            store.append_user_memory(msg.sender_id, fact)
                    await self._save_turn_to_storage(key, session, all_msgs, 1 + len(history))
                    return OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id, content=final_content,
                        metadata=msg.metadata or {},
                    )
                else:
                    # Non-image: prepend processed text to user's message
                    # Wrap in XML tags so the LLM treats it as data, not instructions
                    msg = InboundMessage(
                        channel=msg.channel, sender_id=msg.sender_id, chat_id=msg.chat_id,
                        content=f"<attached_file>\n{processed_text}\n</attached_file>\n\nUser request: {msg.content}",
                        timestamp=msg.timestamp, metadata=msg.metadata,
                        session_key_override=msg.session_key_override,
                    )
                    _attach_mode = True

        # Determine tool usage (disabled for attach mode — file content is untrusted)
        use_tools, content = self._should_use_tools(key, msg.content, sender_id=msg.sender_id)
        if _attach_mode:
            use_tools = False

        # Resolve access-control filtered tool/skill lists
        allowed_tools = self.access.get_allowed_tools(msg.sender_id) if self.access else None
        allowed_skills = self.access.get_allowed_skills(msg.sender_id) if self.access else None

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"), msg.sender_id)
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
                on_busy=busy_cb, sender_id=msg.sender_id,
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
                        sender_id=msg.sender_id,
                    )
        else:
            final_content, _, all_msgs = await self._run_agent_loop(
                initial_messages, on_progress=progress_cb,
                use_tools=use_tools, model=active_model,
                allowed_tools=allowed_tools if use_tools else None,
                sender_id=msg.sender_id,
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

    async def _process_attachment(self, pending: dict[str, str]) -> tuple[str, list[dict] | None]:
        """Process a pending media attachment.

        Returns (text_content, multimodal_content).
        multimodal_content is a list of image_url dicts for images, None otherwise.
        """
        path = pending["path"]
        mime = pending["mime"]
        p = Path(path)
        file_name = p.name

        if not p.is_file():
            logger.warning("/a: file not found: {}", path)
            return f"File not found: {file_name}", None

        logger.info("/a: processing {} ({})", file_name, mime)

        # Image: base64-encode for multimodal vision
        if mime.startswith("image/"):
            b64 = base64.b64encode(p.read_bytes()).decode()
            media_content = [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]
            logger.info("/a: image ready for vision ({} bytes)", len(b64))
            return f"[Image: {file_name}] Describe and analyze this image.", media_content

        # Audio: transcribe via Groq (cached as .txt alongside the audio)
        if mime.startswith("audio/"):
            txt_path = p.with_suffix(".txt")
            if txt_path.is_file():
                transcript = txt_path.read_text(encoding="utf-8")
                logger.info("/a: using cached transcript ({} chars)", len(transcript))
                return f"[Audio: {file_name}]\nTranscript:\n{transcript}", None
            transcriber = self._get_transcription()
            if not transcriber:
                logger.warning("/a: GROQ_API_KEY not set, cannot transcribe")
                return f"[Audio: {file_name}] Transcription unavailable (GROQ_API_KEY not set).", None
            try:
                logger.info("/a: transcribing audio...")
                transcript = await transcriber.transcribe(path)
                if transcript:
                    txt_path.write_text(transcript, encoding="utf-8")
                    logger.info("/a: transcript saved to {} ({} chars)", txt_path, len(transcript))
                    return f"[Audio: {file_name}]\nTranscript:\n{transcript}", None
                logger.warning("/a: transcription returned empty")
                return f"[Audio: {file_name}] Transcription returned empty.", None
            except Exception as e:
                logger.error("/a: transcription failed: {}", e)
                return f"[Audio: {file_name}] Transcription failed: {e}", None

        # PDF: extract text via pymupdf (cached as .txt alongside the PDF)
        if mime == "application/pdf":
            txt_path = p.with_suffix(".pdf.txt")
            if txt_path.is_file():
                extracted = txt_path.read_text(encoding="utf-8")
                logger.info("/a: using cached PDF text ({} chars)", len(extracted))
                return f"[PDF: {file_name}]\n{extracted}", None
            try:
                import fitz
                doc = fitz.open(path)
                text_parts = []
                for page in doc:
                    text_parts.append(page.get_text())
                doc.close()
                extracted = "\n".join(text_parts).strip()
                if extracted:
                    txt_path.write_text(extracted, encoding="utf-8")
                    logger.info("/a: PDF text saved to {} ({} chars)", txt_path, len(extracted))
                    return f"[PDF: {file_name}]\n{extracted}", None
                return f"[PDF: {file_name}] No text extracted (may be image-based).", None
            except ImportError:
                return f"[PDF: {file_name}] PDF extraction unavailable (pymupdf not installed).", None
            except Exception as e:
                return f"[PDF: {file_name}] PDF extraction failed: {e}", None

        # Text files: read content
        text_extensions = {
            ".txt", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".md",
            ".csv", ".xml", ".html", ".css", ".sh", ".bat", ".cfg", ".ini",
            ".toml", ".log", ".sql", ".env", ".gitignore", ".dockerfile",
        }
        is_text = mime.startswith("text/") or p.suffix.lower() in text_extensions
        if is_text:
            try:
                file_content = p.read_text(encoding="utf-8", errors="replace")
                return f"[File: {file_name}]\n{file_content}", None
            except Exception as e:
                return f"[File: {file_name}] Failed to read: {e}", None

        # Other: show file info
        size = p.stat().st_size
        size_str = f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} bytes"
        return f"[File: {file_name}] ({mime}, {size_str}) — unsupported format for direct processing.", None

    def _get_transcription(self):
        """Lazily create a GroqTranscriptionProvider if API key is available."""
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            return None
        from nanobot.providers.transcription import GroqTranscriptionProvider
        return GroqTranscriptionProvider(api_key=key)

    def _handle_admin_command(self, msg: InboundMessage) -> OutboundMessage:
        """Handle /admin slash commands."""
        assert self.access is not None
        raw = msg.content.strip()
        parts = raw.split(None, 2)  # ["/admin", subcommand?, arg?]
        sub = parts[1] if len(parts) > 1 else ""
        arg = parts[2] if len(parts) > 2 else ""

        # /admin <passphrase> — anyone can attempt authentication
        admin_subcommands = (
            "panel", "toggle_tool", "toggle_skill", "passphrase", "revoke", "workspace",
            "self_toggle_tool", "self_toggle_skill", "self_reset",
        )
        if sub and sub not in admin_subcommands:
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
            self_tools = self.access.get_admin_self_tools(msg.sender_id)
            self_skills = self.access.get_admin_self_skills(msg.sender_id)
            workspace_restricted = self.access.is_workspace_restricted(
                msg.sender_id, default=self.restrict_to_workspace
            )
            workspace_override = self.access.get_workspace_override(msg.sender_id)

            lines = ["**Admin Panel**\n"]
            lines.append("**Your Workspace:** "
                         + ("RESTRICTED to workspace dir" if workspace_restricted
                            else "UNRESTRICTED (full filesystem + shell)"))
            lines.append("  Use `/admin workspace on|off`")

            lines.append("\n**Your Tools:** "
                         + ("custom allowlist" if self_tools is not None else "all enabled (no self-restriction)"))
            for t in all_tools:
                if self_tools is None:
                    icon = "ON"
                else:
                    icon = "ON" if t in self_tools else "OFF"
                lines.append(f"  {icon} `{t}`")
            lines.append("  Use `/admin self_toggle_tool <name>` (initializes allowlist on first toggle)")

            lines.append("\n**Your Skills:** "
                         + ("custom allowlist" if self_skills is not None else "all enabled (no self-restriction)"))
            for s in all_skills:
                if self_skills is None:
                    icon = "ON"
                else:
                    icon = "ON" if s in self_skills else "OFF"
                lines.append(f"  {icon} `{s}`")
            lines.append("  Use `/admin self_toggle_skill <name>` · `/admin self_reset` to clear")

            lines.append("\n**Normal User Tools:**")
            for t in all_tools:
                icon = "ON" if t in allowed_tools else "OFF"
                lines.append(f"  {icon} `{t}`")
            lines.append("\n**Normal User Skills:**")
            for s in all_skills:
                icon = "ON" if s in allowed_skills else "OFF"
                lines.append(f"  {icon} `{s}`")
            lines.append("  Use `/admin toggle_tool <name>` · `/admin toggle_skill <name>`")
            content = "\n".join(lines)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=content,
                                  metadata={"_admin_panel": True,
                                            "tools": all_tools,
                                            "skills": all_skills,
                                            "allowed_tools": allowed_tools,
                                            "allowed_skills": allowed_skills,
                                            "self_allowed_tools": self_tools,
                                            "self_allowed_skills": self_skills,
                                            "workspace_restricted": workspace_restricted,
                                            "workspace_override": workspace_override,
                                            "workspace_default": self.restrict_to_workspace,
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

        if sub == "self_toggle_tool":
            if not arg:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Usage: /admin self_toggle_tool <name>")
            if arg not in self.tools.tool_names:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content=f"Unknown tool `{arg}`. See `/admin panel`.")
            new_state = self.access.toggle_admin_self_tool(
                msg.sender_id, arg, self.tools.tool_names
            )
            status = "ON" if new_state else "OFF"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=f"Tool `{arg}` for you: {status}")

        if sub == "self_toggle_skill":
            if not arg:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content="Usage: /admin self_toggle_skill <name>")
            all_skills = [s["name"] for s in self.context.skills.list_skills()]
            if arg not in all_skills:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                      content=f"Unknown skill `{arg}`. See `/admin panel`.")
            new_state = self.access.toggle_admin_self_skill(
                msg.sender_id, arg, all_skills
            )
            status = "ON" if new_state else "OFF"
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=f"Skill `{arg}` for you: {status}")

        if sub == "self_reset":
            self.access.clear_admin_self(msg.sender_id)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="Your tool/skill self-restrictions cleared. All enabled.")

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

        if sub == "workspace":
            choice = arg.lower().strip()
            if choice in ("off", "false", "unrestrict", "0", "no"):
                self.access.set_workspace_override(msg.sender_id, False)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Workspace restriction OFF for your sessions. "
                            "You now have full filesystem and shell access. Other users remain restricted.",
                )
            if choice in ("on", "true", "restrict", "1", "yes"):
                self.access.set_workspace_override(msg.sender_id, True)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Workspace restriction ON for your sessions. "
                            "You're now scoped to the workspace dir.",
                )
            if choice in ("reset", "default", "clear"):
                self.access.set_workspace_override(msg.sender_id, None)
                default_state = "restricted" if self.restrict_to_workspace else "unrestricted"
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"Workspace override cleared. Reverting to config default ({default_state}).",
                )
            if choice in ("", "status"):
                restricted = self.access.is_workspace_restricted(
                    msg.sender_id, default=self.restrict_to_workspace
                )
                override = self.access.get_workspace_override(msg.sender_id)
                state = "restricted" if restricted else "unrestricted"
                source = (
                    "explicit override" if override is not None
                    else f"config default ({'restricted' if self.restrict_to_workspace else 'unrestricted'})"
                )
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"Your workspace state: {state} (from {source}). "
                            "Use `/admin workspace on|off|reset`.",
                )
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="Usage: /admin workspace on|off|reset|status",
            )

        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                              content="Unknown /admin subcommand. Try: panel, toggle_tool, toggle_skill, "
                                      "self_toggle_tool, self_toggle_skill, self_reset, "
                                      "passphrase, revoke, workspace")

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
