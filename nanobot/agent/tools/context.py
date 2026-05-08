"""Per-call execution context for tools.

Tool implementations may consult :data:`current_sender` to discover which user
triggered the active tool call. :class:`ToolRegistry` sets the value before
dispatching to a tool's ``execute`` and resets it afterwards, so the value is
async-safe via :class:`contextvars.ContextVar`.
"""

from contextvars import ContextVar

current_sender: ContextVar[str | None] = ContextVar("nanobot_current_sender", default=None)
