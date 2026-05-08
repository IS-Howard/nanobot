"""Shell execution tool."""

import asyncio
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool

_UV_REWRITES = {
    "python": "uv run python",
    "python3": "uv run python",
    "py": "uv run python",
    "pip": "uv pip",
    "pip3": "uv pip",
}


class ExecTool(Tool):
    """Tool to execute shell commands."""

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
        path_append: str = "",
        python_via_uv: bool = True,
        restrict_resolver: Callable[[], bool] | None = None,
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",          # rm -r, rm -rf, rm -fr
            r"\bdel\s+/[fq]\b",              # del /f, del /q
            r"\brmdir\s+/s\b",               # rmdir /s
            r"(?:^|[;&|]\s*)format\b",       # format (as standalone command only)
            r"\b(mkfs|diskpart)\b",          # disk operations
            r"\bdd\s+if=",                   # dd
            r">\s*/dev/sd",                  # write to disk
            r"\b(shutdown|reboot|poweroff)\b",  # system power
            r":\(\)\s*\{.*\};\s*:",          # fork bomb
        ]
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace
        self._restrict_resolver = restrict_resolver
        self.path_append = path_append
        self.python_via_uv = python_via_uv
        self._powershell: str | None = None
        if os.name == "nt":
            self._powershell = shutil.which("pwsh") or shutil.which("powershell")

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        platform_note = (
            " Runs via PowerShell on Windows; use PowerShell syntax (`;` to chain, `$env:VAR`, backtick continuation)."
            if os.name == "nt"
            else ""
        )
        uv_note = (
            " Bare `python`/`python3`/`pip`/`pip3` are auto-rewritten to `uv run python`/`uv pip`."
            if self.python_via_uv
            else ""
        )
        return (
            "Execute a shell command and return its output. Use with caution."
            + platform_note
            + uv_note
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command"
                },
                "timeout": {
                    "type": "integer",
                    "description": f"Optional timeout in seconds (default {self.timeout}). Raise it for long installs/builds.",
                }
            },
            "required": ["command"]
        }

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error

        if self.python_via_uv:
            command = self._rewrite_for_uv(command)

        env = os.environ.copy()
        if self.path_append:
            env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append
        # Force UTF-8 so Python child output isn't mangled on Windows codepages.
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")

        effective_timeout = timeout if timeout and timeout > 0 else self.timeout

        try:
            process = await self._spawn(command, cwd, env)

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                # Wait for the process to fully terminate so pipes are
                # drained and file descriptors are released.
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                return f"Error: Command timed out after {effective_timeout} seconds"
            
            output_parts = []
            
            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))
            
            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")
            
            if process.returncode != 0:
                output_parts.append(f"\nExit code: {process.returncode}")
            
            result = "\n".join(output_parts) if output_parts else "(no output)"
            
            # Truncate very long output
            max_len = 10000
            if len(result) > max_len:
                result = result[:max_len] + f"\n... (truncated, {len(result) - max_len} more chars)"
            
            return result
            
        except FileNotFoundError as e:
            return f"Error executing command: {e}. Hint: ensure the binary is on PATH."
        except Exception as e:
            return f"Error executing command: {type(e).__name__}: {e}"

    async def _spawn(
        self, command: str, cwd: str, env: dict[str, str]
    ) -> asyncio.subprocess.Process:
        """Spawn the child process. On Windows, route through PowerShell."""
        if os.name == "nt" and self._powershell:
            # Force UTF-8 on the console so captured stdout/stderr decode cleanly,
            # then run the user's command. The wrapper is a single -Command string
            # so PowerShell's own parser handles `;`, `&&`, pipelines, here-strings, etc.
            wrapper = (
                "$ErrorActionPreference='Continue';"
                "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
                "$OutputEncoding=[System.Text.Encoding]::UTF8;"
                f"{command}"
            )
            return await asyncio.create_subprocess_exec(
                self._powershell,
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy", "Bypass",
                "-Command", wrapper,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
        return await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

    @staticmethod
    def _rewrite_for_uv(command: str) -> str:
        """Rewrite bare `python`/`pip` invocations to use uv.

        Only rewrites the leading token of the command (and the leading token
        after `&&`, `||`, `;`, or `|`) so the agent can still invoke explicit
        binaries by path (e.g. `.venv\\Scripts\\python.exe`).
        """
        # Split on shell separators while preserving them, then rewrite each segment.
        segments = re.split(r"(\s*(?:&&|\|\||;|\|)\s*)", command)
        for i in range(0, len(segments), 2):
            segments[i] = ExecTool._rewrite_segment(segments[i])
        return "".join(segments)

    @staticmethod
    def _rewrite_segment(segment: str) -> str:
        m = re.match(r"^(\s*)([A-Za-z][A-Za-z0-9_]*)(\s|$)", segment)
        if not m:
            return segment
        leading, first, trailing = m.group(1), m.group(2), m.group(3)
        replacement = _UV_REWRITES.get(first.lower())
        if not replacement:
            return segment
        rest = segment[m.end():]
        return f"{leading}{replacement}{trailing}{rest}"

    def _effective_restrict(self) -> bool:
        if self._restrict_resolver is not None:
            return bool(self._restrict_resolver())
        return self.restrict_to_workspace

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Best-effort safety guard for potentially destructive commands."""
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command blocked by safety guard (not in allowlist)"

        if self._effective_restrict():
            if "..\\" in cmd or "../" in cmd:
                return "Error: Command blocked by safety guard (path traversal detected)"

            cwd_path = Path(cwd).resolve()

            for raw in self._extract_absolute_paths(cmd):
                try:
                    p = Path(raw.strip()).resolve()
                except Exception:
                    continue
                if p.is_absolute() and cwd_path not in p.parents and p != cwd_path:
                    return "Error: Command blocked by safety guard (path outside working dir)"

        return None

    @staticmethod
    def _extract_absolute_paths(command: str) -> list[str]:
        win_paths = re.findall(r"[A-Za-z]:\\[^\s\"'|><;]+", command)   # Windows: C:\...
        posix_paths = re.findall(r"(?:^|[\s|>])(/[^\s\"'>]+)", command) # POSIX: /absolute only
        return win_paths + posix_paths
