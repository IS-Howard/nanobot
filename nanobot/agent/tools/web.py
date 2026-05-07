"""Web tools: web_search (Brave), web_fetch + web_browse (Chromium via dev-browser CLI)."""

import asyncio
import html
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"

# Result markers used to extract a single JSON payload from dev-browser stdout.
_R_START = "__NB_DB_RESULT_START__"
_R_END = "__NB_DB_RESULT_END__"

_INSTALL_HINT = (
    "dev-browser CLI not found. Install it with: "
    "`npm i -g dev-browser && dev-browser install`"
)


def _strip_tags(text: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _validate_url(url: str) -> tuple[bool, str]:
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as e:
        return False, str(e)


def _html_to_markdown(html_str: str) -> str:
    text = re.sub(
        r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
        lambda m: f"[{_strip_tags(m[2])}]({m[1]})",
        html_str,
        flags=re.I,
    )
    text = re.sub(
        r"<h([1-6])[^>]*>([\s\S]*?)</h\1>",
        lambda m: f'\n{"#" * int(m[1])} {_strip_tags(m[2])}\n',
        text,
        flags=re.I,
    )
    text = re.sub(r"<li[^>]*>([\s\S]*?)</li>", lambda m: f"\n- {_strip_tags(m[1])}", text, flags=re.I)
    text = re.sub(r"</(p|div|section|article)>", "\n\n", text, flags=re.I)
    text = re.sub(r"<(br|hr)\s*/?>", "\n", text, flags=re.I)
    return _normalize(_strip_tags(text))


def _wrap_script(body: str) -> str:
    """Wrap a JS body so it logs its result with markers and surfaces errors as JSON.

    `body` is a JS function body; it MUST end with `return <value>;` (or assign nothing,
    in which case the result is `null`). The dev-browser daemon wraps the whole stdin
    script in `async () => { ... }`, so top-level await is fine but top-level return is not —
    we sidestep that by wrapping `body` in our own inner async IIFE.
    """
    return f"""
let __nb_result;
try {{
  __nb_result = await (async () => {{
{body}
  }})();
}} catch (e) {{
  __nb_result = {{ error: String((e && e.message) || e) }};
}}
console.log({json.dumps(_R_START)});
console.log(JSON.stringify(__nb_result === undefined ? null : __nb_result));
console.log({json.dumps(_R_END)});
"""


class _DevBrowserRunner:
    """Spawns the dev-browser CLI per call, sends a JS script via stdin, parses JSON result."""

    def __init__(
        self,
        binary: str | None = None,
        headless: bool = True,
        script_timeout: float = 60.0,
    ):
        self.binary = binary or "dev-browser"
        self.headless = headless
        self.script_timeout = script_timeout

    def _resolve_binary(self) -> str | None:
        """Resolve binary to a full path so Windows asyncio subprocess can find .cmd/.bat shims."""
        if os.path.isabs(self.binary) and os.path.exists(self.binary):
            return self.binary
        return shutil.which(self.binary)

    def is_available(self) -> bool:
        return self._resolve_binary() is not None

    async def run(self, script: str, timeout: float | None = None) -> dict[str, Any]:
        binary = self._resolve_binary()
        if not binary:
            return {"error": _INSTALL_HINT}

        t = timeout or self.script_timeout
        # On Windows, asyncio.create_subprocess_exec calls CreateProcessW directly,
        # which can't execute .cmd/.bat shims (the shape npm uses for global installs).
        # Route those through cmd.exe so stdin piping survives the shim.
        if os.name == "nt" and binary.lower().endswith((".cmd", ".bat")):
            args = ["cmd.exe", "/c", binary, "--timeout", str(int(t) + 5)]
        else:
            args = [binary, "--timeout", str(int(t) + 5)]
        if self.headless:
            args.append("--headless")

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return {"error": _INSTALL_HINT}
        except Exception as e:
            logger.error("dev-browser spawn failed: {}", e)
            return {"error": f"Failed to spawn dev-browser: {e}"}

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(script.encode("utf-8")), timeout=t
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"error": f"dev-browser script timed out after {t}s"}

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")

        m = re.search(re.escape(_R_START) + r"\s*(.*?)\s*" + re.escape(_R_END), stdout, re.DOTALL)
        if not m:
            err = stderr.strip() or stdout.strip() or "(no output)"
            logger.error("dev-browser no result. exit={} err={}", proc.returncode, err[:500])
            return {"error": f"dev-browser produced no result (exit {proc.returncode}): {err[:500]}"}

        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON from dev-browser: {e}"}


class WebSearchTool(Tool):
    """Search the web using Brave Search API."""

    name = "web_search"
    description = "Search the web. Returns titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {
                "type": "integer",
                "description": "Results (1-10)",
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(
        self, api_key: str | None = None, max_results: int = 5, proxy: str | None = None
    ):
        self._init_api_key = api_key
        self.max_results = max_results
        self.proxy = proxy

    @property
    def api_key(self) -> str:
        return self._init_api_key or os.environ.get("BRAVE_API_KEY", "")

    async def execute(
        self, query: str, count: int | None = None, **kwargs: Any
    ) -> str:
        if not self.api_key:
            return (
                "Error: Brave Search API key not configured. Set it in "
                "~/.nanobot/config.json under tools.web.search.apiKey "
                "(or export BRAVE_API_KEY), then restart the gateway."
            )

        try:
            n = min(max(count or self.max_results, 1), 10)
            logger.debug("WebSearch: {}", "proxy enabled" if self.proxy else "direct connection")
            async with httpx.AsyncClient(proxy=self.proxy) as client:
                r = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": n},
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": self.api_key,
                    },
                    timeout=10.0,
                )
                r.raise_for_status()

            results = r.json().get("web", {}).get("results", [])[:n]
            if not results:
                return f"No results for: {query}"

            lines = [f"Results for: {query}\n"]
            for i, item in enumerate(results, 1):
                lines.append(f"{i}. {item.get('title', '')}\n   {item.get('url', '')}")
                if desc := item.get("description"):
                    lines.append(f"   {desc}")
            return "\n".join(lines)
        except httpx.ProxyError as e:
            logger.error("WebSearch proxy error: {}", e)
            return f"Proxy error: {e}"
        except Exception as e:
            logger.error("WebSearch error: {}", e)
            return f"Error: {e}"


class WebFetchTool(Tool):
    """Fetch a URL via Chromium (dev-browser). Handles JS-rendered pages and modern TLS."""

    name = "web_fetch"
    description = (
        "Fetch a URL using a real Chromium browser. Handles JavaScript-rendered pages and "
        "TLS quirks that block plain HTTP libraries. Returns extracted text or markdown."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {
                "type": "string",
                "enum": ["text", "markdown"],
                "default": "markdown",
                "description": "Extract page as plain text or markdown",
            },
            "maxChars": {
                "type": "integer",
                "minimum": 100,
                "description": "Max characters to return",
            },
        },
        "required": ["url"],
    }

    def __init__(self, runner: _DevBrowserRunner | None = None, max_chars: int = 50000):
        self.runner = runner or _DevBrowserRunner()
        self.max_chars = max_chars

    async def execute(
        self,
        url: str,
        extractMode: str = "markdown",
        maxChars: int | None = None,
        **kwargs: Any,
    ) -> str:
        max_chars = maxChars or self.max_chars
        ok, err = _validate_url(url)
        if not ok:
            return json.dumps({"error": f"URL validation failed: {err}", "url": url}, ensure_ascii=False)

        page_name = f"_nb_fetch_{uuid.uuid4().hex[:10]}"
        body = f"""    const page = await browser.getPage({json.dumps(page_name)});
    const response = await page.goto({json.dumps(url)}, {{ waitUntil: "domcontentloaded", timeout: 30000 }});
    const status = response ? response.status() : 0;
    const finalUrl = page.url();
    const title = await page.title();
    const html = await page.content();
    const text = await page.evaluate(() => document.body ? document.body.innerText : "");
    await browser.closePage({json.dumps(page_name)});
    return {{ status, finalUrl, title, html, text }};"""

        logger.debug("WebFetch via dev-browser: {}", url)
        result = await self.runner.run(_wrap_script(body))

        if "error" in result:
            return json.dumps({"error": result["error"], "url": url}, ensure_ascii=False)

        title = result.get("title") or ""
        if extractMode == "markdown":
            content = _html_to_markdown(result.get("html") or "")
        else:
            content = result.get("text") or ""

        text = f"# {title}\n\n{content}" if title else content
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars]

        return json.dumps(
            {
                "url": url,
                "finalUrl": result.get("finalUrl") or url,
                "status": result.get("status") or 0,
                "extractor": "chromium",
                "truncated": truncated,
                "length": len(text),
                "text": text,
            },
            ensure_ascii=False,
        )


class WebBrowseTool(Tool):
    """Interactive Chromium control with persistent named pages."""

    name = "web_browse"
    description = (
        "Drive a real Chromium browser. Pages persist across calls — reuse the same `page` "
        "name to continue a session (e.g. for login flows or multi-step forms). "
        "Workflow: goto → snapshot/text → click/fill → snapshot/text → ... → close. "
        "Use descriptive page names like 'login' or 'checkout', not 'main'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "goto",
                    "snapshot",
                    "text",
                    "screenshot",
                    "click",
                    "fill",
                    "evaluate",
                    "close",
                    "list",
                ],
                "description": (
                    "goto: navigate page to url. "
                    "snapshot: return ARIA accessibility tree (good for discovering elements). "
                    "text: return body innerText. "
                    "screenshot: save PNG into workspace tmp/. "
                    "click: click by CSS selector. "
                    "fill: type `value` into input matched by CSS selector. "
                    "evaluate: run JS expression on the page, return its value. "
                    "close: close a named page. list: list open pages."
                ),
            },
            "page": {
                "type": "string",
                "description": "Page name (e.g. 'login', 'checkout'). Defaults to 'main'.",
            },
            "url": {"type": "string", "description": "For action=goto."},
            "selector": {
                "type": "string",
                "description": "CSS selector for action=click or action=fill.",
            },
            "value": {"type": "string", "description": "Text to type for action=fill."},
            "code": {
                "type": "string",
                "description": "JS expression for action=evaluate, e.g. '() => document.title'.",
            },
            "filename": {
                "type": "string",
                "description": "Filename for action=screenshot. Saved into workspace/tmp/. .png extension auto-added.",
            },
        },
        "required": ["action"],
    }

    def __init__(
        self,
        runner: _DevBrowserRunner | None = None,
        workspace: Path | None = None,
    ):
        self.runner = runner or _DevBrowserRunner()
        self.workspace = workspace
        self._screenshots_dir = (workspace / "tmp") if workspace else Path("tmp")

    async def execute(self, action: str, **kwargs: Any) -> str:
        page = (kwargs.get("page") or "main").strip() or "main"
        try:
            body = self._build_body(action, page, kwargs)
        except ValueError as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

        logger.debug("WebBrowse: action={} page={}", action, page)
        result = await self.runner.run(_wrap_script(body))
        return json.dumps(result, ensure_ascii=False)

    def _build_body(self, action: str, page: str, kw: dict[str, Any]) -> str:
        p = json.dumps(page)

        if action == "list":
            return "    return { pages: await browser.listPages() };"

        if action == "close":
            return f"    await browser.closePage({p});\n    return {{ closed: {p} }};"

        get_page = f"    const page = await browser.getPage({p});"

        if action == "goto":
            url = kw.get("url")
            if not url:
                raise ValueError("goto requires `url`")
            ok, err = _validate_url(url)
            if not ok:
                raise ValueError(f"URL validation failed: {err}")
            return f"""{get_page}
    const r = await page.goto({json.dumps(url)}, {{ waitUntil: "domcontentloaded", timeout: 30000 }});
    return {{ status: r ? r.status() : 0, url: page.url(), title: await page.title() }};"""

        if action == "snapshot":
            return f"""{get_page}
    const snap = await page.locator('body').ariaSnapshot();
    return {{ snapshot: snap, url: page.url(), title: await page.title() }};"""

        if action == "text":
            return f"""{get_page}
    const t = await page.evaluate(() => document.body ? document.body.innerText : "");
    return {{ text: (t || "").slice(0, 50000), url: page.url(), title: await page.title() }};"""

        if action == "click":
            sel = kw.get("selector")
            if not sel:
                raise ValueError("click requires `selector`")
            return f"""{get_page}
    await page.click({json.dumps(sel)}, {{ timeout: 10000 }});
    return {{ clicked: {json.dumps(sel)}, url: page.url() }};"""

        if action == "fill":
            sel, val = kw.get("selector"), kw.get("value")
            if not sel or val is None:
                raise ValueError("fill requires `selector` and `value`")
            return f"""{get_page}
    await page.fill({json.dumps(sel)}, {json.dumps(val)}, {{ timeout: 10000 }});
    return {{ filled: {json.dumps(sel)} }};"""

        if action == "evaluate":
            code = kw.get("code")
            if not code:
                raise ValueError("evaluate requires `code` (a JS expression)")
            return f"""{get_page}
    const v = await page.evaluate({code});
    return {{ value: v }};"""

        if action == "screenshot":
            raw = kw.get("filename") or f"screenshot_{uuid.uuid4().hex[:8]}"
            name = os.path.basename(raw)
            if not name.lower().endswith(".png"):
                name += ".png"
            self._screenshots_dir.mkdir(parents=True, exist_ok=True)
            full_path = (self._screenshots_dir / name).resolve()
            return f"""{get_page}
    await page.screenshot({{ path: {json.dumps(str(full_path))}, fullPage: false }});
    return {{ savedTo: {json.dumps(str(full_path))}, url: page.url() }};"""

        raise ValueError(f"Unknown action: {action}")
