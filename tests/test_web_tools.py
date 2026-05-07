"""Unit tests for web tools (dev-browser-backed fetch + browse)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.tools.web import (
    WebBrowseTool,
    WebFetchTool,
    _DevBrowserRunner,
    _R_END,
    _R_START,
    _html_to_markdown,
    _validate_url,
    _wrap_script,
)


def test_validate_url_accepts_http_and_https():
    assert _validate_url("https://example.com")[0]
    assert _validate_url("http://example.com")[0]


def test_validate_url_rejects_other_schemes():
    ok, err = _validate_url("file:///etc/passwd")
    assert not ok and "http/https" in err

    ok, err = _validate_url("javascript:alert(1)")
    assert not ok


def test_validate_url_rejects_missing_domain():
    ok, _ = _validate_url("https://")
    assert not ok


def test_html_to_markdown_handles_links_headings_lists():
    md = _html_to_markdown(
        '<h2>Title</h2><p>Hello <a href="https://x.com">link</a></p><ul><li>one</li><li>two</li></ul>'
    )
    assert "## Title" in md
    assert "[link](https://x.com)" in md
    assert "- one" in md
    assert "- two" in md


def test_wrap_script_includes_markers_and_body():
    js = _wrap_script("    return { ok: true };")
    assert _R_START in js
    assert _R_END in js
    assert "return { ok: true };" in js
    assert "JSON.stringify" in js


@pytest.mark.asyncio
async def test_fetch_returns_error_when_dev_browser_missing(monkeypatch):
    runner = _DevBrowserRunner(binary="this-binary-does-not-exist-xyz")
    tool = WebFetchTool(runner=runner)
    out = json.loads(await tool.execute(url="https://example.com"))
    assert "error" in out
    assert "dev-browser" in out["error"]


@pytest.mark.asyncio
async def test_fetch_rejects_invalid_url():
    tool = WebFetchTool(runner=_DevBrowserRunner())
    out = json.loads(await tool.execute(url="ftp://example.com"))
    assert "error" in out
    assert "URL validation" in out["error"]


@pytest.mark.asyncio
async def test_fetch_extracts_markdown_from_runner_output():
    runner = _DevBrowserRunner()
    runner.is_available = lambda: True  # type: ignore
    runner.run = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "status": 200,
            "finalUrl": "https://example.com/",
            "title": "Example",
            "html": "<h1>Heading</h1><p>Body</p>",
            "text": "Heading\nBody",
        }
    )
    tool = WebFetchTool(runner=runner)
    out = json.loads(await tool.execute(url="https://example.com", extractMode="markdown"))
    assert out["status"] == 200
    assert out["extractor"] == "chromium"
    assert "# Example" in out["text"]
    assert "# Heading" in out["text"]


@pytest.mark.asyncio
async def test_fetch_truncates_long_content():
    runner = _DevBrowserRunner()
    runner.is_available = lambda: True  # type: ignore
    runner.run = AsyncMock(  # type: ignore[method-assign]
        return_value={"status": 200, "finalUrl": "u", "title": "", "html": "", "text": "x" * 10000}
    )
    tool = WebFetchTool(runner=runner)
    out = json.loads(await tool.execute(url="https://example.com", extractMode="text", maxChars=100))
    assert out["truncated"]
    assert out["length"] == 100


@pytest.mark.asyncio
async def test_browse_goto_builds_correct_script():
    runner = _DevBrowserRunner()
    runner.is_available = lambda: True  # type: ignore
    captured = {}

    async def fake_run(script, timeout=None):
        captured["script"] = script
        return {"status": 200, "url": "https://example.com/", "title": "Example"}

    runner.run = fake_run  # type: ignore
    tool = WebBrowseTool(runner=runner)
    await tool.execute(action="goto", page="login", url="https://example.com")

    assert "browser.getPage(\"login\")" in captured["script"]
    assert "page.goto(\"https://example.com\"" in captured["script"]


@pytest.mark.asyncio
async def test_browse_close_and_list():
    runner = _DevBrowserRunner()
    runner.is_available = lambda: True  # type: ignore
    captured = []

    async def fake_run(script, timeout=None):
        captured.append(script)
        return {"ok": True}

    runner.run = fake_run  # type: ignore
    tool = WebBrowseTool(runner=runner)
    await tool.execute(action="close", page="checkout")
    await tool.execute(action="list")

    assert "browser.closePage(\"checkout\")" in captured[0]
    assert "browser.listPages()" in captured[1]


@pytest.mark.asyncio
async def test_browse_fill_requires_selector_and_value():
    tool = WebBrowseTool(runner=_DevBrowserRunner())
    out = json.loads(await tool.execute(action="fill", page="x"))
    assert "error" in out
    assert "selector" in out["error"]


@pytest.mark.asyncio
async def test_browse_screenshot_writes_into_workspace_tmp(tmp_path: Path):
    runner = _DevBrowserRunner()
    runner.is_available = lambda: True  # type: ignore
    captured = {}

    async def fake_run(script, timeout=None):
        captured["script"] = script
        return {"savedTo": "x"}

    runner.run = fake_run  # type: ignore
    tool = WebBrowseTool(runner=runner, workspace=tmp_path)
    await tool.execute(action="screenshot", page="main", filename="foo")

    assert (tmp_path / "tmp").is_dir()
    assert "foo.png" in captured["script"]


@pytest.mark.asyncio
async def test_browse_unknown_action_returns_error():
    tool = WebBrowseTool(runner=_DevBrowserRunner())
    out = json.loads(await tool.execute(action="teleport"))
    assert "error" in out
    assert "Unknown action" in out["error"]


@pytest.mark.asyncio
async def test_browse_goto_rejects_javascript_url():
    tool = WebBrowseTool(runner=_DevBrowserRunner())
    out = json.loads(await tool.execute(action="goto", url="javascript:alert(1)"))
    assert "error" in out
    assert "URL validation" in out["error"]
