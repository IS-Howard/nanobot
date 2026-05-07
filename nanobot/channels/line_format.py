"""Pure helpers for LINE outbound rendering: markdown sanitizer, smart chunker, sticker extractor."""

from __future__ import annotations

import re

# Sentinel placeholders use NUL bytes so they cannot collide with user content.
_FENCED_SLOT = "\x00FENCED{idx}\x00"
_INLINE_SLOT = "\x00INLINE{idx}\x00"

_FENCED_RE = re.compile(r"(^|\n)```[^\n]*\n(.*?)\n```(?=\n|\Z)", re.DOTALL)
_INLINE_RE = re.compile(r"`([^`\n]+)`")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$", re.MULTILINE)
_RULE_RE = re.compile(r"^[ \t]*[-*_]{3,}[ \t]*$", re.MULTILINE)
_QUOTE_RE = re.compile(r"^>\s?", re.MULTILINE)
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+", re.MULTILINE)
_LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")
_STRONG_AST_RE = re.compile(r"\*\*(.+?)\*\*")
_STRONG_UND_RE = re.compile(r"__(.+?)__")
_STRIKE_RE = re.compile(r"~~(.+?)~~")
_EMPH_AST_RE = re.compile(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)")
_EMPH_UND_RE = re.compile(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)")
_HTML_RE = re.compile(r"<[^>]+>")

_STICKER_RE = re.compile(r"\[sticker:([^/\s\]]+)/([^\]\s]+)\]")

_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")


def _transform_tables(text: str) -> str:
    """Flatten GFM-style markdown tables into space-separated text rows.

    Drops separator rows (``|---|---|``) entirely and converts data rows into
    ``cell  cell  cell`` (two spaces between cells, outer pipes removed).
    Lines outside a table run untouched. Blank lines or non-table lines end
    the table run.
    """
    if "|" not in text:
        return text
    out: list[str] = []
    for line in text.split("\n"):
        if not _TABLE_ROW_RE.match(line):
            out.append(line)
            continue
        inner = line.strip()[1:-1]
        cells = [c.strip() for c in inner.split("|")]
        if cells and all(_TABLE_SEP_CELL_RE.match(c) for c in cells if c):
            continue  # drop separator row
        out.append("  ".join(cells))
    return "\n".join(out)


def markdown_to_line_text(text: str) -> str:
    """Convert markdown-flavoured text into LINE-friendly plain text.

    LINE's plain text renderer doesn't honour any markdown, so we strip
    decorative markers (`**bold**`, `_italic_`, `` `code` ``, `~~strike~~`),
    drop fence lines around code blocks (indenting their content two spaces
    instead), normalize unordered list bullets to ``•``, strip leading
    ``#`` from headings, replace horizontal rules with a row of ``─``,
    and rewrite inline links ``[label](url)`` as ``label: url``.

    Code-block and inline-code content are protected up-front so markers
    inside them (e.g. ``**not bold**``) survive verbatim.
    """
    if not text:
        return text

    fenced: list[str] = []
    inline: list[str] = []

    def _stash_fenced(m: re.Match[str]) -> str:
        idx = len(fenced)
        fenced.append(m.group(2))
        return f"{m.group(1)}{_FENCED_SLOT.format(idx=idx)}"

    def _stash_inline(m: re.Match[str]) -> str:
        idx = len(inline)
        inline.append(m.group(1))
        return _INLINE_SLOT.format(idx=idx)

    text = _FENCED_RE.sub(_stash_fenced, text)
    text = _INLINE_RE.sub(_stash_inline, text)

    text = _transform_tables(text)
    text = _HEADING_RE.sub(r"\1", text)
    text = _RULE_RE.sub("─" * 20, text)
    text = _QUOTE_RE.sub("┃ ", text)
    text = _BULLET_RE.sub(r"\1• ", text)
    text = _LINK_RE.sub(r"\1: \2", text)
    text = _STRONG_AST_RE.sub(r"\1", text)
    text = _STRONG_UND_RE.sub(r"\1", text)
    text = _STRIKE_RE.sub(r"\1", text)
    text = _EMPH_AST_RE.sub(r"\1", text)
    text = _EMPH_UND_RE.sub(r"\1", text)
    text = _HTML_RE.sub("", text)

    for i, body in enumerate(inline):
        text = text.replace(_INLINE_SLOT.format(idx=i), body)
    for i, body in enumerate(fenced):
        indented = "\n".join(("  " + line) if line else line for line in body.split("\n"))
        text = text.replace(_FENCED_SLOT.format(idx=i), indented)

    return text


def smart_split_text(text: str, limit: int) -> list[str]:
    """Split *text* into chunks no longer than *limit* characters.

    Prefers paragraph (``\\n\\n``) breaks, falls back to sentence boundaries
    (``. ``/``! ``/``? `` plus their CJK equivalents), and only hard-cuts at
    the limit as a last resort. Whitespace at chunk seams is trimmed.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        cut = _find_split(text, limit)
        head = text[:cut].rstrip()
        if head:
            chunks.append(head)
        text = text[cut:].lstrip()
    return chunks


def _find_split(text: str, limit: int) -> int:
    """Return a cut position in [1, limit] preferring paragraph then sentence boundaries."""
    para = text.rfind("\n\n", 0, limit)
    if para > 0:
        return para + 2

    best = -1
    for sep in (". ", "! ", "? ", "。", "！", "？"):
        idx = text.rfind(sep, 0, limit)
        if idx > best:
            best = idx + len(sep)
    if best > 0:
        return best

    return limit


def extract_stickers(text: str) -> list[tuple[str, str | dict[str, str]]]:
    """Split *text* into ordered ``("text", str)`` and ``("sticker", dict)`` segments.

    Sticker markers ``[sticker:packageId/stickerId]`` are extracted into
    sticker payloads; the surrounding text is kept verbatim. Empty text
    segments (e.g. when the input is only a sticker marker) are dropped so
    callers don't emit blank message bubbles. Invalid markers without a
    slash are left as literal text.
    """
    if not text:
        return []

    segments: list[tuple[str, str | dict[str, str]]] = []
    last = 0
    for m in _STICKER_RE.finditer(text):
        before = text[last:m.start()]
        if before:
            segments.append(("text", before))
        segments.append(
            (
                "sticker",
                {"packageId": m.group(1), "stickerId": m.group(2)},
            )
        )
        last = m.end()

    after = text[last:]
    if after:
        segments.append(("text", after))

    if not segments:
        return [("text", text)]
    return segments
