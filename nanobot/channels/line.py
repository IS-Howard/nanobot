"""LINE channel implementation using Messaging API webhooks."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import mimetypes
import tempfile
import time
from typing import TYPE_CHECKING, Any

import httpx
from aiohttp import web
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.line_format import (
    extract_stickers,
    markdown_to_line_text,
    smart_split_text,
)
from nanobot.config.schema import LineConfig

if TYPE_CHECKING:
    from nanobot.agent.access import AccessManager
    from nanobot.storage.postgres import PostgresStorage

LINE_API_BASE = "https://api.line.me/v2/bot"
LINE_DATA_API = "https://api-data.line.me/v2/bot"
LINE_MAX_TEXT = 5000
LINE_MAX_MESSAGES_PER_PUSH = 5
LINE_REPLY_TOKEN_TTL = 20  # seconds; LINE tokens expire at ~30s, use 20s for safety
# Push a "Processing…" stub once the reply token is about to expire (2s before
# our cached TTL). Earlier values produce noise stubs for moderately-slow
# replies that would have arrived in time anyway.
LINE_PROCESSING_STUB_DELAY = LINE_REPLY_TOKEN_TTL - 2

# msg_type -> (mime_type, extension)
_MIME_MAP: dict[str, tuple[str, str]] = {
    "image": ("image/jpeg", ".jpg"),
    "audio": ("audio/mpeg", ".mp3"),
    "video": ("video/mp4", ".mp4"),
    "file": ("application/octet-stream", ".bin"),
}

# Extension -> (file_type, mime_type) for file-type messages
_EXT_TYPES: dict[str, tuple[str, str]] = {
    ".jpg": ("image", "image/jpeg"),
    ".jpeg": ("image", "image/jpeg"),
    ".png": ("image", "image/png"),
    ".gif": ("image", "image/gif"),
    ".mp3": ("audio", "audio/mpeg"),
    ".m4a": ("audio", "audio/mp4"),
    ".wav": ("audio", "audio/wav"),
    ".mp4": ("video", "video/mp4"),
    ".mov": ("video", "video/quicktime"),
    ".avi": ("video", "video/x-msvideo"),
}




def _flex_transcribe_confirm(file_name: str) -> dict:
    """Build Flex bubble asking user to confirm audio transcription."""
    return {
        "type": "flex", "altText": f"Transcribe {file_name}?",
        "contents": {
            "type": "bubble", "size": "kilo",
            "body": {"type": "box", "layout": "vertical", "spacing": "md", "contents": [
                {"type": "text", "text": "Audio Received", "weight": "bold", "size": "md"},
                {"type": "text", "text": file_name, "size": "sm", "color": "#666666", "wrap": True},
            ]},
            "footer": {"type": "box", "layout": "horizontal", "spacing": "md", "contents": [
                {"type": "button", "style": "primary", "color": "#06C755", "height": "sm",
                 "action": {"type": "postback", "label": "Transcribe",
                            "data": "action=transcribe", "displayText": "/transcribe"}},
                {"type": "button", "style": "secondary", "height": "sm",
                 "action": {"type": "postback", "label": "Skip",
                            "data": "action=skip_transcribe", "displayText": "skip"}},
            ]},
        },
    }


def _flex_tool_status(content: str) -> dict | None:
    """Build Flex bubble for tool mode toggle response."""
    # Parse "Tool mode ON (model: ...)" or "Tool mode OFF"
    if not content.startswith("Tool mode "):
        return None
    is_on = "ON" in content
    color = "#06C755" if is_on else "#999999"
    status = "ON" if is_on else "OFF"
    body_contents: list[dict] = [
        {"type": "text", "text": "Tool Mode", "weight": "bold", "flex": 0, "size": "md"},
        {"type": "text", "text": status, "color": color, "weight": "bold", "align": "end", "size": "md"},
    ]
    bubble: dict = {
        "type": "bubble", "size": "kilo",
        "body": {"type": "box", "layout": "horizontal", "contents": body_contents},
    }
    # Extract model info if present
    if "(model: " in content:
        model = content.split("(model: ", 1)[1].rstrip(")")
        bubble["footer"] = {"type": "box", "layout": "vertical", "contents": [
            {"type": "text", "text": model, "size": "xs", "color": "#999999"},
        ]}
    return {"type": "flex", "altText": content, "contents": bubble}


def _flex_help(content: str) -> dict | None:
    """Build Flex bubble for help command response."""
    if not content.startswith("nanobot commands:"):
        return None
    rows: list[dict] = []
    for line in content.split("\n")[1:]:
        line = line.strip()
        if not line:
            continue
        if " - " in line:
            cmd, desc = line.split(" - ", 1)
            rows.append({"type": "box", "layout": "horizontal", "spacing": "sm", "contents": [
                {"type": "text", "text": cmd.strip(), "size": "sm", "color": "#06C755", "flex": 0, "weight": "bold"},
                {"type": "text", "text": desc.strip(), "size": "sm", "color": "#666666", "wrap": True},
            ]})
    if not rows:
        return None
    return {
        "type": "flex", "altText": content,
        "contents": {
            "type": "bubble",
            "body": {"type": "box", "layout": "vertical", "spacing": "md", "contents": [
                {"type": "text", "text": "nanobot", "weight": "bold", "size": "lg"},
                {"type": "separator", "margin": "md"},
                {"type": "box", "layout": "vertical", "margin": "md", "spacing": "sm", "contents": rows},
            ]},
        },
    }


def _flex_admin_panel(
    tools: list[str],
    skills: list[str],
    allowed_tools: list[str],
    allowed_skills: list[str],
) -> dict:
    """Build Flex carousel for the admin permissions panel."""
    def _toggle_bubble(title: str, items: list[str], enabled: list[str], action_prefix: str) -> dict:
        rows: list[dict] = []
        for name in items:
            is_on = name in enabled
            color = "#06C755" if is_on else "#CCCCCC"
            label = f"{'ON' if is_on else 'OFF'} {name}"
            rows.append({
                "type": "box", "layout": "horizontal", "spacing": "sm",
                "margin": "sm",
                "action": {
                    "type": "postback",
                    "label": name[:20],
                    "data": f"action={action_prefix}&name={name}",
                    "displayText": f"/admin {action_prefix} {name}",
                },
                "contents": [
                    {"type": "text", "text": "●" if is_on else "○", "size": "sm",
                     "color": color, "flex": 0, "gravity": "center"},
                    {"type": "text", "text": name, "size": "sm", "color": "#333333", "wrap": True},
                ],
            })
        if not rows:
            rows.append({"type": "text", "text": "(none)", "size": "sm", "color": "#999999"})
        return {
            "type": "bubble", "size": "kilo",
            "body": {"type": "box", "layout": "vertical", "contents": [
                {"type": "text", "text": title, "weight": "bold", "size": "md"},
                {"type": "separator", "margin": "md"},
                {"type": "box", "layout": "vertical", "margin": "md", "contents": rows},
            ]},
        }

    bubbles = []
    # Split tools into groups of 8 per bubble
    for i in range(0, max(len(tools), 1), 8):
        chunk = tools[i:i + 8]
        label = "Tools" if i == 0 else f"Tools ({i + 1}+)"
        bubbles.append(_toggle_bubble(label, chunk, allowed_tools, "toggle_tool"))
    for i in range(0, max(len(skills), 1), 8):
        chunk = skills[i:i + 8]
        label = "Skills" if i == 0 else f"Skills ({i + 1}+)"
        bubbles.append(_toggle_bubble(label, chunk, allowed_skills, "toggle_skill"))

    return {
        "type": "flex", "altText": "Admin Panel",
        "contents": {"type": "carousel", "contents": bubbles},
    }


# ── Bitmap font & PNG renderer for Rich Menu images ────────────────
# 5x7 uppercase bitmap font — each char is 5 columns of 7-bit rows.
_FONT: dict[str, list[int]] = {
    "A": [0x7E, 0x09, 0x09, 0x09, 0x7E],
    "B": [0x7F, 0x49, 0x49, 0x49, 0x36],
    "C": [0x3E, 0x41, 0x41, 0x41, 0x22],
    "D": [0x7F, 0x41, 0x41, 0x41, 0x3E],
    "E": [0x7F, 0x49, 0x49, 0x49, 0x41],
    "F": [0x7F, 0x09, 0x09, 0x09, 0x01],
    "G": [0x3E, 0x41, 0x49, 0x49, 0x3A],
    "H": [0x7F, 0x08, 0x08, 0x08, 0x7F],
    "I": [0x41, 0x7F, 0x41, 0x00, 0x00],
    "J": [0x20, 0x40, 0x40, 0x3F, 0x00],
    "K": [0x7F, 0x08, 0x14, 0x22, 0x41],
    "L": [0x7F, 0x40, 0x40, 0x40, 0x40],
    "M": [0x7F, 0x02, 0x04, 0x02, 0x7F],
    "N": [0x7F, 0x02, 0x0C, 0x10, 0x7F],
    "O": [0x3E, 0x41, 0x41, 0x41, 0x3E],
    "P": [0x7F, 0x09, 0x09, 0x09, 0x06],
    "Q": [0x3E, 0x41, 0x51, 0x21, 0x5E],
    "R": [0x7F, 0x09, 0x19, 0x29, 0x46],
    "S": [0x26, 0x49, 0x49, 0x49, 0x32],
    "T": [0x01, 0x01, 0x7F, 0x01, 0x01],
    "U": [0x3F, 0x40, 0x40, 0x40, 0x3F],
    "V": [0x0F, 0x30, 0x40, 0x30, 0x0F],
    "W": [0x7F, 0x20, 0x10, 0x20, 0x7F],
    "X": [0x63, 0x14, 0x08, 0x14, 0x63],
    "Y": [0x03, 0x04, 0x78, 0x04, 0x03],
    "Z": [0x61, 0x51, 0x49, 0x45, 0x43],
    "a": [0x20, 0x54, 0x54, 0x54, 0x78],
    "b": [0x7F, 0x44, 0x44, 0x44, 0x38],
    "c": [0x38, 0x44, 0x44, 0x44, 0x28],
    "d": [0x38, 0x44, 0x44, 0x44, 0x7F],
    "e": [0x38, 0x54, 0x54, 0x54, 0x18],
    "f": [0x08, 0x7E, 0x09, 0x01, 0x02],
    "g": [0x08, 0x54, 0x54, 0x54, 0x3C],
    "h": [0x7F, 0x08, 0x04, 0x04, 0x78],
    "i": [0x00, 0x44, 0x7D, 0x40, 0x00],
    "j": [0x20, 0x40, 0x44, 0x3D, 0x00],
    "k": [0x7F, 0x10, 0x28, 0x44, 0x00],
    "l": [0x00, 0x41, 0x7F, 0x40, 0x00],
    "m": [0x7C, 0x04, 0x18, 0x04, 0x78],
    "n": [0x7C, 0x08, 0x04, 0x04, 0x78],
    "o": [0x38, 0x44, 0x44, 0x44, 0x38],
    "p": [0x7C, 0x14, 0x14, 0x14, 0x08],
    "q": [0x08, 0x14, 0x14, 0x14, 0x7C],
    "r": [0x7C, 0x08, 0x04, 0x04, 0x08],
    "s": [0x48, 0x54, 0x54, 0x54, 0x24],
    "t": [0x04, 0x3F, 0x44, 0x40, 0x20],
    "u": [0x3C, 0x40, 0x40, 0x20, 0x7C],
    "v": [0x1C, 0x20, 0x40, 0x20, 0x1C],
    "w": [0x3C, 0x40, 0x30, 0x40, 0x3C],
    "x": [0x44, 0x28, 0x10, 0x28, 0x44],
    "y": [0x0C, 0x50, 0x50, 0x50, 0x3C],
    "z": [0x44, 0x64, 0x54, 0x4C, 0x44],
    " ": [0x00, 0x00, 0x00, 0x00, 0x00],
    "0": [0x3E, 0x51, 0x49, 0x45, 0x3E],
    "1": [0x00, 0x42, 0x7F, 0x40, 0x00],
    "2": [0x42, 0x61, 0x51, 0x49, 0x46],
    "3": [0x22, 0x41, 0x49, 0x49, 0x36],
    "4": [0x18, 0x14, 0x12, 0x7F, 0x10],
    "5": [0x27, 0x45, 0x45, 0x45, 0x39],
    "6": [0x3E, 0x49, 0x49, 0x49, 0x32],
    "7": [0x01, 0x71, 0x09, 0x05, 0x03],
    "8": [0x36, 0x49, 0x49, 0x49, 0x36],
    "9": [0x26, 0x49, 0x49, 0x49, 0x3E],
}


def _render_rich_menu_png(
    width: int, height: int,
    rows: list[list[tuple[str, tuple[int, int, int]]]],
    *,
    row_heights: list[int] | None = None,
    tab_row: int | None = None,
    active_tab: int | None = None,
) -> bytes:
    """Render a Rich Menu PNG with labeled colored cells arranged in rows.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        rows: List of rows, each row is a list of (label, (r, g, b)) cells.
              Cells in each row are evenly divided across the width.
        row_heights: Optional list of pixel heights per row. If None, rows are equal.
        tab_row: Index of the row to render as tabs (with 3D pressed/released effect).
        active_tab: Index of the active (pressed) tab cell in the tab row.
    """
    import struct
    import zlib

    scale = 8   # each font pixel = 8x8 real pixels
    gap = 4     # px separator between cells
    bevel = 8   # 3D border thickness for tabs
    corner_r = 24  # rounded corner radius for tabs
    tab_gap = 10   # gap between tab cells (filled with strip bg)

    pixels = bytearray(width * height * 3)
    num_rows = len(rows)

    # Calculate row positions
    if row_heights and len(row_heights) == num_rows:
        y_starts = []
        acc = 0
        for h in row_heights:
            y_starts.append(acc)
            acc += h
        rh_list = list(row_heights)
    else:
        rh = height // num_rows
        y_starts = [ri * rh for ri in range(num_rows)]
        rh_list = [rh] * num_rows

    for ri, row in enumerate(rows):
        y0 = y_starts[ri]
        y1 = y0 + rh_list[ri] if ri < num_rows - 1 else height
        cell_width = width // len(row)
        is_tab = (ri == tab_row and active_tab is not None)

        # Fill tab strip background first
        if is_tab:
            strip_bg = bytes((35, 35, 40))
            for y in range(y0, y1):
                for x in range(width):
                    off = (y * width + x) * 3
                    pixels[off:off + 3] = strip_bg

        for ci, (label, (r, g, b)) in enumerate(row):
            x0 = ci * cell_width
            x1 = x0 + cell_width if ci < len(row) - 1 else width

            if is_tab:
                is_active = (ci == active_tab)
                # Inset tab cell by tab_gap/2 on each side
                tx0 = x0 + tab_gap // 2
                tx1 = x1 - tab_gap // 2
                tw = tx1 - tx0
                th = y1 - y0

                # Derive 3D bevel colors
                if is_active:
                    # Pressed: shadow top/left, highlight bottom/right
                    sr, sg, sb = max(r - 40, 0), max(g - 40, 0), max(b - 40, 0)
                    hr, hg, hb = min(r + 40, 255), min(g + 40, 255), min(b + 40, 255)
                else:
                    # Released: highlight top/left, shadow bottom/right
                    hr, hg, hb = min(r + 45, 255), min(g + 45, 255), min(b + 45, 255)
                    sr, sg, sb = max(r - 45, 0), max(g - 45, 0), max(b - 45, 0)

                for y in range(y0, y1):
                    for x in range(tx0, tx1):
                        lx = x - tx0
                        ly = y - y0

                        # Rounded top corners
                        in_corner = False
                        if lx < corner_r and ly < corner_r:
                            dx, dy = corner_r - lx, corner_r - ly
                            if dx * dx + dy * dy > corner_r * corner_r:
                                in_corner = True
                        elif lx > tw - corner_r - 1 and ly < corner_r:
                            dx, dy = lx - (tw - corner_r - 1), corner_r - ly
                            if dx * dx + dy * dy > corner_r * corner_r:
                                in_corner = True

                        if in_corner:
                            continue  # Leave as strip background

                        # 3D bevel effect
                        dt = ly             # distance from top
                        dl = lx             # distance from left
                        db = th - 1 - ly    # distance from bottom
                        dr = tw - 1 - lx    # distance from right

                        if is_active:
                            if dt < bevel:
                                color = (sr, sg, sb)
                            elif dl < bevel:
                                color = (sr, sg, sb)
                            elif db < bevel:
                                color = (hr, hg, hb)
                            elif dr < bevel:
                                color = (hr, hg, hb)
                            else:
                                color = (r, g, b)
                        else:
                            if dt < bevel:
                                color = (hr, hg, hb)
                            elif dl < bevel:
                                color = (hr, hg, hb)
                            elif db < bevel:
                                color = (sr, sg, sb)
                            elif dr < bevel:
                                color = (sr, sg, sb)
                            else:
                                color = (r, g, b)

                        off = (y * width + x) * 3
                        pixels[off:off + 3] = bytes(color)

                # Label centered within tab bounds
                lx0, lx1 = tx0, tx1
            else:
                # Standard block rendering
                for y in range(y0, y1):
                    for x in range(x0, x1):
                        # Skip top separator if previous row is the tab row
                        is_sep = (
                            (x < x0 + gap // 2 and ci > 0)
                            or (y < y0 + gap // 2 and ri > 0 and ri - 1 != tab_row)
                        )
                        if is_sep:
                            pixels[(y * width + x) * 3:(y * width + x) * 3 + 3] = b"\xff\xff\xff"
                        else:
                            pixels[(y * width + x) * 3:(y * width + x) * 3 + 3] = bytes((r, g, b))
                lx0, lx1 = x0, x1

            # Render label centered in cell (or tab)
            char_w = 5 * scale + scale
            text_w = len(label) * char_w - scale
            text_h = 7 * scale
            tx = lx0 + (lx1 - lx0 - text_w) // 2
            ty = y0 + (y1 - y0 - text_h) // 2

            for chi, ch in enumerate(label):
                glyph = _FONT.get(ch)
                if not glyph:
                    continue
                cx = tx + chi * char_w
                for col in range(5):
                    bits = glyph[col]
                    for row_bit in range(7):
                        if bits & (1 << row_bit):
                            for dy in range(scale):
                                for dx in range(scale):
                                    px = cx + col * scale + dx
                                    py = ty + row_bit * scale + dy
                                    if 0 <= px < width and 0 <= py < height:
                                        off = (py * width + px) * 3
                                        pixels[off:off + 3] = b"\xff\xff\xff"

    # Encode as PNG
    raw_rows = bytearray()
    for y in range(height):
        raw_rows.append(0)  # filter byte
        raw_rows.extend(pixels[y * width * 3:(y + 1) * width * 3])

    compressed = zlib.compress(bytes(raw_rows), 6)

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _chunk(b"IDAT", compressed)
    png += _chunk(b"IEND", b"")
    return png


class LineChannel(BaseChannel):
    """LINE channel using Messaging API with webhook receiver."""

    name = "line"

    def __init__(
        self,
        config: LineConfig,
        bus: MessageBus,
        storage: PostgresStorage | None = None,
        max_files_per_session: int = 2,
        access: AccessManager | None = None,
    ):
        super().__init__(config, bus)
        self.config: LineConfig = config
        self._runner: web.AppRunner | None = None
        self._http: httpx.AsyncClient | None = None
        self._storage = storage
        self._max_files = max_files_per_session
        self._access = access
        self._rich_menu_admin: str | None = None
        self._rich_menu_normal: str | None = None
        # reply_token cache: chat_id -> (token, received_time)
        self._reply_tokens: dict[str, tuple[str, float]] = {}
        # Per-chat message queue for 429 rate-limit fallback
        self._pending_queues: dict[str, list[dict[str, Any]]] = {}
        # Loading animation keep-alive tasks per chat
        self._loading_tasks: dict[str, asyncio.Task[None]] = {}
        # Per-chat "processing..." stub task (fires if agent stays silent past LINE_PROCESSING_STUB_DELAY)
        self._stub_tasks: dict[str, asyncio.Task[None]] = {}
        # Track latest source_type per chat ("user" | "group" | "room") for quick-reply gating
        self._chat_source_types: dict[str, str] = {}

    async def start(self) -> None:
        """Start webhook server and listen for LINE events."""
        if not self.config.channel_access_token or not self.config.channel_secret:
            logger.error("LINE channel_access_token and channel_secret are required")
            return

        self._running = True
        self._http = httpx.AsyncClient(timeout=30)

        # Probe bot info
        try:
            resp = await self._http.get(
                f"{LINE_API_BASE}/info",
                headers=self._auth_headers,
            )
            if resp.status_code == 200:
                info = resp.json()
                logger.info("LINE bot connected: {}", info.get("displayName", "unknown"))
            else:
                logger.warning("LINE bot probe failed ({}): {}", resp.status_code, resp.text)
        except Exception as e:
            logger.warning("LINE bot probe error: {}", e)

        # Start webhook server
        app = web.Application()
        app.router.add_post(self.config.webhook_path, self._handle_webhook)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.config.host, self.config.port)
        await site.start()

        logger.info(
            "LINE webhook listening on http://{}:{}{}",
            self.config.host,
            self.config.port,
            self.config.webhook_path,
        )

        # Set up Rich Menus for admin / normal users
        await self._setup_rich_menus()

        # Keep running
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop webhook server and clean up."""
        self._running = False
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        if self._http:
            await self._http.aclose()
            self._http = None

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message to LINE via reply API (free); queue remainder for user retrieval."""
        if not self._http:
            logger.warning("LINE client not running")
            return

        # Any outbound message means the agent is no longer silent — drop the stub.
        self._cancel_processing_stub(msg.chat_id)

        is_progress = msg.metadata.get("_progress", False)
        if not is_progress:
            self._cancel_loading(msg.chat_id)
        messages: list[dict[str, Any]] = []

        # Check if content should be upgraded to Flex Message
        flex_msg = None
        if msg.content and not is_progress:
            if msg.metadata.get("_transcribe_confirm"):
                flex_msg = _flex_transcribe_confirm(msg.metadata.get("_file_name", "audio"))
            elif msg.metadata.get("_admin_panel"):
                flex_msg = _flex_admin_panel(
                    msg.metadata.get("tools", []),
                    msg.metadata.get("skills", []),
                    msg.metadata.get("allowed_tools", []),
                    msg.metadata.get("allowed_skills", []),
                )
            else:
                flex_msg = _flex_tool_status(msg.content) or _flex_help(msg.content)

        # Switch Rich Menu when user authenticates as admin
        if msg.metadata.get("_admin_auth") and self._rich_menu_admin:
            asyncio.create_task(self._link_rich_menu(msg.chat_id, self._rich_menu_admin))

        if flex_msg:
            messages.append(flex_msg)
        elif msg.content:
            sanitized = markdown_to_line_text(msg.content)
            segments = (
                extract_stickers(sanitized)
                if self.config.stickers_enabled
                else [("text", sanitized)]
            )
            for kind, payload in segments:
                if kind == "sticker":
                    messages.append({"type": "sticker", **payload})
                else:
                    for chunk in smart_split_text(payload, LINE_MAX_TEXT):
                        messages.append({"type": "text", "text": chunk})

        # Media messages
        for media_url in msg.media or []:
            messages.append({
                "type": "image",
                "originalContentUrl": media_url,
                "previewImageUrl": media_url,
            })

        # Attach context-aware quick-reply buttons to the last user-visible
        # message of this batch (text or flex). Stop appears only while busy;
        # Continue when a queue is present or the reply token is near expiry.
        if self._should_attach_quick_reply(msg.chat_id):
            self._attach_quick_reply(messages, msg.chat_id, is_progress)

        if not messages:
            return

        # Try reply API first (free)
        token_entry = self._reply_tokens.pop(msg.chat_id, None)
        if token_entry and not is_progress:
            token, ts = token_entry
            if time.monotonic() - ts < LINE_REPLY_TOKEN_TTL:
                first_batch = messages[:LINE_MAX_MESSAGES_PER_PUSH]
                replied = await self._reply_messages(token, first_batch)
                if replied:
                    messages = messages[LINE_MAX_MESSAGES_PER_PUSH:]

        # Queue any remaining messages for user retrieval via "." / Continue
        if messages:
            self._queue_messages(msg.chat_id, messages)

    # ── Webhook handling ──────────────────────────────────────────────

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        """Handle incoming LINE webhook POST."""
        body = await request.read()
        signature = request.headers.get("X-Line-Signature", "")

        if not self._verify_signature(body, signature):
            logger.warning("LINE webhook: invalid signature")
            return web.Response(status=403, text="Invalid signature")

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return web.Response(status=400, text="Bad request")

        for event in data.get("events", []):
            try:
                await self._process_event(event)
            except Exception:
                logger.exception("Error processing LINE event")

        return web.Response(status=200, text="OK")

    def _verify_signature(self, body: bytes, signature: str) -> bool:
        """Verify X-Line-Signature using channel secret."""
        mac = hmac.new(
            self.config.channel_secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).digest()
        expected = base64.b64encode(mac).decode("utf-8")
        return hmac.compare_digest(signature, expected)

    async def _process_event(self, event: dict[str, Any]) -> None:
        """Process a single LINE webhook event."""
        event_type = event.get("type")

        # Handle postback events (from Quick Reply buttons)
        if event_type == "postback":
            await self._process_postback(event)
            return

        if event_type != "message":
            logger.debug("LINE event ignored: {}", event_type)
            return

        source = event.get("source", {})
        sender_id = source.get("userId", "")
        source_type = source.get("type", "")  # user, group, room

        if not sender_id:
            return

        # Determine chat_id based on source type
        if source_type == "group":
            chat_id = source.get("groupId", sender_id)
        elif source_type == "room":
            chat_id = source.get("roomId", sender_id)
        else:
            chat_id = sender_id
        self._chat_source_types[chat_id] = source_type

        message = event.get("message", {})
        msg_type = message.get("type", "")
        content = ""
        media: list[str] = []

        if msg_type == "text":
            content = message.get("text", "")
        elif msg_type == "location":
            title = message.get("title", "")
            address = message.get("address", "")
            lat = message.get("latitude", "")
            lng = message.get("longitude", "")
            parts = [p for p in [title, address] if p]
            content = f"[location: {' '.join(parts)} ({lat},{lng})]"
        elif msg_type == "sticker":
            pkg = message.get("packageId", "")
            stk = message.get("stickerId", "")
            content = f"[sticker:{pkg}/{stk}]"
        elif msg_type in ("image", "video", "audio", "file"):
            content, media = await self._handle_media(
                message, msg_type, sender_id, chat_id,
            )
        else:
            content = f"[{msg_type}]"

        if not content:
            return

        reply_token = event.get("replyToken", "")

        # "." is a dedicated flush trigger — never forward to agent
        if content.strip() == ".":
            if self._pending_queues.get(chat_id) and reply_token:
                await self._flush_queue(chat_id, reply_token)
            return

        # If there are queued messages, use the reply token to flush them
        # instead of saving it for the agent response.
        if self._pending_queues.get(chat_id) and reply_token:
            await self._flush_queue(chat_id, reply_token)
            # Token consumed — agent response will be queued too
            reply_token = ""

        # Show loading animation (fire-and-forget)
        if msg_type == "text":
            asyncio.create_task(self._show_loading(chat_id))

        if reply_token:
            self._reply_tokens[chat_id] = (reply_token, time.monotonic())
            self._schedule_processing_stub(chat_id)

        logger.debug(
            "LINE message: type={} sender={} chat={} text={}",
            msg_type, sender_id, chat_id, content[:80],
        )

        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            media=media,
            metadata={
                "line": {
                    "reply_token": reply_token,
                    "source_type": source_type,
                }
            },
        )

    async def _process_postback(self, event: dict[str, Any]) -> None:
        """Process a postback event (from Quick Reply and admin panel buttons)."""
        data = event.get("postback", {}).get("data", "")
        action_map = {
            "tool": "/tool",
            "consolidate": "/consolidate",
            "cleanup": "/cleanup",
            "new": "/new",
            "stop": "/stop",
            "attach": "/a",
            "transcribe": "/transcribe",
            "admin_panel": "/admin panel",
            "help": "/help",
        }
        params = dict(p.split("=", 1) for p in data.split("&") if "=" in p)
        action = params.get("action", "")

        # Skip transcription — no-op, just ignore
        if action == "skip_transcribe":
            return

        # Handle "continue" postback — flush queued messages
        if action == "continue":
            source = event.get("source", {})
            sender_id = source.get("userId", "")
            source_type = source.get("type", "")
            if source_type == "group":
                chat_id = source.get("groupId", sender_id)
            elif source_type == "room":
                chat_id = source.get("roomId", sender_id)
            else:
                chat_id = sender_id
            reply_token = event.get("replyToken", "")
            await self._flush_queue(chat_id, reply_token)
            return
        # Dynamic admin toggle commands
        if action in ("toggle_tool", "toggle_skill"):
            name = params.get("name", "")
            command = f"/admin {action} {name}" if name else None
        else:
            command = action_map.get(action)
        if not command:
            logger.debug("LINE postback ignored: {}", data)
            return

        source = event.get("source", {})
        sender_id = source.get("userId", "")
        source_type = source.get("type", "")
        if not sender_id:
            return

        if source_type == "group":
            chat_id = source.get("groupId", sender_id)
        elif source_type == "room":
            chat_id = source.get("roomId", sender_id)
        else:
            chat_id = sender_id
        self._chat_source_types[chat_id] = source_type

        reply_token = event.get("replyToken", "")
        if reply_token:
            self._reply_tokens[chat_id] = (reply_token, time.monotonic())

        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=command,
            metadata={"line": {"reply_token": reply_token, "source_type": source_type}},
        )

    async def _setup_rich_menus(self) -> None:
        """Create compact single-row Rich Menus for admin and normal users.

        The rich menu now hosts only rare/system actions plus an always-visible
        Stop button (the kill-switch users reach for when the bot is stuck or
        a reply is mid-flight before any quick-reply has rendered):

        - Admin:  [Stop | Consolidate | Cleanup | Admin]
        - Normal: [Stop | Consolidate | Cleanup]

        Frequent chat actions (New, Tool, Attach, Help) and the contextual
        Continue button live on quick-reply attached to outbound messages.
        """
        if not self._http:
            return
        try:
            # ── Clean up old aliases (from the previous tabbed layout) ───
            for alias_id in (
                "nanobot-chat-admin", "nanobot-chat-normal",
                "nanobot-system-admin", "nanobot-system-normal",
            ):
                await self._http.delete(
                    f"{LINE_API_BASE}/richmenu/alias/{alias_id}",
                    headers=self._auth_headers,
                )

            # ── Clean up old menus ───────────────────────────────
            resp = await self._http.get(
                f"{LINE_API_BASE}/richmenu/list",
                headers=self._auth_headers,
            )
            if resp.status_code == 200:
                old_names = {
                    "nanobot_admin", "nanobot_normal",
                    "nanobot_admin_q", "nanobot_normal_q",
                    "nanobot_chat_admin", "nanobot_chat_normal",
                    "nanobot_system_admin", "nanobot_system_normal",
                }
                for rm in resp.json().get("richmenus", []):
                    if rm.get("name", "") in old_names:
                        await self._http.delete(
                            f"{LINE_API_BASE}/richmenu/{rm['richMenuId']}",
                            headers=self._auth_headers,
                        )

            # Compact LINE rich-menu canvas (2500x843 = half-height).
            menu_h = 843

            # ── Admin: [Stop | Consolidate | Cleanup | Admin] ────
            cw4 = 625  # 2500 / 4
            admin_menu = {
                "size": {"width": 2500, "height": menu_h},
                "selected": True,
                "name": "nanobot_admin",
                "chatBarText": "Menu",
                "areas": [
                    {"bounds": {"x": 0, "y": 0, "width": cw4, "height": menu_h},
                     "action": {"type": "postback", "label": "Stop", "data": "action=stop"}},
                    {"bounds": {"x": cw4, "y": 0, "width": cw4, "height": menu_h},
                     "action": {"type": "postback", "label": "Consolidate",
                                "data": "action=consolidate"}},
                    {"bounds": {"x": cw4 * 2, "y": 0, "width": cw4, "height": menu_h},
                     "action": {"type": "postback", "label": "Cleanup", "data": "action=cleanup"}},
                    {"bounds": {"x": cw4 * 3, "y": 0, "width": cw4, "height": menu_h},
                     "action": {"type": "postback", "label": "Admin", "data": "action=admin_panel"}},
                ],
            }
            resp = await self._http.post(
                f"{LINE_API_BASE}/richmenu", headers=self._auth_headers, json=admin_menu,
            )
            if resp.status_code == 200:
                self._rich_menu_admin = resp.json().get("richMenuId")
                await self._upload_rich_menu_image(self._rich_menu_admin, [
                    [("Stop", (180, 50, 50)), ("Consolidate", (120, 90, 40)),
                     ("Cleanup", (140, 60, 60)), ("Admin", (30, 120, 70))],
                ])
                logger.info("Created admin Rich Menu: {}", self._rich_menu_admin)

            # ── Normal: [Stop | Consolidate | Cleanup] ───────────
            cw3 = 833  # 2500 / 3 (rounded down)
            normal_menu = {
                "size": {"width": 2500, "height": menu_h},
                "selected": True,
                "name": "nanobot_normal",
                "chatBarText": "Menu",
                "areas": [
                    {"bounds": {"x": 0, "y": 0, "width": cw3, "height": menu_h},
                     "action": {"type": "postback", "label": "Stop", "data": "action=stop"}},
                    {"bounds": {"x": cw3, "y": 0, "width": cw3 + 1, "height": menu_h},
                     "action": {"type": "postback", "label": "Consolidate",
                                "data": "action=consolidate"}},
                    {"bounds": {"x": cw3 * 2, "y": 0, "width": cw3 + 1, "height": menu_h},
                     "action": {"type": "postback", "label": "Cleanup", "data": "action=cleanup"}},
                ],
            }
            resp = await self._http.post(
                f"{LINE_API_BASE}/richmenu", headers=self._auth_headers, json=normal_menu,
            )
            if resp.status_code == 200:
                self._rich_menu_normal = resp.json().get("richMenuId")
                await self._upload_rich_menu_image(self._rich_menu_normal, [
                    [("Stop", (180, 50, 50)), ("Consolidate", (120, 90, 40)),
                     ("Cleanup", (140, 60, 60))],
                ])
                # Set as default for all users.
                await self._http.post(
                    f"{LINE_API_BASE}/user/all/richmenu/{self._rich_menu_normal}",
                    headers=self._auth_headers,
                )
                logger.info("Created normal Rich Menu (default): {}", self._rich_menu_normal)

            # Re-link admin menu to known admin users.
            if self._rich_menu_admin and self._access:
                admins = self._access._data.get("admins", [])
                for uid in admins:
                    await self._link_rich_menu(uid, self._rich_menu_admin)
                if admins:
                    logger.info("Re-linked admin Rich Menu for {} user(s)", len(admins))

        except Exception as e:
            logger.warning("Failed to setup Rich Menus: {}", e)

    async def _upload_rich_menu_image(
        self, menu_id: str, rows: list[list[tuple[str, tuple[int, int, int]]]],
        *,
        row_heights: list[int] | None = None,
        tab_row: int | None = None,
        active_tab: int | None = None,
    ) -> None:
        """Upload a PNG image with labeled colored cells for a Rich Menu.

        Args:
            menu_id: Rich Menu ID to upload to.
            rows: List of rows, each row is a list of (label, (r, g, b)) cells.
            row_heights: Optional pixel heights per row. If None, 843px per row.
            tab_row: Index of the row to render as tabs.
            active_tab: Index of the active (pressed) tab cell.
        """
        if not self._http or not menu_id:
            return
        try:
            if row_heights:
                img_height = sum(row_heights)
            else:
                img_height = 843 * len(rows)
            png = _render_rich_menu_png(
                2500, img_height, rows,
                row_heights=row_heights, tab_row=tab_row, active_tab=active_tab,
            )
            resp = await self._http.post(
                f"{LINE_DATA_API}/richmenu/{menu_id}/content",
                headers={
                    "Authorization": f"Bearer {self.config.channel_access_token}",
                    "Content-Type": "image/png",
                },
                content=png,
            )
            if resp.status_code != 200:
                logger.warning("Rich Menu image upload failed ({}): {}", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("Rich Menu image upload error: {}", e)

    async def _link_rich_menu(self, user_id: str, menu_id: str) -> None:
        """Link a Rich Menu to a specific user."""
        if not self._http or not menu_id:
            return
        try:
            resp = await self._http.post(
                f"{LINE_API_BASE}/user/{user_id}/richmenu/{menu_id}",
                headers=self._auth_headers,
            )
            if resp.status_code != 200:
                logger.warning("Rich Menu link failed ({}): {}", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("Rich Menu link error: {}", e)

    async def _handle_media(
        self,
        message: dict[str, Any],
        msg_type: str,
        sender_id: str,
        chat_id: str,
    ) -> tuple[str, list[str]]:
        """Download media content, save to temp file and optionally to DB."""
        message_id = message.get("id", "")
        file_name = message.get("fileName", "")
        media: list[str] = []

        # Determine MIME type and extension
        if msg_type == "file" and file_name:
            ext = "." + file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
            if ext in _EXT_TYPES:
                file_type, mime_type = _EXT_TYPES[ext]
            else:
                file_type, mime_type = "file", mimetypes.guess_type(file_name)[0] or "application/octet-stream"
            content = f"[file: {file_name}]"
        else:
            mime_type, ext = _MIME_MAP.get(msg_type, ("application/octet-stream", ".bin"))
            file_type = msg_type
            content = f"[{msg_type}]"

        # Download content from LINE
        data = await self._download_content(message_id)
        if not data:
            return content, media

        # Write to temp file for context builder
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            tmp.write(data)
            tmp.close()
            media.append(tmp.name)
        except Exception as e:
            logger.error("Failed to write temp file: {}", e)

        # Save to PostgreSQL if available
        if self._storage:
            session_key = f"line:{chat_id}"
            try:
                await self._storage.save_file(
                    session_key=session_key,
                    user_id=sender_id,
                    file_type=file_type,
                    mime_type=mime_type,
                    data=data,
                    message_id=message_id,
                )
                await self._storage.cleanup_files(session_key, self._max_files)
                logger.info("LINE {} saved to DB for session {}", file_type, session_key)
            except Exception as e:
                logger.error("Failed to save file to DB: {}", e)

        return content, media

    # ── LINE API calls ────────────────────────────────────────────────

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.channel_access_token}",
        }

    async def _download_content(self, message_id: str) -> bytes | None:
        """Download media content from LINE content API."""
        if not self._http or not message_id:
            return None
        try:
            resp = await self._http.get(
                f"{LINE_DATA_API}/message/{message_id}/content",
                headers={"Authorization": f"Bearer {self.config.channel_access_token}"},
            )
            if resp.status_code == 200:
                return resp.content
            logger.warning("LINE content download failed ({}): {}", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.error("LINE content download error: {}", e)
        return None

    async def show_busy(self, chat_id: str) -> None:
        """Show LINE loading animation and keep it alive until the next show_busy or send."""
        logger.debug("LINE show_busy for chat_id={}", chat_id)
        # Cancel any existing keep-alive loop for this chat
        old_task = self._loading_tasks.pop(chat_id, None)
        if old_task:
            old_task.cancel()
        # Start a new keep-alive loop
        self._loading_tasks[chat_id] = asyncio.create_task(self._loading_keep_alive(chat_id))

    def _cancel_loading(self, chat_id: str) -> None:
        """Cancel the loading keep-alive loop for a chat."""
        task = self._loading_tasks.pop(chat_id, None)
        if task:
            task.cancel()

    def clear_busy(self, chat_id: str) -> None:
        """Cancel the loading animation for a chat."""
        self._cancel_loading(chat_id)

    async def _loading_keep_alive(self, chat_id: str) -> None:
        """Repeatedly send loading animation every 25s until cancelled."""
        try:
            while True:
                await self._show_loading(chat_id)
                await asyncio.sleep(25)
        except asyncio.CancelledError:
            pass

    def _schedule_processing_stub(self, chat_id: str) -> None:
        """Schedule a delayed 'processing...' push so Stop becomes tappable.

        Quick-reply buttons only render under sent messages and the loading
        animation can't carry buttons. If the agent doesn't emit anything
        within LINE_PROCESSING_STUB_DELAY seconds, push a small stub via the
        push API so the user can tap Stop while still waiting. The stub is
        cancelled the moment the agent emits its first real reply.
        """
        old = self._stub_tasks.pop(chat_id, None)
        if old:
            old.cancel()
        if not self._should_attach_quick_reply(chat_id):
            return
        self._stub_tasks[chat_id] = asyncio.create_task(self._processing_stub_loop(chat_id))

    def _cancel_processing_stub(self, chat_id: str) -> None:
        """Cancel any pending processing stub task for *chat_id*."""
        task = self._stub_tasks.pop(chat_id, None)
        if task:
            task.cancel()

    async def _processing_stub_loop(self, chat_id: str) -> None:
        """Wait, then push a 'processing...' stub if the agent is still silent."""
        try:
            await asyncio.sleep(LINE_PROCESSING_STUB_DELAY)
            stub_msg: dict[str, Any] = {"type": "text", "text": "⏳ Processing…"}
            self._attach_quick_reply([stub_msg], chat_id, is_progress=True)
            await self._push_messages(chat_id, [stub_msg])
        except asyncio.CancelledError:
            pass

    async def _push_messages(self, chat_id: str, messages: list[dict[str, Any]]) -> bool:
        """Push messages to a chat (counts against monthly quota)."""
        if not self._http:
            return False
        try:
            resp = await self._http.post(
                f"{LINE_API_BASE}/message/push",
                headers=self._auth_headers,
                json={"to": chat_id, "messages": messages},
            )
            if resp.status_code == 200:
                return True
            logger.warning("LINE push failed ({}): {}", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("LINE push error: {}", e)
        return False

    async def _show_loading(self, chat_id: str) -> None:
        """Show loading animation in LINE chat (lasts 30s)."""
        if not self._http:
            return
        try:
            resp = await self._http.post(
                f"{LINE_API_BASE}/chat/loading/start",
                headers=self._auth_headers,
                json={"chatId": chat_id, "loadingSeconds": 30},
            )
            if resp.status_code != 202:
                logger.warning("LINE loading API returned {}: {}", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("LINE loading API error: {}", e)

    async def _reply_messages(self, reply_token: str, messages: list[dict[str, Any]]) -> bool:
        """Reply to a message using LINE reply API (free, no quota cost).

        Returns True if reply succeeded, False otherwise.
        """
        if not self._http:
            return False
        try:
            resp = await self._http.post(
                f"{LINE_API_BASE}/message/reply",
                headers=self._auth_headers,
                json={"replyToken": reply_token, "messages": messages},
            )
            if resp.status_code == 200:
                logger.debug("LINE reply succeeded")
                return True
            logger.warning("LINE reply failed ({}): {}", resp.status_code, resp.text[:200])
            return False
        except Exception as e:
            logger.warning("LINE reply error: {}", e)
            return False

    def _queue_messages(self, chat_id: str, messages: list[dict[str, Any]]) -> None:
        """Add messages to the pending queue and show Continue rich menu."""
        self._pending_queues.setdefault(chat_id, []).extend(messages)
        logger.info("LINE queued {} message(s) for {} (total: {})",
                     len(messages), chat_id, len(self._pending_queues[chat_id]))

    async def _flush_queue(self, chat_id: str, reply_token: str | None = None) -> bool:
        """Flush queued messages for a chat using the reply API (free).

        Returns True if there were messages to flush.
        """
        queue = self._pending_queues.get(chat_id)
        if not queue:
            return False

        if reply_token:
            send_count = min(len(queue), LINE_MAX_MESSAGES_PER_PUSH)
            batch = queue[:send_count]
            ok = await self._reply_messages(reply_token, batch)
            if ok:
                del queue[:send_count]

        if not queue:
            del self._pending_queues[chat_id]
        else:
            logger.info("LINE queue for {}: {} message(s) remaining", chat_id, len(queue))

        return True


    def _should_attach_quick_reply(self, chat_id: str) -> bool:
        """Return True iff quick-reply buttons should be attached for this chat."""
        if not self.config.quick_reply_enabled:
            return False
        source_type = self._chat_source_types.get(chat_id, "user")
        if source_type in ("group", "room") and not self.config.quick_reply_in_groups:
            return False
        return True

    def _is_busy(self, chat_id: str, is_progress: bool) -> bool:
        """Agent is processing iff this is a progress chunk or a loading task is active."""
        return is_progress or chat_id in self._loading_tasks

    def _is_queueing(self, chat_id: str, is_busy: bool) -> bool:
        """Continue is meaningful when a queue exists, or the reply token is near
        expiry while the agent is still busy (so the user must tap Continue to
        provide a fresh reply token before the queue starts filling)."""
        if self._pending_queues.get(chat_id):
            return True
        if is_busy:
            token_entry = self._reply_tokens.get(chat_id)
            if token_entry and time.monotonic() - token_entry[1] > LINE_REPLY_TOKEN_TTL - 5:
                return True
        return False

    def _build_quick_reply_items(
        self,
        chat_id: str,
        is_progress: bool,
        *,
        will_queue: bool = False,
        force_idle: bool = False,
    ) -> list[dict[str, Any]]:
        """Build a context-aware quick-reply list (max 13 items).

        ``will_queue=True`` forces the Continue button on (used when this
        send is about to overflow into the queue). ``force_idle=True`` skips
        all contextual buttons (used for messages destined for the queue
        tail, which the user only sees after a flush).
        """
        items: list[dict[str, Any]] = []
        is_busy = False
        if not force_idle:
            is_busy = self._is_busy(chat_id, is_progress)
            if is_busy:
                items.append({
                    "type": "action",
                    "action": {"type": "postback", "label": "Stop", "data": "action=stop"},
                })
            if will_queue or self._is_queueing(chat_id, is_busy):
                items.append({
                    "type": "action",
                    "action": {"type": "postback", "label": "Continue",
                               "data": "action=continue"},
                })
        # Static nav (New / Tool / Help) only when the agent is idle — tapping
        # them mid-processing would interrupt the in-flight reply.
        if not is_busy:
            for cfg in self.config.quick_reply_actions:
                if len(items) >= 13:
                    break
                action: dict[str, Any] = {"type": cfg.type, "label": cfg.label}
                if cfg.type == "postback":
                    action["data"] = cfg.data
                    if cfg.display_text:
                        action["displayText"] = cfg.display_text
                elif cfg.type == "message":
                    action["text"] = cfg.text or cfg.label
                items.append({"type": "action", "action": action})
        return items

    def _attach_quick_reply(
        self, messages: list[dict[str, Any]], chat_id: str, is_progress: bool
    ) -> None:
        """Attach quick-reply blocks at the user-visible boundaries.

        LINE quick-reply only renders under the message it's attached to,
        so for overflow batches we attach buttons twice:

        1. To the last text/flex message of the immediately-sent batch (first 5),
           with current state plus a forced Continue if overflow is imminent.
           This is what the user actually sees right after sending.
        2. To the last text/flex message of the queued tail, with idle nav
           buttons only — by flush time, the user has already chosen to
           continue and the queue is being drained.
        """
        if not messages:
            return
        batch_size = LINE_MAX_MESSAGES_PER_PUSH
        will_overflow = len(messages) > batch_size

        items_now = self._build_quick_reply_items(
            chat_id, is_progress, will_queue=will_overflow,
        )
        if items_now:
            for m in reversed(messages[:batch_size]):
                if m.get("type") in ("text", "flex"):
                    m["quickReply"] = {"items": items_now}
                    break

        if will_overflow:
            items_later = self._build_quick_reply_items(
                chat_id, is_progress=False, force_idle=True,
            )
            if items_later:
                for m in reversed(messages[batch_size:]):
                    if m.get("type") in ("text", "flex"):
                        m["quickReply"] = {"items": items_later}
                        break

    def _is_admin(self, chat_id: str) -> bool:
        """Check if a user is admin via access manager."""
        if not self._access:
            return False
        return chat_id in self._access._data.get("admins", [])


def _split_text(text: str, limit: int = LINE_MAX_TEXT) -> list[str]:
    """Backward-compatible shim — delegate to smart_split_text in line_format."""
    return smart_split_text(text, limit)
