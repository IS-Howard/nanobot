"""LINE channel implementation using Messaging API webhooks."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import mimetypes
import tempfile
from typing import TYPE_CHECKING, Any

import httpx
from aiohttp import web
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import LineConfig

if TYPE_CHECKING:
    from nanobot.storage.postgres import PostgresStorage

LINE_API_BASE = "https://api.line.me/v2/bot"
LINE_DATA_API = "https://api-data.line.me/v2/bot"
LINE_MAX_TEXT = 5000
LINE_MAX_MESSAGES_PER_PUSH = 5

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
) -> bytes:
    """Render a Rich Menu PNG with labeled colored cells arranged in rows.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        rows: List of rows, each row is a list of (label, (r, g, b)) cells.
              Cells in each row are evenly divided across the width.
    """
    import struct
    import zlib

    scale = 8   # each font pixel = 8x8 real pixels
    gap = 4     # px separator between cells

    pixels = bytearray(width * height * 3)
    num_rows = len(rows)
    row_height = height // num_rows

    for ri, row in enumerate(rows):
        y0 = ri * row_height
        y1 = y0 + row_height if ri < num_rows - 1 else height
        cell_width = width // len(row)

        for ci, (label, (r, g, b)) in enumerate(row):
            x0 = ci * cell_width
            x1 = x0 + cell_width if ci < len(row) - 1 else width

            # Fill cell background with separators
            for y in range(y0, y1):
                for x in range(x0, x1):
                    is_sep = ((x < x0 + gap // 2 and ci > 0)
                              or (y < y0 + gap // 2 and ri > 0))
                    if is_sep:
                        pixels[(y * width + x) * 3:(y * width + x) * 3 + 3] = b"\xff\xff\xff"
                    else:
                        pixels[(y * width + x) * 3:(y * width + x) * 3 + 3] = bytes((r, g, b))

            # Render label centered in cell
            char_w = 5 * scale + scale
            text_w = len(label) * char_w - scale
            text_h = 7 * scale
            tx = x0 + (x1 - x0 - text_w) // 2
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
    ):
        super().__init__(config, bus)
        self.config: LineConfig = config
        self._runner: web.AppRunner | None = None
        self._http: httpx.AsyncClient | None = None
        self._storage = storage
        self._max_files = max_files_per_session
        self._rich_menu_admin: str | None = None
        self._rich_menu_normal: str | None = None

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
        """Send a message to LINE via push message API."""
        if not self._http:
            logger.warning("LINE client not running")
            return

        is_progress = msg.metadata.get("_progress", False)
        messages: list[dict[str, Any]] = []

        # Check if content should be upgraded to Flex Message
        flex_msg = None
        if msg.content and not is_progress:
            if msg.metadata.get("_admin_panel"):
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
            for chunk in _split_text(msg.content, LINE_MAX_TEXT):
                messages.append({"type": "text", "text": chunk})

        # Media messages
        for media_url in msg.media or []:
            messages.append({
                "type": "image",
                "originalContentUrl": media_url,
                "previewImageUrl": media_url,
            })

        if not messages:
            return

        # LINE allows max 5 messages per push call
        for i in range(0, len(messages), LINE_MAX_MESSAGES_PER_PUSH):
            batch = messages[i : i + LINE_MAX_MESSAGES_PER_PUSH]
            await self._push_messages(msg.chat_id, batch)

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

        # Show loading animation (fire-and-forget)
        if msg_type == "text":
            asyncio.create_task(self._show_loading(chat_id))

        reply_token = event.get("replyToken", "")

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
            "admin_panel": "/admin panel",
        }
        params = dict(p.split("=", 1) for p in data.split("&") if "=" in p)
        action = params.get("action", "")
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

        await self._handle_message(
            sender_id=sender_id,
            chat_id=chat_id,
            content=command,
            metadata={"line": {"source_type": source_type}},
        )

    async def _setup_rich_menus(self) -> None:
        """Create admin and normal Rich Menus via LINE API."""
        if not self._http:
            return
        try:
            # Delete existing rich menus to avoid accumulation
            resp = await self._http.get(
                f"{LINE_API_BASE}/richmenu/list",
                headers=self._auth_headers,
            )
            if resp.status_code == 200:
                for rm in resp.json().get("richmenus", []):
                    name = rm.get("name", "")
                    if name in ("nanobot_admin", "nanobot_normal"):
                        await self._http.delete(
                            f"{LINE_API_BASE}/richmenu/{rm['richMenuId']}",
                            headers=self._auth_headers,
                        )

            # Admin Rich Menu: 2x3 grid (2500x1686)
            # Row 1: Admin | Tool Mode | New Chat
            # Row 2: Consolidate | Cleanup | Help
            hw = 833   # cell width (2500/3)
            hh = 843   # cell height (1686/2)
            admin_menu = {
                "size": {"width": 2500, "height": 1686},
                "selected": True,
                "name": "nanobot_admin",
                "chatBarText": "Menu",
                "areas": [
                    {"bounds": {"x": 0, "y": 0, "width": hw, "height": hh},
                     "action": {"type": "postback", "label": "Admin", "data": "action=admin_panel",
                                "displayText": "/admin panel"}},
                    {"bounds": {"x": hw, "y": 0, "width": hw + 1, "height": hh},
                     "action": {"type": "postback", "label": "Tool Mode", "data": "action=tool",
                                "displayText": "/tool"}},
                    {"bounds": {"x": hw * 2, "y": 0, "width": hw + 1, "height": hh},
                     "action": {"type": "postback", "label": "New Chat", "data": "action=new",
                                "displayText": "/new"}},
                    {"bounds": {"x": 0, "y": hh, "width": hw, "height": hh},
                     "action": {"type": "postback", "label": "Consolidate", "data": "action=consolidate",
                                "displayText": "/consolidate"}},
                    {"bounds": {"x": hw, "y": hh, "width": hw + 1, "height": hh},
                     "action": {"type": "postback", "label": "Cleanup", "data": "action=cleanup",
                                "displayText": "/cleanup"}},
                    {"bounds": {"x": hw * 2, "y": hh, "width": hw + 1, "height": hh},
                     "action": {"type": "message", "label": "Help", "text": "/help"}},
                ],
            }
            resp = await self._http.post(
                f"{LINE_API_BASE}/richmenu",
                headers=self._auth_headers,
                json=admin_menu,
            )
            if resp.status_code == 200:
                self._rich_menu_admin = resp.json().get("richMenuId")
                await self._upload_rich_menu_image(self._rich_menu_admin, [
                    [("Admin", (30, 120, 70)), ("Tool Mode", (50, 90, 160)), ("New Chat", (80, 80, 90))],
                    [("Consolidate", (120, 90, 40)), ("Cleanup", (140, 60, 60)), ("Help", (60, 60, 80))],
                ])
                logger.info("Created admin Rich Menu: {}", self._rich_menu_admin)

            # Normal Rich Menu: 2x2 grid (2500x1686)
            # Row 1: New Chat | Help
            # Row 2: Consolidate | Cleanup
            nhw = 1250  # cell width (2500/2)
            normal_menu = {
                "size": {"width": 2500, "height": 1686},
                "selected": True,
                "name": "nanobot_normal",
                "chatBarText": "Menu",
                "areas": [
                    {"bounds": {"x": 0, "y": 0, "width": nhw, "height": hh},
                     "action": {"type": "postback", "label": "New Chat", "data": "action=new",
                                "displayText": "/new"}},
                    {"bounds": {"x": nhw, "y": 0, "width": nhw, "height": hh},
                     "action": {"type": "message", "label": "Help", "text": "/help"}},
                    {"bounds": {"x": 0, "y": hh, "width": nhw, "height": hh},
                     "action": {"type": "postback", "label": "Consolidate", "data": "action=consolidate",
                                "displayText": "/consolidate"}},
                    {"bounds": {"x": nhw, "y": hh, "width": nhw, "height": hh},
                     "action": {"type": "postback", "label": "Cleanup", "data": "action=cleanup",
                                "displayText": "/cleanup"}},
                ],
            }
            resp = await self._http.post(
                f"{LINE_API_BASE}/richmenu",
                headers=self._auth_headers,
                json=normal_menu,
            )
            if resp.status_code == 200:
                self._rich_menu_normal = resp.json().get("richMenuId")
                await self._upload_rich_menu_image(self._rich_menu_normal, [
                    [("New Chat", (50, 90, 160)), ("Help", (60, 60, 80))],
                    [("Consolidate", (120, 90, 40)), ("Cleanup", (140, 60, 60))],
                ])
                # Set as default for all users
                await self._http.post(
                    f"{LINE_API_BASE}/user/all/richmenu/{self._rich_menu_normal}",
                    headers=self._auth_headers,
                )
                logger.info("Created normal Rich Menu (default): {}", self._rich_menu_normal)

        except Exception as e:
            logger.warning("Failed to setup Rich Menus: {}", e)

    async def _upload_rich_menu_image(
        self, menu_id: str, rows: list[list[tuple[str, tuple[int, int, int]]]],
    ) -> None:
        """Upload a PNG image with labeled colored cells for a Rich Menu.

        Args:
            menu_id: Rich Menu ID to upload to.
            rows: List of rows, each row is a list of (label, (r, g, b)) cells.
        """
        if not self._http or not menu_id:
            return
        try:
            img_height = 843 * len(rows)
            png = _render_rich_menu_png(2500, img_height, rows)
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
        """Show LINE loading animation when agent is busy."""
        logger.debug("LINE show_busy for chat_id={}", chat_id)
        await self._show_loading(chat_id)

    async def _show_loading(self, chat_id: str) -> None:
        """Show loading animation in LINE chat."""
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

    async def _push_messages(self, to: str, messages: list[dict[str, Any]]) -> None:
        """Push messages to a user/group via LINE Messaging API."""
        if not self._http:
            return
        try:
            resp = await self._http.post(
                f"{LINE_API_BASE}/message/push",
                headers=self._auth_headers,
                json={"to": to, "messages": messages},
            )
            if resp.status_code != 200:
                logger.error("LINE push failed ({}): {}", resp.status_code, resp.text)
        except Exception as e:
            logger.error("LINE push error: {}", e)


def _split_text(text: str, limit: int = LINE_MAX_TEXT) -> list[str]:
    """Split text into chunks respecting the LINE character limit."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # Try to split at last newline before limit
        idx = text.rfind("\n", 0, limit)
        if idx <= 0:
            idx = limit
        chunks.append(text[:idx])
        text = text[idx:].lstrip("\n")
    return chunks
