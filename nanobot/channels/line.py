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


def _quick_reply() -> dict:
    """Quick Reply buttons for common actions."""
    return {"items": [
        {"type": "action", "action": {
            "type": "postback", "label": "Tool Mode",
            "data": "action=tool", "displayText": "/tool",
        }},
        {"type": "action", "action": {
            "type": "postback", "label": "Consolidate",
            "data": "action=consolidate", "displayText": "/consolidate",
        }},
        {"type": "action", "action": {
            "type": "postback", "label": "New Chat",
            "data": "action=new", "displayText": "/new",
        }},
    ]}


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
            flex_msg = _flex_tool_status(msg.content) or _flex_help(msg.content)

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

        # Attach Quick Reply buttons to the last message (non-progress only)
        if not is_progress and messages:
            messages[-1]["quickReply"] = _quick_reply()

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
        """Process a postback event (from Quick Reply buttons)."""
        data = event.get("postback", {}).get("data", "")
        action_map = {"tool": "/tool", "consolidate": "/consolidate", "new": "/new"}
        params = dict(p.split("=", 1) for p in data.split("&") if "=" in p)
        command = action_map.get(params.get("action", ""))
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
                f"{LINE_API_BASE}/chat/loading",
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
