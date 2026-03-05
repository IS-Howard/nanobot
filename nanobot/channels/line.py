"""LINE channel implementation using Messaging API webhooks."""

import asyncio
import base64
import hashlib
import hmac
import json
from typing import Any

import httpx
from aiohttp import web
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import LineConfig

LINE_API_BASE = "https://api.line.me/v2/bot"
LINE_MAX_TEXT = 5000
LINE_MAX_MESSAGES_PER_PUSH = 5


class LineChannel(BaseChannel):
    """LINE channel using Messaging API with webhook receiver."""

    name = "line"

    def __init__(self, config: LineConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: LineConfig = config
        self._runner: web.AppRunner | None = None
        self._http: httpx.AsyncClient | None = None

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

        messages: list[dict[str, Any]] = []

        # Text messages (split if over 5000 chars)
        if msg.content:
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
            content = f"[{msg_type}]"
        else:
            content = f"[{msg_type}]"

        if not content:
            return

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

    # ── LINE API calls ────────────────────────────────────────────────

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.channel_access_token}",
        }

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
