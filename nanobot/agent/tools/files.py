"""File tools: get_file_info and analyze_file for stored media."""

from __future__ import annotations

import base64
import tempfile
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.providers.transcription import GroqTranscriptionProvider
    from nanobot.storage.postgres import PostgresStorage


class FileInfoTool(Tool):
    """Get info about the most recently uploaded file in the current session."""

    name = "get_file_info"
    description = "Get info about the most recently uploaded file in the current chat session."
    parameters = {
        "type": "object",
        "properties": {
            "file_type": {
                "type": "string",
                "enum": ["image", "audio", "video", "file"],
                "description": "Filter by file type (optional)",
            },
        },
    }

    def __init__(self, storage: PostgresStorage):
        self._storage = storage
        self._session_key = ""

    def set_context(self, session_key: str) -> None:
        self._session_key = session_key

    async def execute(self, file_type: str | None = None, **kwargs: Any) -> str:
        if not self._session_key:
            return "Error: No session context available."

        file = await self._storage.get_latest_file(self._session_key, file_type)
        if not file:
            kind = file_type or "files"
            return f"No {kind} found in current session."

        return (
            f"Latest file: type={file['file_type']}, mime={file['mime_type']}, "
            f"size={file['file_size_bytes']} bytes, uploaded={file['uploaded_at']}"
        )


class FileAnalysisTool(Tool):
    """Analyze the most recently uploaded file using vision or transcription."""

    name = "analyze_file"
    description = (
        "Analyze the most recently uploaded file (image or audio) in the current chat. "
        "For images, uses vision to describe or analyze. "
        "For audio, transcribes first then analyzes the text."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "What to analyze or look for in the file",
            },
            "file_type": {
                "type": "string",
                "enum": ["image", "audio"],
                "description": "Type of file to analyze (default: latest of any type)",
            },
        },
        "required": ["prompt"],
    }

    def __init__(
        self,
        storage: PostgresStorage,
        provider: LLMProvider,
        model: str,
        transcription: GroqTranscriptionProvider | None = None,
    ):
        self._storage = storage
        self._provider = provider
        self._model = model
        self._transcription = transcription
        self._session_key = ""

    def set_context(self, session_key: str) -> None:
        self._session_key = session_key

    async def execute(self, prompt: str, file_type: str | None = None, **kwargs: Any) -> str:
        if not self._session_key:
            return "Error: No session context available."

        file = await self._storage.get_latest_file(self._session_key, file_type)
        if not file:
            kind = file_type or "file"
            return f"No {kind} found in current session."

        mime: str = file["mime_type"]
        data: bytes = file["file_data"]

        if mime.startswith("image/"):
            return await self._analyze_image(data, mime, prompt)

        if mime.startswith("audio/"):
            return await self._analyze_audio(data, prompt)

        return f"Unsupported file type for analysis: {mime}"

    async def _analyze_image(self, data: bytes, mime: str, prompt: str) -> str:
        """Analyze image via multimodal LLM call."""
        try:
            b64 = base64.b64encode(data).decode()
            messages = [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]}]
            response = await self._provider.chat(
                messages=messages, model=self._model, max_tokens=2048,
            )
            return response.content or "Analysis complete but no content returned."
        except Exception as e:
            logger.error("Image analysis error: {}", e)
            return f"Error analyzing image: {e}"

    async def _analyze_audio(self, data: bytes, prompt: str) -> str:
        """Transcribe audio then analyze the text."""
        if not self._transcription:
            return "Audio transcription not configured (need GROQ_API_KEY)."

        try:
            # Write to temp file for Groq Whisper API
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tmp.write(data)
            tmp.close()

            text = await self._transcription.transcribe(tmp.name)
            if not text:
                return "Transcription produced no text."

            # Analyze transcript with LLM
            messages = [{"role": "user", "content": f"Audio transcript:\n{text}\n\n{prompt}"}]
            response = await self._provider.chat(
                messages=messages, model=self._model, max_tokens=2048,
            )
            return response.content or f"Transcript: {text}"
        except Exception as e:
            logger.error("Audio analysis error: {}", e)
            return f"Error analyzing audio: {e}"
