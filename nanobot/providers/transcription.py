"""Voice transcription providers.

Two backends share the same ``async transcribe(file_path) -> str`` contract:

- :class:`GroqTranscriptionProvider` — Groq's hosted Whisper API (needs a key).
- :class:`FasterWhisperTranscriptionProvider` — local, offline transcription via
  `faster-whisper <https://github.com/SYSTRAN/faster-whisper>`_ (no network, no key).
"""

import asyncio
import importlib.util
import os
from pathlib import Path

import httpx
from loguru import logger


class GroqTranscriptionProvider:
    """
    Voice transcription provider using Groq's Whisper API.

    Groq offers extremely fast transcription with a generous free tier.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file using Groq.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text.
        """
        if not self.api_key:
            logger.warning("Groq API key not configured for transcription")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files = {
                        "file": (path.name, f),
                        "model": (None, "whisper-large-v3"),
                    }
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                    }

                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        files=files,
                        timeout=60.0
                    )

                    response.raise_for_status()
                    data = response.json()
                    return data.get("text", "")

        except Exception as e:
            logger.error("Groq transcription error: {}", e)
            return ""


# Loaded WhisperModel instances are cached across calls (loading is expensive).
# Keyed by (model, device, compute_type).
_WHISPER_MODEL_CACHE: dict[tuple[str, str, str], object] = {}


class FasterWhisperTranscriptionProvider:
    """
    Local voice transcription using faster-whisper (offline, no API key).

    Runs Whisper on-device via CTranslate2. The model is downloaded once from
    Hugging Face on first use and cached; subsequent runs are fully offline.

    Install with: ``pip install nanobot-ai[whisper]`` (or ``pip install faster-whisper``).
    """

    def __init__(
        self,
        model: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        cpu_threads: int = 0,
    ):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.language = language or None
        self.cpu_threads = cpu_threads or 0

    @staticmethod
    def is_available() -> bool:
        """True if the faster-whisper package is importable."""
        return importlib.util.find_spec("faster_whisper") is not None

    def _load_model(self):
        key = (self.model_name, self.device, self.compute_type, self.cpu_threads)
        cached = _WHISPER_MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        from faster_whisper import WhisperModel  # raises ImportError if not installed

        logger.info(
            "Loading faster-whisper model '{}' (device={}, compute_type={}, cpu_threads={})...",
            self.model_name, self.device, self.compute_type, self.cpu_threads or "default",
        )
        model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,  # 0 = ctranslate2 default
        )
        _WHISPER_MODEL_CACHE[key] = model
        return model

    def _transcribe_sync(self, path: Path) -> str:
        model = self._load_model()
        segments, info = model.transcribe(str(path), language=self.language)
        # segments is a generator; iterating runs the actual transcription.
        text = "".join(segment.text for segment in segments).strip()
        logger.info(
            "faster-whisper: transcribed {} ({} chars, lang={})",
            path.name, len(text), getattr(info, "language", "?"),
        )
        return text

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file locally with faster-whisper.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text (empty string on error / missing file).
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""
        try:
            # Model load + inference are CPU/GPU-bound; keep the event loop free.
            return await asyncio.to_thread(self._transcribe_sync, path)
        except ImportError:
            logger.error(
                "faster-whisper not installed; run: pip install nanobot-ai[whisper]"
            )
            return ""
        except Exception as e:
            logger.error("faster-whisper transcription error: {}", e)
            return ""
