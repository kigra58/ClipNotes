"""Piper text-to-speech service and transcript analysis helpers.

Piper is an optional dependency: when it (or a voice model file) is missing the
service reports ``available == False`` and the web UI degrades to a notice
instead of crashing.
"""

import io
import logging
import re
import time
import uuid
import wave
from pathlib import Path
from typing import Any

try:  # pragma: no cover - depends on environment
    from piper import PiperVoice
except ImportError:  # pragma: no cover - depends on environment
    PiperVoice = None

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|(?<=\n)\s*")

# Reading time estimate (words per minute).
READING_WORDS_PER_MINUTE = 200


def analyze_text(text: str) -> dict[str, Any]:
    """Analyze a transcript: counts and an estimated speaking duration.

    Args:
        text: The transcript text.

    Returns:
        A dict with ``characters``, ``words``, ``sentences`` and
        ``estimated_duration_seconds`` (based on a 200 wpm reading rate).
    """
    words = text.split()
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s]
    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len(sentences),
        "estimated_duration_seconds": round(len(words) * 60.0 / READING_WORDS_PER_MINUTE, 1),
    }


class TTSService:
    """Synthesize speech from text using a Piper voice model.

    The model is loaded lazily on first synthesis so the pages still render
    even when the model file is missing or Piper is not installed.
    """

    def __init__(
        self,
        voice_model: Path,
        cache_dir: Path,
        max_chars: int,
        synthesis_timeout_seconds: int = 300,
    ) -> None:
        self.voice_model = Path(voice_model)
        self.cache_dir = Path(cache_dir)
        self.max_chars = max_chars
        self.synthesis_timeout_seconds = synthesis_timeout_seconds
        self._voice: Any = None
        self._sample_rate = 22050

    @property
    def available(self) -> bool:
        """True when Piper is importable and a voice model file exists."""
        return PiperVoice is not None and self.voice_model.is_file()

    def _load(self) -> Any:
        if self._voice is None:
            if not self.available:
                raise RuntimeError("TTS voice model is not available.")
            logger.info("Loading Piper voice model %s", self.voice_model.name)
            self._voice = PiperVoice.load(self.voice_model)
            self._sample_rate = self._voice.config.sample_rate
        return self._voice

    def _clean(self, text: str) -> str:
        return " ".join(text.split())

    def _check(self, text: str) -> str:
        cleaned = self._clean(text)
        if not cleaned:
            raise ValueError("Text is empty.")
        if len(cleaned) > self.max_chars:
            raise ValueError(f"Text exceeds the {self.max_chars} character limit.")
        return cleaned

    @staticmethod
    def _write_wav(frames: list[bytes], sample_rate: int, out: Any) -> float:
        """Write 16-bit mono PCM frames to a WAV file object.

        Args:
            frames: Raw int16 little-endian sample chunks.
            sample_rate: Sample rate of the audio.
            out: A file-like object (path via ``wave.open``, or a BytesIO).

        Returns:
            The audio duration in seconds.
        """
        total_bytes = sum(len(frame) for frame in frames)
        with wave.open(out, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(b"".join(frames))
        return round(total_bytes / (sample_rate * 2), 2)

    def synthesize_to_file(self, text: str, out_path: Path) -> float:
        """Synthesize ``text`` to a 16-bit mono WAV file.

        Args:
            text: The text to speak.
            out_path: Destination path for the ``.wav`` file.

        Returns:
            The audio duration in seconds.

        Raises:
            ValueError: If ``text`` is empty or exceeds ``max_chars``.
            RuntimeError: If the voice model cannot be loaded.
        """
        cleaned = self._check(text)
        voice = self._load()
        frames = [chunk.audio_int16_bytes for chunk in voice.synthesize(cleaned)]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return self._write_wav(frames, self._sample_rate, str(out_path))

    def synthesize_bytes(self, text: str) -> tuple[bytes, float]:
        """Synthesize ``text`` in memory.

        Args:
            text: The text to speak.

        Returns:
            A ``(wav_bytes, duration_seconds)`` tuple.

        Raises:
            ValueError: If ``text`` is empty or exceeds ``max_chars``.
            RuntimeError: If the voice model cannot be loaded.
        """
        cleaned = self._check(text)
        voice = self._load()
        frames = [chunk.audio_int16_bytes for chunk in voice.synthesize(cleaned)]
        buffer = io.BytesIO()
        duration = self._write_wav(frames, self._sample_rate, buffer)
        return buffer.getvalue(), duration

    def cache_audio(self, text: str, user_id: int) -> Path:
        """Synthesize ``text`` into a cached WAV file for the given user.

        Args:
            text: The text to speak.
            user_id: The user who requested the clip (used for ownership).

        Returns:
            The path of the cached WAV file.
        """
        name = f"{user_id}_{uuid.uuid4().hex}.wav"
        out_path = self.cache_dir / name
        self.synthesize_to_file(text, out_path)
        return out_path

    def cleanup_stale(self, max_age_seconds: int = 3600) -> int:
        """Remove cached WAV files older than ``max_age_seconds``.

        Args:
            max_age_seconds: Files older than this are deleted.

        Returns:
            The number of files removed.
        """
        if not self.cache_dir.is_dir():
            return 0
        cutoff = time.time() - max_age_seconds
        removed = 0
        for path in self.cache_dir.glob("*.wav"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
                    removed += 1
            except OSError:  # pragma: no cover - best-effort cleanup
                logger.debug("Could not remove stale TTS file %s", path)
        return removed
