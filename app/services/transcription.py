"""Speech-to-text transcription service powered by faster-whisper."""

import logging
import threading
from typing import Any

from app.exceptions import TranscriptionError

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Transcribes audio files and reuses a single in-memory Whisper model."""

    def __init__(self, model_name: str, device: str, compute_type: str) -> None:
        """Initialize the service configuration (model loads lazily once).

        Args:
            model_name: Name of the Whisper model to load.
            device: Inference device, e.g. ``"cpu"`` or ``"cuda"``.
            compute_type: Precision, e.g. ``"int8"`` or ``"float16"``.
        """
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._model: Any = None
        self._load_lock = threading.Lock()

    def load_model(self) -> None:
        """Load the Whisper model into memory if it is not loaded yet.

        Thread-safe: concurrent callers block until the model is ready.
        """
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            from faster_whisper import WhisperModel

            logger.info(
                "Loading Whisper model '%s' on device '%s' (compute_type=%s)",
                self._model_name,
                self._device,
                self._compute_type,
            )
            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
            logger.info("Whisper model loaded")

    @property
    def model(self) -> Any:
        """Return the lazily-loaded Whisper model.

        Returns:
            The loaded ``WhisperModel`` instance.
        """
        self.load_model()
        return self._model

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe an audio file and return language + segments.

        Args:
            audio_path: Path to the audio file to transcribe.

        Returns:
            A dict with ``language``, ``language_probability``,
            ``segments`` and ``transcript`` keys.

        Raises:
            TranscriptionError: If transcription fails.
        """
        try:
            segments_iter, info = self.model.transcribe(audio_path, beam_size=5)
            segments = [self._clean_segment(segment) for segment in segments_iter]
        except Exception as exc:
            logger.exception("Transcription failed for audio file %s", audio_path)
            raise TranscriptionError() from exc

        if not segments:
            raise TranscriptionError(detail="No speech was detected in the audio.")

        transcript = " ".join(segment["text"] for segment in segments)

        logger.info(
            "Transcription completed (language=%s, probability=%.2f, segments=%d)",
            info.language,
            info.language_probability,
            len(segments),
        )

        return {
            "language": info.language,
            "language_probability": float(info.language_probability),
            "segments": segments,
            "transcript": transcript,
        }

    def _clean_segment(self, segment: Any) -> dict:
        """Convert a faster-whisper segment into a cleaned dict.

        Args:
            segment: A faster-whisper segment object.

        Returns:
            A dict with ``start``, ``end`` and normalized ``text``.
        """
        return {
            "start": float(segment.start),
            "end": float(segment.end),
            "text": " ".join(str(segment.text).split()),
        }
