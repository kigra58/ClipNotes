"""Speech-to-text transcription service powered by faster-whisper."""

import logging
import threading
from collections.abc import Callable
from typing import Any

from app.exceptions import TranscriptionError

logger = logging.getLogger(__name__)

_TARGET_SAMPLE_RATE = 16000


def chunk_windows(
    total_seconds: float,
    chunk_seconds: float,
    overlap_seconds: float,
) -> list[tuple[float, float]]:
    """Split a duration into overlapping transcription windows.

    Args:
        total_seconds: Total audio duration in seconds.
        chunk_seconds: Desired window length in seconds.
        overlap_seconds: Overlap between consecutive windows in seconds.

    Returns:
        A list of ``(start, end)`` windows covering the full duration. Each
        window after the first overlaps the previous one by ``overlap_seconds``
        so speech at boundaries is never cut off.
    """
    if total_seconds <= 0:
        return []
    if total_seconds <= chunk_seconds:
        return [(0.0, total_seconds)]
    step = max(chunk_seconds - overlap_seconds, 1.0)
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < total_seconds - 1e-6:
        end = min(start + chunk_seconds, total_seconds)
        windows.append((start, end))
        if end >= total_seconds - 1e-6:
            break
        start += step
    return windows


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

    def transcribe_stream(
        self,
        audio_path: str,
        on_progress: Callable[[str, float], None] | None = None,
        chunk_seconds: float = 60.0,
        overlap_seconds: float = 3.0,
    ) -> dict:
        """Transcribe a long audio file in overlapping chunks with progress.

        The audio is decoded to a mono 16 kHz float array once, then split
        into overlapping windows. Each window is transcribed separately (with
        the language detected on the first window reused for the rest, keeping
        the transcript consistent) and its timestamps are offset back into the
        full-file timeline. Segments that fall entirely inside an overlap
        region are dropped so adjacent windows do not duplicate text.

        Args:
            audio_path: Path to the audio file to transcribe.
            on_progress: Optional callback invoked as ``(message, percent)``
                while chunks are processed.
            chunk_seconds: Length of each transcription window in seconds.
            overlap_seconds: Overlap between consecutive windows in seconds.

        Returns:
            A dict with ``language``, ``language_probability``,
            ``segments`` and ``transcript`` keys.

        Raises:
            TranscriptionError: If transcription fails or no speech is found.
        """
        from faster_whisper.audio import decode_audio

        on_progress = on_progress or (lambda _message, _percent: None)
        try:
            audio = decode_audio(audio_path)
        except Exception as exc:
            logger.exception("Failed to decode audio file %s", audio_path)
            raise TranscriptionError() from exc

        total_seconds = len(audio) / _TARGET_SAMPLE_RATE
        if total_seconds <= 0:
            raise TranscriptionError(detail="No speech was detected in the audio.")

        windows = chunk_windows(total_seconds, chunk_seconds, overlap_seconds)
        total_chunks = len(windows)

        language: str | None = None
        language_probability = 0.0
        segments: list[dict[str, Any]] = []

        try:
            for index, (start, end) in enumerate(windows):
                first_sample = int(start * _TARGET_SAMPLE_RATE)
                last_sample = min(int(end * _TARGET_SAMPLE_RATE), len(audio))
                chunk = audio[first_sample:last_sample]
                if chunk.size == 0:
                    continue

                percent = round(5.0 + (index / total_chunks) * 85.0)
                on_progress(f"Transcribing chunk {index + 1} of {total_chunks}…", percent)

                segments_iter, info = self.model.transcribe(
                    chunk,
                    beam_size=5,
                    language=language,
                    without_timestamps=False,
                )
                if index == 0:
                    language = info.language
                    language_probability = float(info.language_probability)

                trim_before = start + overlap_seconds if index > 0 else start
                for segment in segments_iter:
                    seg_start = float(segment.start) + start
                    seg_end = float(segment.end) + start
                    if index > 0 and seg_end <= trim_before:
                        continue
                    segments.append(
                        {
                            "start": seg_start,
                            "end": seg_end,
                            "text": " ".join(str(segment.text).split()),
                        }
                    )
        except TranscriptionError:
            raise
        except Exception as exc:
            logger.exception("Streaming transcription failed for audio file %s", audio_path)
            raise TranscriptionError() from exc

        if not segments:
            raise TranscriptionError(detail="No speech was detected in the audio.")

        transcript = " ".join(segment["text"] for segment in segments)

        logger.info(
            "Streaming transcription completed (language=%s, probability=%.2f, "
            "segments=%d, chunks=%d)",
            language,
            language_probability,
            len(segments),
            total_chunks,
        )

        return {
            "language": language or "en",
            "language_probability": language_probability,
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
