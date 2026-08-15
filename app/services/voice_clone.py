"""Local voice-cloning service built on Coqui XTTS.

Coqui XTTS is an optional, heavy dependency (PyTorch plus a ~1.8 GB model).
Like Piper, it is imported lazily: when the package or the model is missing
the service reports ``available == False`` and the UI degrades to a notice
instead of crashing.

XTTS does not train a model per voice. Instead it extracts a speaker
embedding from a reference clip (the ``reference_wav`` stored on a voice
profile) and conditions generation on that embedding, so "cloning" a
YouTuber is simply capturing a clean sample of their voice and saving it.
"""

import io
import logging
import subprocess
import threading
import uuid
import wave
from pathlib import Path
from typing import Any

try:  # pragma: no cover - depends on environment
    # coqui-tts imports ``isin_mps_friendly`` from transformers.pytorch_utils,
    # a helper that newer transformers (>= 4.46) removed. Restore it before
    # importing TTS so the XTTS stack works with a modern transformers install.
    import transformers.pytorch_utils as _transformers_pt_utils  # type: ignore

    if not hasattr(_transformers_pt_utils, "isin_mps_friendly"):

        def _isin_mps_friendly(elements, test_elements):
            import torch

            if elements.device.type == "mps":
                return elements.cpu().isin(test_elements).to(elements.device)
            return torch.isin(elements, test_elements)

        _transformers_pt_utils.isin_mps_friendly = _isin_mps_friendly
except ImportError:  # pragma: no cover - transformers not installed
    pass

try:  # pragma: no cover - depends on environment
    from TTS.api import TTS
except ImportError:  # pragma: no cover - depends on environment
    TTS = None

logger = logging.getLogger(__name__)


class VoiceCloneService:
    """Synthesize speech in a cloned voice using a Coqui XTTS model.

    The model is loaded lazily on first synthesis so the pages still render
    even when the package is missing.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        device: str = "cpu",
        max_chars: int = 5000,
        synthesis_timeout_seconds: int = 600,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.max_chars = max_chars
        self.synthesis_timeout_seconds = synthesis_timeout_seconds
        self._tts: Any = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        """True when the XTTS package is installed (model loads lazily)."""
        return TTS is not None

    def _load(self) -> Any:
        if self._tts is None:
            with self._lock:
                if self._tts is None:
                    if not self.available:
                        raise RuntimeError(
                            "Voice cloning requires the Coqui TTS package "
                            "(pip install coqui-tts)."
                        )
                    logger.info("Loading XTTS voice clone model %s", self.model_name)
                    self._tts = TTS(model_name=self.model_name, progress_bar=False).to(
                        self.device
                    )
        return self._tts

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(text.split())

    def _check(self, text: str) -> str:
        cleaned = self._clean(text)
        if not cleaned:
            raise ValueError("Text is empty.")
        if len(cleaned) > self.max_chars:
            raise ValueError(f"Text exceeds the {self.max_chars} character limit.")
        return cleaned

    def _profile_kwargs(self, profile: dict[str, Any]) -> dict[str, Any]:
        """Build the speaker-conditioning kwargs for a voice profile.

        ``speaker`` is used by coqui purely as a cache key for the generated
        embedding (the voice itself comes from ``speaker_wav``), so it is kept
        short and stable — a long value would be slugified into a filename that
        exceeds Windows' path length limit. The cache is written to our own
        directory rather than the model's ``voices/`` dir, keeping the
        registered-voices list clean.
        """
        reference_wav = Path(self.cache_dir) / profile["reference_wav"]
        return {
            "speaker_wav": str(reference_wav),
            "speaker": f"profile_{profile['id']}",
            "voice_dir": str(self.cache_dir / "xtts_voices"),
        }

    def synthesize_to_file(
        self, text: str, profile: dict[str, Any], out_path: Path
    ) -> float:
        """Synthesize ``text`` in ``profile``'s voice to a WAV file.

        Args:
            text: The text to speak.
            profile: A voice profile row from the database.
            out_path: Destination path for the ``.wav`` file.

        Returns:
            The audio duration in seconds.

        Raises:
            ValueError: If ``text`` is empty or exceeds ``max_chars``.
            RuntimeError: If the XTTS package is not installed.
        """
        cleaned = self._check(text)
        language = profile.get("language") or "en"
        tts = self._load()
        tts.tts_to_file(
            text=cleaned,
            language=language,
            file_path=str(out_path),
            **self._profile_kwargs(profile),
        )
        with wave.open(str(out_path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
        return round(frames / rate, 2)

    def synthesize_bytes(self, text: str, profile: dict[str, Any]) -> tuple[bytes, float]:
        """Synthesize ``text`` in ``profile``'s voice in memory.

        Returns:
            A ``(wav_bytes, duration_seconds)`` tuple.
        """
        cleaned = self._check(text)
        language = profile.get("language") or "en"
        tts = self._load()
        samples = tts.tts(text=cleaned, language=language, **self._profile_kwargs(profile))
        pcm = (samples.astype("float64") * 32767.0).clip(-32768, 32767).astype("int16")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(pcm.tobytes())
        duration = round(len(samples) / 24000, 2)
        return buffer.getvalue(), duration

    def cache_audio(
        self, text: str, profile: dict[str, Any], user_id: int, cache_dir: Path | None = None
    ) -> Path:
        """Synthesize ``text`` into a cached WAV file for the given user.

        Args:
            text: The text to speak.
            profile: A voice profile row from the database.
            user_id: The user who requested the clip (used for ownership).
            cache_dir: Destination directory; defaults to the clone directory.

        Returns:
            The path of the cached WAV file.
        """
        name = f"{user_id}_{uuid.uuid4().hex}.wav"
        out_path = Path(cache_dir or self.cache_dir) / name
        self.synthesize_to_file(text, profile, out_path)
        return out_path

    # ------------------------------------------------------------------
    # Voices registered through coqui-tts' own voice-registration API
    # (the ``.pth`` files the model stores in its ``voices/`` directory).
    # ------------------------------------------------------------------

    @staticmethod
    def _model_voices_dir(model_name: str) -> Path:
        """Resolve the directory where coqui-tts caches registered voices.

        Delegates to coqui's own path helper so the result always matches where
        the model actually lives (honoring ``TTS_HOME``/``XDG_DATA_HOME`` and the
        ``%LOCALAPPDATA%\\tts`` default on Windows), i.e.
        ``<data>/tts/<model_full_name>/voices``.
        """
        from TTS.utils.manage import get_user_data_dir

        return Path(get_user_data_dir("tts")) / model_name.replace("/", "--") / "voices"

    def list_registered_voices(self) -> list[str]:
        """Names of voices registered via coqui's voice-registration API.

        Each is a ``.pth`` speaker embedding saved in the model's ``voices/``
        directory (e.g. by the official XTTS demo). Returns ``[]`` when the
        package is not installed.
        """
        if not self.available:
            return []
        return sorted(p.stem for p in self._model_voices_dir(self.model_name).glob("*.pth"))

    def _registered_kwargs(self, voice_name: str) -> dict[str, Any]:
        """Speaker-conditioning kwargs for a registered voice."""
        return {"speaker": voice_name}

    def synthesize_registered_to_file(
        self, text: str, voice_name: str, out_path: Path
    ) -> float:
        """Synthesize ``text`` with a registered voice into a WAV file.

        Args:
            text: The text to speak.
            voice_name: The name of a registered voice (a ``.pth`` stem).
            out_path: Destination path for the ``.wav`` file.

        Returns:
            The audio duration in seconds.
        """
        cleaned = self._check(text)
        tts = self._load()
        tts.tts_to_file(
            text=cleaned,
            language="en",
            file_path=str(out_path),
            **self._registered_kwargs(voice_name),
        )
        with wave.open(str(out_path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
        return round(frames / rate, 2)

    def synthesize_registered_bytes(
        self, text: str, voice_name: str
    ) -> tuple[bytes, float]:
        """Synthesize ``text`` with a registered voice in memory.

        Returns:
            A ``(wav_bytes, duration_seconds)`` tuple.
        """
        cleaned = self._check(text)
        tts = self._load()
        samples = tts.tts(text=cleaned, language="en", **self._registered_kwargs(voice_name))
        pcm = (samples.astype("float64") * 32767.0).clip(-32768, 32767).astype("int16")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(pcm.tobytes())
        duration = round(len(samples) / 24000, 2)
        return buffer.getvalue(), duration

    def cache_registered_audio(
        self,
        text: str,
        voice_name: str,
        user_id: int,
        cache_dir: Path | None = None,
    ) -> Path:
        """Synthesize ``text`` with a registered voice into a cached WAV file."""
        name = f"{user_id}_{uuid.uuid4().hex}.wav"
        out_path = Path(cache_dir or self.cache_dir) / name
        self.synthesize_registered_to_file(text, voice_name, out_path)
        return out_path


def extract_reference_wav(source_path: Path, dest_path: Path, sample_rate: int = 24000) -> Path:
    """Re-encode a downloaded audio clip into a clean mono WAV reference.

    XTTS conditions on the reference clip, so a consistent 16-bit mono WAV at
    its native sample rate keeps the voice conditioning stable.

    Args:
        source_path: The downloaded audio file (any ffmpeg-supported codec).
        dest_path: Destination ``.wav`` path.
        sample_rate: Target sample rate in Hz.

    Returns:
        ``dest_path``.
    """
    ffmpeg = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-sample_fmt",
            "s16",
            str(dest_path),
        ],
        capture_output=True,
    )
    if ffmpeg.returncode != 0:
        stderr = ffmpeg.stderr.decode(errors="replace") or "unknown ffmpeg error"
        logger.error("Could not build reference WAV from %s: %s", source_path, stderr)
        raise RuntimeError("Could not extract the voice sample.")
    return dest_path


def pick_sample_window(
    segments: list[dict[str, Any]], target_seconds: float = 12.0
) -> tuple[float, float, str]:
    """Choose a clean, speech-dense window of ``segments`` for cloning.

    Picks the longest individual segment as a seed, then extends forwards and
    backwards through neighbours until the window reaches ``target_seconds``.
    The reference text is the transcript text overlapping that window, which
    keeps XTTS conditioning aligned with the audio it was extracted from.

    Args:
        segments: Transcript segments with ``start``, ``end`` and ``text``.
        target_seconds: Desired reference clip length in seconds.

    Returns:
        A ``(start, end, reference_text)`` tuple, or ``(0.0, 0.0, "")`` when
        there are no usable segments.
    """
    usable = [
        segment
        for segment in segments
        if segment.get("end", 0) - segment.get("start", 0) >= 0.5
        and (segment.get("text") or "").strip()
    ]
    if not usable:
        return (0.0, 0.0, "")

    seed = max(usable, key=lambda segment: segment["end"] - segment["start"])
    ordered = sorted(usable, key=lambda segment: segment["start"])
    start = seed["start"]
    end = seed["end"]

    changed = True
    while changed and (end - start) < target_seconds:
        changed = False
        for segment in ordered:
            if segment["start"] >= end and segment["start"] <= end + 2.0:
                end = segment["end"]
                changed = True
            elif segment["end"] <= start and segment["end"] >= start - 2.0:
                start = segment["start"]
                changed = True

    if end - start > target_seconds:
        end = start + target_seconds

    reference_text = " ".join(
        segment["text"].strip()
        for segment in ordered
        if segment["start"] < end and segment["end"] > start
    ).strip()
    return round(start, 2), round(end, 2), reference_text


def make_reference_clip(
    source_path: Path,
    cache_dir: Path,
    sample_rate: int = 24000,
) -> tuple[Path, str]:
    """Write a voice clone reference WAV and return its file name.

    The clip is named ``ref_<uuid>.wav`` and stored inside ``cache_dir``;
    only the file name is persisted on the profile row.

    Args:
        source_path: The downloaded audio clip to condition on.
        cache_dir: Directory that stores reference clips.

    Returns:
        A ``(wav_path, file_name)`` tuple.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = f"ref_{uuid.uuid4().hex}.wav"
    dest = cache_dir / name
    extract_reference_wav(source_path, dest, sample_rate)
    return dest, name
