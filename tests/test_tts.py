"""Tests for the text-to-speech feature (web page, Datastar action, JSON API)."""

import ast
import io
import os
import tempfile
import time
import wave
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.database import Database
from app.services.email import EmailService
from app.services.pipeline import run_transcription
from app.services.tts import TTSService

VIDEO_ID = "dQw4w9WgXcQ"
WATCH_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
EMAIL = "tts@example.com"
PASSWORD = "test-password-123"
OTHER_EMAIL = "other@example.com"


def _write_wav(path_or_buffer: str | io.BytesIO) -> None:
    """Write a minimal valid 16-bit mono WAV file."""
    with wave.open(path_or_buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(b"\x00\x00" * 1000)


class FakeYouTubeService:
    """In-memory stand-in for YouTubeService to avoid real downloads."""

    def get_metadata(self, youtube_url: str) -> dict:
        return {"video_id": VIDEO_ID, "title": "Fake Video", "duration": 60.0, "uploader": "Fake Channel"}

    def download_audio(self, youtube_url: str) -> dict:
        return {
            "video_id": VIDEO_ID,
            "title": "Fake Video",
            "duration": 60.0,
            "uploader": "Fake Channel",
            "audio_path": "temp/fake.mp3",
        }

    def cleanup(self, video_id: str) -> None:
        pass


class FakeTranscriptionService:
    """In-memory stand-in for TranscriptionService."""

    def transcribe(self, audio_path: str) -> dict:
        return {
            "language": "en",
            "language_probability": 0.98,
            "transcript": "Hello everyone.",
            "segments": [{"start": 0.0, "end": 4.5, "text": "Hello everyone."}],
        }


class FakeRAG:
    """In-memory stand-in for the chunking service."""

    def build_chunks(self, segments: list) -> list:
        return []


class FakeChat:
    """In-memory stand-in for ChatService with chat features disabled."""

    available = False


class FakeTTSService:
    """In-memory stand-in for TTSService that writes minimal WAV files."""

    def __init__(self) -> None:
        self.cache_dir = Path(tempfile.mkdtemp())
        self.voices_dir = Path(tempfile.mkdtemp())
        self.max_chars = 10000
        self.synthesis_timeout_seconds = 300
        self.available = True
        self.synthesized: list[str] = []
        self.current_voice = "en_US-lessac-medium"

    @property
    def voice(self) -> str:
        return self.current_voice

    def list_voices(self) -> list[dict]:
        return [
            {
                "name": self.current_voice,
                "label": "English (US) · Lessac Medium (default)",
                "default": True,
                "lang": "en",
            },
            {"name": "en_GB-alba-medium", "label": "English (GB) · Alba Medium", "default": False, "lang": "en"},
        ]

    def set_voice(self, name: str) -> None:
        if name not in (voice["name"] for voice in self.list_voices()):
            raise ValueError(f"Unknown voice '{name}'.")
        self.current_voice = name

    def _check(self, text: str) -> str:
        cleaned = " ".join(text.split())
        if not cleaned:
            raise ValueError("Text is empty.")
        if len(cleaned) > self.max_chars:
            raise ValueError(f"Text exceeds the {self.max_chars} character limit.")
        return cleaned

    def synthesize_to_file(self, text: str, out_path: Path) -> float:
        self.synthesized.append(self._check(text))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_wav(str(out_path))
        return 0.25

    def synthesize_bytes(self, text: str) -> tuple[bytes, float]:
        self.synthesized.append(self._check(text))
        buffer = io.BytesIO()
        _write_wav(buffer)
        return buffer.getvalue(), 0.25

    def cache_audio(self, text: str, user_id: int) -> Path:
        self.synthesized.append(self._check(text))
        out_path = self.cache_dir / f"{user_id}_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.wav"
        _write_wav(str(out_path))
        return out_path

    def cleanup_stale(self) -> int:
        return 0

    def to_mp3(self, wav_path: Path) -> Path:
        mp3_path = wav_path.with_suffix(".mp3")
        if not mp3_path.is_file():
            mp3_path.write_bytes(b"ID3fake-mp3")
        return mp3_path


@pytest.fixture()
def client() -> TestClient:
    """A test client wired with fake heavy services and a temp database."""
    db = Database(Path(tempfile.mkdtemp()) / "test.db")
    app.state.database = db
    app.state.youtube_service = FakeYouTubeService()
    app.state.transcription_service = FakeTranscriptionService()
    app.state.rag = FakeRAG()
    app.state.chat = FakeChat()
    app.state.email_service = EmailService(host="", port=465, username="", password="")
    app.state.pipeline = run_transcription
    app.state.tts_service = FakeTTSService()
    return TestClient(app)


def signup(client: TestClient, email: str = EMAIL) -> None:
    """Register a user; the client cookie jar now holds the JWT."""
    response = client.post(
        "/auth/signup",
        data={"email": email, "password": PASSWORD, "confirm": PASSWORD},
        follow_redirects=False,
    )
    assert response.status_code == 303


def transcribe_video(client: TestClient) -> int:
    """Create a stored video via the JSON API and return its id."""
    response = client.post("/api/v1/transcribe", json={"youtube_url": WATCH_URL})
    assert response.status_code == 200
    return response.json()["transcript_id"]


# ---------------------------------------------------------------------------
# TTSService voice selection
# ---------------------------------------------------------------------------


def _write_fake_model(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-onnx")


def test_tts_service_lists_available_voices(tmp_path: Path) -> None:
    """list_voices scans the voices dir and keeps the default model."""
    voices_dir = tmp_path / "voices"
    _write_fake_model(voices_dir / "en_US-lessac-medium.onnx")
    _write_fake_model(voices_dir / "en_GB-alba-medium.onnx")
    service = TTSService(
        voice_model=voices_dir / "en_US-lessac-medium.onnx",
        voices_dir=voices_dir,
        cache_dir=tmp_path / "cache",
        max_chars=10000,
    )

    voices = service.list_voices()
    names = [voice["name"] for voice in voices]
    assert names == ["en_GB-alba-medium", "en_US-lessac-medium"]
    default = next(voice for voice in voices if voice["name"] == "en_US-lessac-medium")
    assert default["default"] is True
    assert all(voice["label"] for voice in voices)
    assert all(voice["lang"] == voice["name"].split("_", 1)[0] for voice in voices)


def test_tts_service_includes_default_voice_outside_dir(tmp_path: Path) -> None:
    """The configured default model appears even when outside voices_dir."""
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    default_model = tmp_path / "elsewhere" / "en_US-amy-medium.onnx"
    _write_fake_model(default_model)
    service = TTSService(
        voice_model=default_model,
        voices_dir=voices_dir,
        cache_dir=tmp_path / "cache",
        max_chars=10000,
    )

    voices = service.list_voices()
    assert any(voice["name"] == "en_US-amy-medium" for voice in voices)


def test_tts_service_set_voice_switches_model(tmp_path: Path) -> None:
    """set_voice points the service at a different model file."""
    voices_dir = tmp_path / "voices"
    _write_fake_model(voices_dir / "en_US-lessac-medium.onnx")
    _write_fake_model(voices_dir / "en_GB-alba-medium.onnx")
    service = TTSService(
        voice_model=voices_dir / "en_US-lessac-medium.onnx",
        voices_dir=voices_dir,
        cache_dir=tmp_path / "cache",
        max_chars=10000,
    )

    service.set_voice("en_GB-alba-medium")
    assert service.voice == "en_GB-alba-medium"
    assert service.voice_model == voices_dir / "en_GB-alba-medium.onnx"


def test_tts_service_set_voice_rejects_unknown(tmp_path: Path) -> None:
    """An unknown voice name raises a ValueError listing available voices."""
    voices_dir = tmp_path / "voices"
    _write_fake_model(voices_dir / "en_US-lessac-medium.onnx")
    service = TTSService(
        voice_model=voices_dir / "en_US-lessac-medium.onnx",
        voices_dir=voices_dir,
        cache_dir=tmp_path / "cache",
        max_chars=10000,
    )

    with pytest.raises(ValueError, match="Unknown voice"):
        service.set_voice("does-not-exist")


def test_tts_service_to_mp3_reuses_cached(tmp_path: Path) -> None:
    """to_mp3 returns an existing cached MP3 without reconverting."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    wav = cache_dir / "1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.wav"
    _write_wav(str(wav))
    mp3 = wav.with_suffix(".mp3")
    mp3.write_bytes(b"cached")

    service = TTSService(
        voice_model=tmp_path / "voice.onnx",
        voices_dir=tmp_path / "voices",
        cache_dir=cache_dir,
        max_chars=10000,
    )
    assert service.to_mp3(wav) == mp3


def test_tts_service_to_mp3_requires_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """to_mp3 raises a clear error when ffmpeg is not installed."""
    monkeypatch.setattr("app.services.tts.shutil.which", lambda _: None)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    wav = cache_dir / "1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.wav"
    _write_wav(str(wav))

    service = TTSService(
        voice_model=tmp_path / "voice.onnx",
        voices_dir=tmp_path / "voices",
        cache_dir=cache_dir,
        max_chars=10000,
    )
    with pytest.raises(RuntimeError, match="ffmpeg"):
        service.to_mp3(wav)


def test_tts_service_cleanup_stale_removes_wav_and_mp3(tmp_path: Path) -> None:
    """cleanup_stale purges old WAV and MP3 cache files."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "old.wav").write_bytes(b"wav")
    (cache_dir / "old.mp3").write_bytes(b"mp3")
    fresh_wav = cache_dir / "fresh.wav"
    fresh_wav.write_bytes(b"wav")
    fresh_mp3 = cache_dir / "fresh.mp3"
    fresh_mp3.write_bytes(b"mp3")

    old = time.time() - 7200
    os.utime(cache_dir / "old.wav", (old, old))
    os.utime(cache_dir / "old.mp3", (old, old))

    service = TTSService(
        voice_model=tmp_path / "voice.onnx",
        voices_dir=tmp_path / "voices",
        cache_dir=cache_dir,
        max_chars=10000,
    )
    assert service.cleanup_stale(max_age_seconds=3600) == 2
    assert not (cache_dir / "old.wav").exists()
    assert not (cache_dir / "old.mp3").exists()
    assert (cache_dir / "fresh.wav").exists()
    assert (cache_dir / "fresh.mp3").exists()


class _SignalsParser(HTMLParser):
    """Extract the form's ``data-signals`` attribute (entities decoded)."""

    def __init__(self) -> None:
        super().__init__()
        self.value = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name == "data-signals":
                self.value = value


def parse_signals_attribute(html: str) -> dict:
    """Parse the form's data-signals attribute as a dict."""
    parser = _SignalsParser()
    parser.feed(html)
    assert parser.value is not None, "data-signals attribute not found"
    return ast.literal_eval(
        parser.value.replace("null", "None").replace("false", "False").replace("true", "True")
    )


# ---------------------------------------------------------------------------
# Speak page
# ---------------------------------------------------------------------------


def test_speak_page_redirects_anon(client: TestClient) -> None:
    """Anonymous visitors are redirected to the login page."""
    response = client.get("/speak", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers.get("location") == "/login"


def test_speak_page_renders(client: TestClient) -> None:
    """The speak page renders with the input form."""
    signup(client)
    response = client.get("/speak")
    assert response.status_code == 200
    assert "Speak a transcript" in response.text
    assert "tts_text" in response.text


def test_speak_page_skeleton_tied_to_speaking_signal(client: TestClient) -> None:
    """The results pane shows a skeleton while synthesizing, never a spinner."""
    signup(client)
    response = client.get("/speak")
    assert response.status_code == 200
    assert 'data-show="$speaking"' in response.text
    assert "speak-skeleton" in response.text
    assert 'data-show="$speaking && !$tts_error"' not in response.text


def test_speak_page_prefills_from_video(client: TestClient) -> None:
    """?video_id prefills the textarea with the stored transcript."""
    signup(client)
    video_id = transcribe_video(client)
    response = client.get(f"/speak?video_id={video_id}")
    assert response.status_code == 200
    assert "Hello everyone." in response.text


def test_speak_page_ignores_unknown_video(client: TestClient) -> None:
    """An unknown or foreign video id leaves the form empty."""
    signup(client)
    response = client.get("/speak?video_id=999")
    assert response.status_code == 200
    assert "Hello everyone." not in response.text


def test_speak_page_does_not_leak_other_users_transcript(client: TestClient) -> None:
    """A user cannot prefill another user's transcript via ?video_id."""
    signup(client)
    video_id = transcribe_video(client)
    signup(client, email=OTHER_EMAIL)
    response = client.get(f"/speak?video_id={video_id}")
    assert response.status_code == 200
    assert "Hello everyone." not in response.text
    assert "tts_text" in response.text


def test_speak_page_signals_attribute_parses(client: TestClient) -> None:
    """The data-signals attribute survives HTML parsing as valid signal values."""
    signup(client)
    video_id = transcribe_video(client)
    response = client.get(f"/speak?video_id={video_id}")
    assert response.status_code == 200
    signals = parse_signals_attribute(response.text)
    assert signals["tts_text"] == "Hello everyone."
    assert signals["speaking"] is False
    assert signals["tts_error"] == ""
    assert signals["tts_stats"] == {
        "characters": None,
        "words": None,
        "sentences": None,
        "estimated_duration_seconds": None,
    }


def test_speak_page_stats_subscribe_to_nested_paths(client: TestClient) -> None:
    """The stat cells must read nested signal paths so Datastar updates them."""
    signup(client)
    response = client.get("/speak")
    assert response.status_code == 200
    for path in (
        "$tts_stats.characters",
        "$tts_stats.words",
        "$tts_stats.sentences",
        "$tts_stats.estimated_duration_seconds",
    ):
        assert path in response.text
    assert "($tts_stats || {})" not in response.text


def test_speak_page_signals_escape_quotes_and_newlines(client: TestClient) -> None:
    """Transcripts with double quotes and newlines stay valid inside the attribute."""
    signup(client)
    user = app.state.database.get_user_by_email(EMAIL)
    tricky = 'He said "hi everyone."\nAnd then: "bye".'
    video_id = app.state.database.save_video(
        user_id=user["id"],
        youtube_id="tricky01",
        youtube_url=f"https://www.youtube.com/watch?v=tricky01",
        title="Tricky",
        uploader="Channel",
        language="en",
        language_probability=0.99,
        duration=60.0,
        transcript=tricky,
        segments=[{"start": 0.0, "end": 1.0, "text": tricky}],
        chunks=[],
    )
    response = client.get(f"/speak?video_id={video_id}")
    assert response.status_code == 200
    signals = parse_signals_attribute(response.text)
    assert signals["tts_text"] == tricky


def test_speak_page_global_helper_defined_before_datastar(client: TestClient) -> None:
    """speak.js must load before datastar.js so its global is set when Datastar evaluates."""
    signup(client)
    response = client.get("/speak")
    assert response.status_code == 200
    speak_pos = response.text.index("static/speak.js")
    datastar_pos = response.text.index("static/datastar.js")
    assert speak_pos < datastar_pos


def test_speak_page_reading_time_helper_defined_inline_before_modules(client: TestClient) -> None:
    """formatReadingTime is set by an inline (non-module) script in <head>.

    During SPA navigation Datastar morphs <head> before <body>, so a classic
    script runs before the speak-page elements are scanned. If the helper were
    only defined in speak.js (a module, loaded asynchronously) the reading-time
    data-text expression would throw during the initial scan and the bindings
    after it (the voice note) would never be installed.
    """
    signup(client)
    response = client.get("/speak")
    assert response.status_code == 200
    body = response.text
    inline_pos = body.index("window.formatReadingTime = window.formatReadingTime ||")
    first_module = body.index('type="module"')
    assert inline_pos < first_module
    assert "static/datastar.js" in body


def test_speak_page_lists_voice_options(client: TestClient) -> None:
    """The speak page renders a voice selector with every available voice."""
    signup(client)
    response = client.get("/speak")
    assert response.status_code == 200
    assert 'data-bind="voice"' in response.text
    assert "en_US-lessac-medium" in response.text
    assert "en_GB-alba-medium" in response.text


def test_speak_page_voice_options_carry_language(client: TestClient) -> None:
    """Voice options expose their language so auto-detection can pick a voice."""
    signup(client)
    response = client.get("/speak")
    assert response.status_code == 200
    assert 'data-lang="en"' in response.text
    assert 'data-default-voice="en_US-lessac-medium"' in response.text


def test_speak_page_voice_signal_defaults_to_current_voice(client: TestClient) -> None:
    """The voice signal defaults to the service's active voice."""
    signup(client)
    response = client.get("/speak")
    signals = parse_signals_attribute(response.text)
    assert signals["voice"] == "en_US-lessac-medium"
    assert signals["tts_voice"] is None


def test_speak_page_mp3_loading_signal_defaults_false(client: TestClient) -> None:
    """The MP3 download starts with loading disabled."""
    signup(client)
    response = client.get("/speak")
    signals = parse_signals_attribute(response.text)
    assert signals["tts_mp3_loading"] is False


def test_speak_page_mp3_button_has_loading_state(client: TestClient) -> None:
    """The download button swaps to a spinner and disables while converting."""
    signup(client)
    response = client.get("/speak")
    assert response.status_code == 200
    assert 'data-class:is-loading="$tts_mp3_loading"' in response.text
    assert 'data-attr:aria-disabled="$tts_mp3_loading"' in response.text
    assert "speak-download-spinner" in response.text
    assert "Download MP3" in response.text


def test_speak_page_mp3_button_sits_in_status_row(client: TestClient) -> None:
    """The download button is paired with the status text, shown once audio exists."""
    signup(client)
    response = client.get("/speak")
    assert response.status_code == 200
    row = response.text.index('class="speak-status-row"')
    button = response.text.index("speak-download-btn")
    assert button > row
    assert 'data-show="$tts_mp3_url"' in response.text
    assert "speak-player-actions" not in response.text


# ---------------------------------------------------------------------------
# Datastar action
# ---------------------------------------------------------------------------


def test_speak_action_synthesizes(client: TestClient) -> None:
    """A Datastar POST synthesizes audio and patches the stats + player URL."""
    signup(client)
    response = client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={"tts_text": "Hello world."},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "event: datastar-patch-signals" in body
    assert "tts_url" in body
    assert "tts_stats" in body
    assert "/tts/audio/" in body
    assert '"tts_stats":null' not in body


def test_speak_action_does_not_reset_stats_to_null(client: TestClient) -> None:
    """The synthesis-reset patch must not null out tts_stats (breaks signal updates)."""
    signup(client)
    response = client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={"tts_text": "Hello world."},
    )
    assert response.status_code == 200
    events = [e for e in response.text.split("\n\n") if e.strip()]
    assert len(events) == 2
    assert "tts_stats" not in events[0]
    assert '"characters":12' in events[1]


def test_speak_action_rejects_empty_text(client: TestClient) -> None:
    """An empty transcript produces a friendly error patch."""
    signup(client)
    response = client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={"tts_text": "   "},
    )
    assert response.status_code == 200
    assert "Enter some text to speak first." in response.text


def test_speak_action_uses_selected_voice(client: TestClient) -> None:
    """The chosen voice is applied before synthesis and reported back."""
    signup(client)
    tts = app.state.tts_service
    assert tts.voice == "en_US-lessac-medium"
    response = client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={"tts_text": "Hello world.", "voice": "en_GB-alba-medium"},
    )
    assert response.status_code == 200
    assert tts.voice == "en_GB-alba-medium"
    assert '"en_GB-alba-medium"' in response.text


def test_speak_action_rejects_unknown_voice(client: TestClient) -> None:
    """An unknown voice name produces a friendly error and no synthesis."""
    signup(client)
    tts = app.state.tts_service
    response = client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={"tts_text": "Hello world.", "voice": "does-not-exist"},
    )
    assert response.status_code == 200
    assert "Unknown voice" in response.text
    assert tts.synthesized == []


def test_speak_action_requires_auth(client: TestClient) -> None:
    """Anonymous Datastar actions swap to the login page via SSE (no reload)."""
    response = client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={"tts_text": "Hello"},
    )
    assert response.status_code == 200
    assert "history.pushState({}, '', \"/login\")" in response.text


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------


def test_api_tts_analyze_requires_auth(client: TestClient) -> None:
    """POST /api/v1/tts/analyze without a session returns 401."""
    response = client.post("/api/v1/tts/analyze", json={"text": "Hello."})
    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required."}


def test_api_tts_analyze_returns_stats(client: TestClient) -> None:
    """The analyze endpoint reports counts and an estimated reading time."""
    signup(client)
    response = client.post("/api/v1/tts/analyze", json={"text": "Hello world. This is a test."})
    assert response.status_code == 200
    payload = response.json()
    assert payload["characters"] == len("Hello world. This is a test.")
    assert payload["words"] == 6
    assert payload["sentences"] == 2
    assert payload["estimated_duration_seconds"] == pytest.approx(6 * 60.0 / 200)


def test_api_tts_speak_requires_auth(client: TestClient) -> None:
    """POST /api/v1/tts without a session returns 401."""
    response = client.post("/api/v1/tts", json={"text": "Hello."})
    assert response.status_code == 401


def test_api_tts_speak_returns_wav(client: TestClient) -> None:
    """Authenticated speak returns WAV audio with a duration header."""
    signup(client)
    response = client.post("/api/v1/tts", json={"text": "Hello world."})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")
    assert response.headers["x-duration-seconds"] == "0.25"


def test_api_tts_speak_rejects_too_long(client: TestClient) -> None:
    """Text over the configured limit is rejected with 400."""
    signup(client)
    response = client.post("/api/v1/tts", json={"text": "x" * 10001})
    assert response.status_code == 400
    assert "character limit" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Audio streaming
# ---------------------------------------------------------------------------


def test_audio_stream_requires_auth(client: TestClient) -> None:
    """GET /tts/audio/{name} without a session returns 401."""
    response = client.get("/tts/audio/1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.wav")
    assert response.status_code == 401


def test_audio_stream_serves_own_audio(client: TestClient) -> None:
    """A user can stream their own synthesized clip."""
    signup(client)
    client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={"tts_text": "Hello"},
    )
    response = client.get("/tts/audio/1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.wav")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")


def test_audio_stream_rejects_other_users(client: TestClient) -> None:
    """A user cannot stream another user's clip."""
    signup(client)
    client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={"tts_text": "Hello"},
    )
    signup(client, email=OTHER_EMAIL)
    response = client.get("/tts/audio/1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.wav")
    assert response.status_code == 404


def test_audio_stream_rejects_invalid_names(client: TestClient) -> None:
    """Names that do not match the cache pattern are rejected."""
    signup(client)
    for name in ("../../etc/passwd", "1_bad.wav", "1_zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz.wav"):
        response = client.get(f"/tts/audio/{name}")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# MP3 download
# ---------------------------------------------------------------------------


def test_mp3_download_requires_auth(client: TestClient) -> None:
    """GET /tts/audio/{name}/mp3 without a session returns 401."""
    response = client.get("/tts/audio/1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.wav/mp3")
    assert response.status_code == 401


def test_mp3_download_serves_own_audio(client: TestClient) -> None:
    """A user can download their own synthesized clip as MP3."""
    signup(client)
    client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={"tts_text": "Hello"},
    )
    response = client.get("/tts/audio/1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.wav/mp3")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["content-disposition"] == 'attachment; filename="transcript.mp3"'
    assert response.content.startswith(b"ID3")


def test_mp3_download_reuses_cached_file(client: TestClient) -> None:
    """A second download reuses the cached MP3 without reconverting."""
    signup(client)
    client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={"tts_text": "Hello"},
    )
    service = app.state.tts_service
    wav = service.cache_dir / "1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.wav"
    mp3 = wav.with_suffix(".mp3")
    assert not mp3.is_file()
    client.get("/tts/audio/1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.wav/mp3")
    assert mp3.is_file()

    mp3.write_bytes(b"ID3second")
    response = client.get("/tts/audio/1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.wav/mp3")
    assert response.content == b"ID3second"


def test_mp3_download_rejects_other_users(client: TestClient) -> None:
    """A user cannot download another user's clip."""
    signup(client)
    client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={"tts_text": "Hello"},
    )
    signup(client, email=OTHER_EMAIL)
    response = client.get("/tts/audio/1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.wav/mp3")
    assert response.status_code == 404


def test_mp3_download_rejects_invalid_names(client: TestClient) -> None:
    """Names that do not match the cache pattern are rejected."""
    signup(client)
    for name in ("../../etc/passwd", "1_bad.wav", "1_zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz.wav"):
        response = client.get(f"/tts/audio/{name}/mp3")
        assert response.status_code == 404


def test_speak_action_reports_mp3_url(client: TestClient) -> None:
    """The synthesis patch advertises the MP3 download URL."""
    signup(client)
    response = client.post(
        "/speak",
        headers={"Datastar-Request": "true"},
        json={"tts_text": "Hello world."},
    )
    assert response.status_code == 200
    assert "tts_mp3_url" in response.text
    assert "/tts/audio/1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.wav/mp3" in response.text
